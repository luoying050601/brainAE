import os
Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
import sys

sys.path.append(Proj_dir)
# print(sys.path)
import json
import torch
# import scipy.io as scio
import numpy as np
from src.com.util.data_format import normalization
from transformers import BertModel, BertTokenizer
from src.com.model.run_auto_encoder_coco import Autoencoder
from torch import nn

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True)
PROJ_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
# /Storage/ying/project/brainAE/output/is_1024_vanilla_0.013152188140832419.bin
# /Storage/ying/project/brainAE/output/is_1024_vanilla_2.332835287281445.bin
model_file = '/Storage/ying/project/brainAE/output/is_1024_vanilla_2.332835287281445.bin'

# 这里调整模型
dataset_name = "COCO2014"

def get_sentence_embedding(sentence, option):
    # if YOU NEED [cls] and [seq] PRESENTATION：True
    input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = bert_model(input_ids)

    all_layers_output = outputs[2]
    # list of torch.FloatTensor (one for the output of each layer + the output of the embeddings) of shape (batch_size, sequence_length, hidden_size): Hidden-states of the model at the output of each layer plus the initial embedding outputs.

    if option == "last_layer":
        sent_embeddings = all_layers_output[-1]  # last layer
    elif option == "second_to_last_layer":
        sent_embeddings = all_layers_output[-2]  # second to last layer
    else:
        sent_embeddings = all_layers_output[-1]  # last layer

    sent_embeddings = torch.squeeze(sent_embeddings, dim=0)
    # print(sent_embeddings.shape)

    # Calculate the average of all token vectors.
    sentence_embedding_avg = torch.mean(sent_embeddings, dim=0)
    # print(sentence_embedding_avg.shape)

    return sent_embeddings.detach(), sentence_embedding_avg.detach()


def load_brain_data():
    coco_brain_ROI = json.loads(
        open("/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/coco_brain_ROI.json", 'r', encoding='utf-8').read())
    return coco_brain_ROI


def calculate_corrcoef(dataset):

    participants = ['CSI1','CSI2','CSI3','CSI4']
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # sentence1_tsv_path = '../../../resource/text_807.tsv'
    # sentence2_tsv_path = '../../../resource/text_564.tsv'
    # sentence3_tsv_path = '../../../resource/text_423.tsv'
    # sentence4_tsv_path = '../../../resource/text_180.tsv'
    # # join exp1, 2and 3
    # type1_participants = ['P01', 'M02', 'M04', 'M07', 'M15']
    # # join exp1 and 3
    # type2_participants = ['M08', 'M09', 'M14']
    # # join exp1 and 3
    # type3_participants = ['M03']
    # # only join exp1
    # type4_participants = ['M01', 'M05', 'M06', 'M10', 'M13', 'M16', 'M17']
    pre_trained_model = Autoencoder()
    pre_trained_model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(os.path.join(model_file)).items()})

    pre_trained_model.to(device)

    corr_user = {}
    # norm_path = '/Storage/ying/resources/BrainBertTorch/dataset/' + dataset + '/norm/'

    brain_data = load_brain_data()
    for user in participants:
        if user == 'CSI1':
            key = 'CSI01'
        if user == 'CSI2':
            key = 'CSI02'
        if user == 'CSI3':
            key = 'CSI03'
        if user == 'CSI4':
            key = 'CSI04'
        label_list = open("/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/caption/" + key + "_captioning.txt",
                          'r').readlines()
        # sentence_df_path = '/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/caption/' +key +'_captioning.txt'
        brain_original = np.zeros((np.array(brain_data[user]).shape[0], 3104))
        brain_original[:, :np.array(brain_data[user]).shape[1]] = np.array(brain_data[user])
        train_X = np.array(brain_original)
        train_X, _, _ = normalization(train_X)
        train_X = torch.Tensor(train_X)
        train_X = train_X.to(device, dtype=torch.float)
        y, z = pre_trained_model(train_X)
        corr_list = []
        # feature_shape = z[0].shape[0]

        # sentence_df = pd.read_csv(sentence_df_path, sep='\t', header=None)
        # sentence_df.columns = ['text', 'type']
        for index in range((z.shape[0])):
            sentence = label_list[index]
            sent_embeddings, t = get_sentence_embedding(sentence, "last_layer")
            # z[index, :].cpu().detach().numpy(), t.cpu().detach().numpy()
            corr = np.corrcoef(z[index, :].cpu().detach().numpy(), t.cpu().detach().numpy())[0, 1]
            corr_list.append(corr)
        corr_user[user] = corr_list
    with open(dataset+f'_corr_user.json','w') as f:
        json.dump(corr_user, f)
        f.close()

def create_brain_npz(dataset):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    pre_trained_model = Autoencoder()
    pre_trained_model = nn.DataParallel(pre_trained_model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])  # multi-GPU

    checkpoint = torch.load(os.path.join(model_file))
    pre_trained_model.load_state_dict(checkpoint)

    pre_trained_model.to(device)

    norm_path = '/Storage/ying/resources/BrainBertTorch/dataset/' + dataset + '/norm/'
    feature_path = '/Storage/ying/resources/BrainBertTorch/dataset/' + dataset + '/features/'
    brain_data = load_brain_data()
    for user in brain_data.keys():
        brain_original = np.zeros((np.array(brain_data[user]).shape[0], 3104))
        brain_original[:, :np.array(brain_data[user]).shape[1]] = np.array(brain_data[user])
        train_X = np.array(brain_original)
        train_X, _, _ = normalization(train_X)
        train_X = torch.Tensor(train_X)
        train_X = train_X.to(device, dtype=torch.float)
        y, z = pre_trained_model(train_X)

        for index in range(brain_original.shape[0]):
            if user == 'CSI1':
                user = 'CSI01'
            if user == 'CSI2':
                user = 'CSI02'
            if user == 'CSI3':
                user = 'CSI03'
            if user == 'CSI4':
                user = 'CSI04'
            npz_filename = dataset_name + '_' + user + '_' + str(index) + '.npz'
            print(npz_filename)
            np.savez(norm_path + npz_filename, features=brain_original[index,:])
            np.savez(feature_path + npz_filename, features=z[index].cpu().data.numpy().tolist())

if __name__ == "__main__":
    # dataset = 'pereira'
    create_brain_npz(dataset_name)

    # print(features['data'])
    calculate_corrcoef('coco')
