import os
Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
import sys
import time

sys.path.append(Proj_dir)
# print(sys.path)
import json
import torch
import scipy.io as scio
import numpy as np
from src.com.util.data_format import normalization,make_print_to_file
from transformers import BertModel, BertTokenizer
from src.com.model.run_auto_encoder import Autoencoder
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True)
PROJ_DIR = os.path.abspath(os.path.join(os.getcwd(), "."))
# 这里调整模型
# 768
participants = {'18', '22', '23', '24', '26', '28', '30', '31', '35', '36', '37',
                      '39', '41', '42', '43', '44', '45', '47', '48', '49', '50', '51', '53'}

def get_sentence_embedding(sentence, option,z_size):
    if z_size == 1024:
        bert_version = 'bert-large-uncased'
    elif z_size == 768:
        bert_version = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    bert_model = BertModel.from_pretrained(bert_version, output_hidden_states=True)
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


def load_brain_data(dataset, user):
    if dataset == 'pereira':
        paticipants_1 = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07',
                         'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17', 'P01']
        paticipants_2 = ['M02', 'M04', 'M07', 'M08', 'M09', 'M14', 'M15', 'P01']
        paticipants_3 = ['M02', 'M03', 'M04', 'M07', 'M15', 'P01']
        a = []
        # dataset_path = "/Storage/ying/resources/pereira2018/'+user+'/data_180concepts_wordclouds.mat"
        if user in paticipants_1:
            exp1_path = '/Storage/ying/resources/pereira2018/' + user + '/data_180concepts_wordclouds.mat'
            exp1_data = scio.loadmat(exp1_path)
            examples = exp1_data['examples']
            # print(user, examples.shape)
            ROI_path = '../../../resource/' + user + '_roi.mat'
            data = scio.loadmat(ROI_path)
            roi = data['index']

            for i in range((examples.shape[0])):
               b = []
               for index in roi[0]:
                    b.append(examples[i][index])
               a.append(b[:46840])
            if user in paticipants_2:
                exp2_path = '/Storage/ying/resources/pereira2018/' + user + '/data_384sentences.mat'
                exp2_data = scio.loadmat(exp2_path)
                examples = exp2_data['examples_passagesentences']
                # b = []
                for i in range((examples.shape[0])):

                    b = []
                    for index in roi[0]:
                        b.append(examples[i][index])

                    a.append(b[:46840])
            if user in paticipants_3:
                exp3_path = '/Storage/ying/resources/pereira2018/' + user + '/data_243sentences.mat'
                exp3_data = scio.loadmat(exp3_path)
                # print(exp3_data.keys())
                examples = exp3_data['examples_passagesentences']
                for i in range((examples.shape[0])):

                    b = []
                    for index in roi[0]:
                        b.append(examples[i][index])
                    a.append(b[:46840])
    elif dataset == 'alice_ae' or dataset == 'alice':
        a = []
        brain_path = '/Storage/ying/resources/BrainBertTorch/brain/alice/ae_npy/'
        files = os.listdir(brain_path)
        for file in files:  # 遍历文件夹
            file_spilt = file.replace('.npy', '').split('_')
            sub = file_spilt[1]
            if sub == user:
                brain_data = np.load(brain_path + file)
                # print(user, brain_data.shape)
                a.append(brain_data[:46840])
    return a


def create_brain_npz(dataset):
    model_file = '/Storage/ying/project/brainAE/output/model_1e-05.bin'
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    pre_trained_model = Autoencoder()
    pre_trained_model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(os.path.join(model_file)).items()})
    pre_trained_model.to(device)

    norm_path = '/home/sakura/resources/BrainBertTorch/dataset/' + dataset + '/norm/'
    feature_path = '/home/sakura/resources/BrainBertTorch/dataset/' + dataset + '/features/'
    feature_data = {}
    norm_data = {}
    for user in participants:
        brain_data = load_brain_data(dataset, user)
        train_X = np.array(brain_data)
        train_X, _, _ = normalization(train_X)
        train_X = torch.Tensor(train_X)
        train_X = train_X.to(device, dtype=torch.float)
        y, z = pre_trained_model(train_X)
        # encoded_imgs, _, _ = normalization(encoded_imgs)
        # z = tf.reshape(encoded_imgs, [-1, 1024])
        norm_shape = train_X[0].shape
        feature_shape = z[0].shape[0]
        for index in range((z.shape[0])):
            npz_filename = dataset + '_' + user + '_' + str(index) + '.npz'
            norm_data['norm'] = {'data': train_X[index].cpu().data.tolist(),
                                 'shape': list(norm_shape),
                                 'type': None,
                                 'kind': None}
            feature_data['features'] = {'data': z[index].cpu().data.numpy().tolist(),
                                        'shape': [feature_shape],
                                        'type': None,
                                        'kind': None}
            np.savez(feature_path + npz_filename, features=feature_data['features'])
            np.savez(norm_path + npz_filename, features=norm_data['norm'])

            # print(k[1])


def calculate_corrcoef(dataset,z_size):
    if z_size == 1024:
        model_file = '/Storage/ying/project/brainAE/output/model1024_nol1_0.036462158693567566.bin'
    elif z_size == 768:
        model_file = '/Storage/ying/project/brainAE/output/model768_0.21957904651113178.bin'

    import pandas as pd
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    sentence_df_path = "/Storage/ying/project/brainAE/src/com/model/alice_sentence.tsv"
    brain2txt_path = "/Storage/ying/project/brainAE/src/com/model/brain2txt.json"
    brain2txt = json.loads(open(brain2txt_path, 'r', encoding='utf-8').read())

    pre_trained_model = Autoencoder()
    state_dict = {k.replace('module.', ''): v for k, v in torch.load(os.path.join(model_file)).items()}
    pre_trained_model.load_state_dict(state_dict)

    pre_trained_model.to(device)

    corr_user = {}
    for user in participants:
        print(user)
        #     file_spilt = file.replace('.npy', '').split('_')
        brain_data = load_brain_data(dataset, user)
        #     user = file_spilt[1]
        #     index = int(file_spilt[3])

        train_X = np.array(brain_data)
        train_X, _, _ = normalization(train_X)
        train_X = torch.Tensor(train_X)
        train_X = train_X.to(device, dtype=torch.float)
        y, z = pre_trained_model(train_X)
        corr_list = []
        # feature_shape = z[0].shape[0]

        sentence_df = pd.read_csv(sentence_df_path, sep='\t', header=0)
        # sentence_df.columns = ['text', 'type']

        # sentence_df.columns = ['sentences', 'type']
        for index in range((z.shape[0])):
            sentence_code = brain2txt.get('alice_'+str(user)+'_'+str(index)+'.npz')
            sentence_id = sentence_code.split('-')
            s_id = sentence_id[2]
            print(s_id)
            sentence = sentence_df.loc[int(s_id)]['sentences']
                # entence_df.loc[s_id]['sentences']
            sent_embeddings, t = get_sentence_embedding(sentence, "last_layer",z_size)
            # z[index, :].cpu().detach().numpy(), t.cpu().detach().numpy()
            corr = np.corrcoef(z[index, :].cpu().detach().numpy(), t.cpu().detach().numpy())[0, 1]
            corr_list.append(corr)
        corr_user[user] = corr_list
    with open(dataset+f'_corr_nol1_'+str(z_size)+'.json', 'w', encoding="utf8") as f:
        print(corr_user)
        json.dump(corr_user, f)
        f.close()


if __name__ == "__main__":
    start = time.perf_counter()
    make_print_to_file('.')
    calculate_corrcoef('alice_ae',1024)
    # print(corr_list)
    end = time.perf_counter()
    time_cost = str((end - start) / 60)
    print("time-cost:", time_cost)
    # create_brain_npz('alice_ae')