import os
Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
import sys

sys.path.append(Proj_dir)
# print(sys.path)
import json
import torch
import scipy.io as scio
import numpy as np
from src.com.util.data_format import normalization
from transformers import BertModel, BertTokenizer
from src.com.model.run_auto_encoder import Autoencoder

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True)
PROJ_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
# 这里调整模型
model_file = PROJ_DIR + '/benchmark/model_1e-05.bin'
# sent_to_brain_json = PROJ_DIR + "/text_tmp/alice_sent_to_brain.json"
dataset_name = "pereira"
participants = {'P01', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07',
                'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17'}

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
            print(user, examples.shape)
            ROI_path = '../../../resource/' + user + '_roi.mat'
            data = scio.loadmat(ROI_path)
            # read roi index
            roi = data['index']

            for i in range((examples.shape[0])):
                # exp1_save_path = '/Storage/ying/resources/BrainBertTorch/brain/pereira/npy/' + \
                #                  'pereira_' + user + '_exp1_' + str(i) + '.npy'
                b = []
                for index in roi[0]:
                    b.append(examples[i][index])
                # np.save(exp1_save_path, a)
                a.append(b[:46840])
            if user in paticipants_2:
                exp2_path = '/Storage/ying/resources/pereira2018/' + user + '/data_384sentences.mat'
                exp2_data = scio.loadmat(exp2_path)
                examples = exp2_data['examples_passagesentences']
                # b = []
                for i in range((examples.shape[0])):
                    # exp2_save_path = '/Storage/ying/resources/BrainBertTorch/brain/pereira/npy/' + \
                    #                  'pereira_' + user + '_exp2_' + str(i) + '.npy'
                    # print(exp2_save_path)
                    b = []
                    for index in roi[0]:
                        b.append(examples[i][index])
                    # print(len(a))
                    # np.save(exp2_save_path, a)
                    a.append(b[:46840])
            if user in paticipants_3:
                exp3_path = '/Storage/ying/resources/pereira2018/' + user + '/data_243sentences.mat'
                exp3_data = scio.loadmat(exp3_path)
                # print(exp3_data.keys())
                examples = exp3_data['examples_passagesentences']
                for i in range((examples.shape[0])):
                    # print(i)
                    # exp3_save_path = '/Storage/ying/resources/BrainBertTorch/brain/pereira/npy/' + \
                    #                  'pereira_' + user + '_exp3_' + str(i) + '.npy'
                    # print(exp3_save_path)
                    b = []
                    for index in roi[0]:
                        b.append(examples[i][index])
                    # print(len(a))
                    # np.save(exp3_save_path, a)
                    a.append(b[:46840])
    return a


def create_brain_npz(dataset):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    pre_trained_model = Autoencoder()
    # pre_trained_model = nn.DataParallel(pre_trained_model, device_ids=[0, 1, 2, 4, 5, 6, 7])  # multi-GPU

    # checkpoint = torch.load(os.path.join(model_file))
    # pre_trained_model.load_state_dict(checkpoint)
    pre_trained_model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(os.path.join(model_file)).items()})

    pre_trained_model.to(device)

    norm_path = '/Storage/ying/resources/BrainBertTorch/dataset/' + dataset + '/norm/'
    feature_path = '/Storage/ying/resources/BrainBertTorch/dataset/' + dataset + '/features/'
    feature_data = {}
    norm_data = {}
    # norm_path = '/Storage/ying/resources/BrainBertTorch/brain/' + dataset + '/npy/'
    # files = os.listdir(norm_path)

    for user in participants:
        #     file_spilt = file.replace('.npy', '').split('_')
        brain_data = load_brain_data(dataset, user)
        #     user = file_spilt[1]
        #     index = int(file_spilt[3])

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
            npz_filename = dataset_name + '_' + user + '_' + str(index) + '.npz'
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


def calculate_corrcoef(dataset):
    import pandas as pd
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    sentence1_tsv_path = '../../../resource/text_807.tsv'
    sentence2_tsv_path = '../../../resource/text_564.tsv'
    sentence3_tsv_path = '../../../resource/text_423.tsv'
    sentence4_tsv_path = '../../../resource/text_180.tsv'
    # join exp1, 2and 3
    type1_participants = ['P01', 'M02', 'M04', 'M07', 'M15']
    # join exp1 and 3
    type2_participants = ['M08', 'M09', 'M14']
    # join exp1 and 3
    type3_participants = ['M03']
    # only join exp1
    type4_participants = ['M01', 'M05', 'M06', 'M10', 'M13', 'M16', 'M17']
    pre_trained_model = Autoencoder()
    pre_trained_model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(os.path.join(model_file)).items()})

    pre_trained_model.to(device)

    corr_user = {}
    for user in participants:
        brain_data = load_brain_data(dataset, user)

        train_X = np.array(brain_data)
        train_X, _, _ = normalization(train_X)
        train_X = torch.Tensor(train_X)
        train_X = train_X.to(device, dtype=torch.float)
        y, z = pre_trained_model(train_X)
        corr_list = []
        # feature_shape = z[0].shape[0]
        if user in type1_participants:
            sentence_df_path = sentence1_tsv_path
        elif user in type2_participants:
            sentence_df_path = sentence2_tsv_path
        elif user in type3_participants:
            sentence_df_path = sentence3_tsv_path
        elif user in type4_participants:
            sentence_df_path = sentence4_tsv_path
        sentence_df = pd.read_csv(sentence_df_path, sep='\t', header=None)
        sentence_df.columns = ['text', 'type']
        for index in range((z.shape[0])):
            sentence = sentence_df.loc[index]['text']
            sent_embeddings, t = get_sentence_embedding(sentence, "last_layer")
            # z[index, :].cpu().detach().numpy(), t.cpu().detach().numpy()
            corr = np.corrcoef(z[index, :].cpu().detach().numpy(), t.cpu().detach().numpy())[0, 1]
            corr_list.append(corr)
        corr_user[user] = corr_list
    json.dump(corr_user, dataset+f'_corr_user.json')


if __name__ == "__main__":
    # dataset = 'pereira'
    # feature_path = '/Storage/ying/resources/BrainBertTorch/dataset/' + dataset + '/features/'
    # user = 'P01'
    # npz_filename = dataset_name + '_' + user + '_' + str(0) + '.npz'
    # # participants = {'P01', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07',
    # #                 'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17'}
    # features = np.load(feature_path+npz_filename)
    # print(features['data'])
    calculate_corrcoef('pereira')
