# prince dataset extraction
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6"  # GPU

Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
import sys
import time
import h5py

sys.path.append(Proj_dir)
# print(sys.path)
# import json
import torch
from tqdm import tqdm, trange

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
import scipy.io as scio
import numpy as np
from tqdm import tqdm
from src.com.util.data_format import make_print_to_file
from src.com.util.data_format import normalization
from transformers import BertModel, BertTokenizer
from src.com.model.run_auto_encoder_prince import Autoencoder, config, Dataset
# import pandas as pd

_config = config()
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True)
PROJ_DIR = os.path.abspath(os.path.join(os.getcwd(), "."))
# 这里调整模型
# 768
# participants = {'18', '22', '23', '24', '26', '28', '30', '31', '35', '36', '37',
#                 '39', '41', '42', '43', '44', '45', '47', '48', '49', '50', '51', '53'}
AE_participants = ['sub-EN058', 'sub-EN062', 'sub-EN063', 'sub-EN064', 'sub-EN068', 'sub-EN075', 'sub-EN076',
                   'sub-EN077', 'sub-EN086', 'sub-EN087', 'sub-EN067', 'sub-EN069', 'sub-EN072', 'sub-EN073',
                   'sub-EN074', 'sub-EN078', 'sub-EN079', 'sub-EN081', 'sub-EN083', 'sub-EN084']
participants = [
    'sub_EN057', 'sub_EN058', 'sub_EN059', 'sub_EN061', 'sub_EN062',
    'sub_EN063', 'sub_EN064',
    'sub_EN065', 'sub_EN067', 'sub_EN068', 'sub_EN069', 'sub_EN070', 'sub_EN072', 'sub_EN073', 'sub_EN074', 'sub_EN075',
    'sub_EN076', 'sub_EN077', 'sub_EN078', 'sub_EN079', 'sub_EN081', 'sub_EN082', 'sub_EN083', 'sub_EN084', 'sub_EN086',
    'sub_EN087', 'sub_EN088', 'sub_EN089', 'sub_EN091', 'sub_EN092', 'sub_EN094', 'sub_EN095', 'sub_EN096', 'sub_EN097',
    'sub_EN098', 'sub_EN099', 'sub_EN100', 'sub_EN101', 'sub_EN103', 'sub_EN104', 'sub_EN105', 'sub_EN106', 'sub_EN108',
    'sub_EN109', 'sub_EN110', 'sub_EN113', 'sub_EN114', 'sub_EN115']


def get_sentence_embedding(sentence, option, z_size):
    bert_version = 'bert-base-uncased'
    if z_size == 1024:
        bert_version = 'bert-large-uncased'
    # elif z_size == 768:
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

    # Calculate the average of all token vectors.
    sentence_embedding_avg = torch.mean(sent_embeddings, dim=0)

    return sent_embeddings.detach(), sentence_embedding_avg.detach()


# %%
def eval(data_loader, model):
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    z_list = []
    for bi, d in enumerate(tk0):
        x = d['brain']
        # x = standardization(x, 'valid')
        x, _, _ = normalization(x)
        x = torch.Tensor(x).float()
        x = x.to(_config.DEVICE, dtype=torch.float)
        y, z = model(x.float())
        z_list.append(z.T.tolist())
    return np.concatenate(z_list, axis=1)


def load_brain_data(dataset, _type, user):  # user "sub_EN070"
    a = []
    if dataset == 'pereira':
        paticipants_1 = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07',
                         'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17', 'P01']
        paticipants_2 = ['M02', 'M04', 'M07', 'M08', 'M09', 'M14', 'M15', 'P01']
        paticipants_3 = ['M02', 'M03', 'M04', 'M07', 'M15', 'P01']
        # dataset_path = "/Storage/ying/resources/pereira2018/'+user+'/data_180concepts_wordclouds.mat"
        if user in paticipants_1:
            exp1_path = '/Storage/ying/resources/pereira2018/' + user + '/data_180concepts_wordclouds.mat'
            exp1_data = scio.loadmat(exp1_path)
            examples = exp1_data['examples']

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
                examples = exp3_data['examples_passagesentences']
                for i in range((examples.shape[0])):

                    b = []
                    for index in roi[0]:
                        b.append(examples[i][index])
                    a.append(b[:46840])
    elif dataset == 'alice_ae' or dataset == 'alice':

        brain_path = '/Storage/ying/resources/BrainBertTorch/brain/' + dataset + '/ae_npy/'
        files = sorted(os.listdir(brain_path))
        for file in files:  # 遍历文件夹
            file_spilt = file.replace('.npy', '').split('_')
            sub = file_spilt[1]
            if sub == user:
                brain_data = np.load(brain_path + file)
                a.append(brain_data[:46840])
    else:  # prince
        prince_proj_path = "/Storage2/ying/project/littlePrinceDatasetProj/"  # laplace  home local

        # sentence_df = json.load(open(prince_proj_path + f'resources/section_word_segment_dict_id.json', 'r'))
        brain_path = prince_proj_path + 'resources/preprocessed/downstreamTask/'  #
        files = sorted(os.listdir(brain_path))
        # i = 0
        a = []
        for file in files:  # 遍历文件夹
            subj = 'sub-' + file.split('_')[1]
            if subj == user.replace('_', '-'):
                with h5py.File(brain_path + file, "r") as file:
                    dataset = file["brain"]
                    brain_data = dataset[:46840]
                    data_size = dataset.shape[1]

                    for i in trange(data_size):
                        a.append({
                            'index': i,
                            'participant': subj,
                            'brain': brain_data[:, i],
                            #
                        })
            #

            # break # TODO debug model
    return a


def create_brain_npz(_type, dataset):
    # print(_type)
    if _type == 'bert-large-uncased':
        model_file = '/Storage2/ying/project/brainAE/benchmark/prince_1024_AE_0.04.bin'
        config.LATENT_SIZE = 1024
    elif _type == 'sentence-camembert-large':
        model_file = '/Storage/ying/project/brainAE/output/sentence-camembert-large_768_0.083.bin'
        config.LATENT_SIZE = 1024
    elif _type == 'SimCSE':
        model_file = '/Storage/ying/project/brainAE/output/SimCSE_768_0.076.bin'
        config.LATENT_SIZE = 768
    elif _type == 'SimCSE_bert_large_unsup':
        model_file = '/Storage/ying/project/brainAE/output/SimCSE_bert_large_unsup_1024_0.10.bin'
        config.LATENT_SIZE = 1024
    elif _type == 'SimCSE_large':
        model_file = '/Storage/ying/project/brainAE/output/SimCSE_large_1024_0.09.bin'
        config.LATENT_SIZE = 1024
    elif _type == 'SimCSE_roberta_large':
        model_file = '/Storage/ying/project/brainAE/output/SimCSE_roberta_large_1024_0.072.bin'
        config.LATENT_SIZE = 1024
    elif _type == 'SimCSE_roberta_large_sup':
        model_file = '/Storage/ying/project/brainAE/output/SimCSE_roberta_large_sup_1024_0.078.bin'
        config.LATENT_SIZE = 1024
    # model_file = '/Storage/ying/project/brainAE/output/model_1e-05.bin'
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    pre_trained_model = Autoencoder()
    pre_trained_model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(os.path.join(model_file)).items()})
    pre_trained_model.to(device)

    norm_path = '/Storage2/ying/project/brainLMProj/dataset/' + dataset + '/' + _type + '/norm/'
    feature_path = '/Storage2/ying/project/brainLMProj/dataset/' + dataset + '/' + _type + '/features/'
    feature_data = {}
    norm_data = {}
    for user in tqdm(sorted(participants)):
        if user.replace('_', '-') not in AE_participants:
            print(user)
            brain_data = load_brain_data(dataset, _type, user)
            valid_dataset = Dataset(
                dataList=brain_data,
                _is_eval=True

            )
            valid_data_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=32,
                num_workers=5,
                drop_last=True,
                shuffle=True, pin_memory=True
            )
            z = eval(valid_data_loader, pre_trained_model)

            z = z.transpose()  # 2816 1024
            # feature_shape = z[0]
            for index in trange(len(z)):
                npz_filename = dataset + '_' + user + '_' + str(index) + '.npz'
                norm_data['norm'] = {'data': z[index].tolist(),
                                     'shape': [1024],
                                     'type': None,
                                     'kind': None}
                feature_data['features'] = {'data': z[index].tolist(),
                                            'shape': [1024],
                                            'type': None,
                                            'kind': None}
                np.savez(feature_path + npz_filename, features=feature_data['features'])
                np.savez(norm_path + npz_filename, features=norm_data['norm'])
            print(user, "finished")


if __name__ == "__main__":
    start = time.perf_counter()
    make_print_to_file('.')
    # _type = ''
    create_brain_npz(_type='bert-large-uncased', dataset='prince')

    end = time.perf_counter()
    time_cost = str((end - start) / 60)
    print("time-cost:", time_cost)
