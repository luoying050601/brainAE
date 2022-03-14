import os
# Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "src/com/model/"))
Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# print(Proj_dir)
p = os.path.abspath(os.path.join(os.getcwd(), "."))
# print(p)
import sys
sys.path.append(p)
import json
import time
import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel
from src.com.util.data_format import normalization,standardization
import transformers
from transformers import AdamW
from tqdm.notebook import tqdm

bert_model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

lr = 1e-5
criterion_l2 = nn.MSELoss()
criterion_l1 = nn.L1Loss()
corr_list = []

def get_sentence_embedding(sentence, tokenizer, option):
    # if YOU NEED [cls] and [seq] PRESENTATION：True
    input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=False)).unsqueeze(0)  # Batch size 1
    outputs = bert_model(input_ids)

    all_layers_output = outputs[2]
    # list of torch.FloatTensor
    # (one for the output of each layer + the output of the embeddings)
    # of shape (batch_size, sequence_length, hidden_size):
    # Hidden-states of the model at the output of each layer
    # plus the initial embedding outputs.

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


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.dense_enc1 = nn.Linear(46840, 8192)  # [batch*size]->[batch*size]
        self.bn1 = nn.BatchNorm1d(8192)
        self.dense_enc2 = nn.Linear(8192, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        # 3l
        self.dense_enc3 = nn.Linear(4096, 1024)
        # self.dense_enc3 = nn.Linear(8192, 1024)
        # vanilla
        self.dense_enc4 = nn.Linear(8192, 1024)

        self.dense_dec0 = nn.Linear(1024, 8192)

        self.dense_dec1 = nn.Linear(1024, 4096)
        self.bn4 = nn.BatchNorm1d(4096)
        self.dense_dec2 = nn.Linear(4096, 8192)
        self.bn5 = nn.BatchNorm1d(8192)
        self.drop1 = nn.Dropout(p=0.2)
        self.dense_dec3 = nn.Linear(8192, 46840)

    def encoder(self, x):
        x = F.relu(self.dense_enc1(x))
        x = self.bn1(x)
        # vanilla
        x = F.relu(self.dense_enc4(x))
        # 3 layer
        # x = F.relu(self.dense_enc2(x))
        # x = self.bn2(x)
        # x = F.relu(self.dense_enc3(x))
        return x

    def decoder(self, x):
        # 3 layer
        # x = F.relu(self.dense_dec1(x))
        # x = self.bn4(x)
        # x = F.relu(self.dense_dec2(x))
        # vanilla
        x = F.relu(self.dense_dec0(x))
        x = self.bn5(x)
        x = self.drop1(x)
        x = F.relu(self.dense_dec3(x))
        return x

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z


class config:
    import random
    SEED = random.randint(0,100)
    print("seed:",SEED)
    _type = "1024_vanilla"
    # KFOLD = 5
    SAVE_DIR = '/Storage/ying/project/brainAE/output'
    # TRAIN_FILE = 'train.tsv'
    # VAL_FILE = 'valid.tsv'
    # TEST_FILE = 'test.tsv'
    # OOF_FILE = os.path.join(SAVE_DIR, 'oof.csv')
    MAX_LEN = 2048
    MODEL = 'bert-large-uncased'
    TOKENIZER = transformers.AutoTokenizer.from_pretrained(MODEL)
    EPOCHS = 75
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    @property
    def type(self):
        return self._type


def seed_all(seed=42):
    """
  Fix seed for reproducibility
  """
    # python RNG
    import random
    random.seed(seed)

    # pytorch RNGs

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # numpy RNG

    np.random.seed(seed)


def preprocess(text):
    # text = html.unescape(text)
    # text = text.translate(transl_table)
    # text = text.replace('…', '...')
    # text = re.sub(control_char_regex, ' ', text)
    # text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    text = ' '.join(text.split())
    text = text.lower()
    text = text.strip()

    # text = text.replace('HTTPURL', 'URL')
    # text = emoji.demojize(text)

    # text = unidecode.unidecode(text)
    # text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'So')

    return text


def process_data(data, tokenizer, max_len):
    text = preprocess(data['label'])
    embeding, embeding_avg = get_sentence_embedding(text, tokenizer, "last_layer")
    # input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False, pad_to_max_length=True)).unsqueeze(
    #     0)  # Batch size 1
    # model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    # token_ids = tokenizer.encode(text, add_special_tokens=False,pad_to_max_length=True), dim=0)
    # # mask = [1] * len(token_ids)
    #
    # padding = max_len - len(token_ids)
    #
    # if padding >= 0:
    #     token_ids = token_ids + ([0] * padding)
    #     # mask = mask + ([0] * padding)
    # else:
    #     token_ids = token_ids[0:max_len]
    #     # mask = mask[0:max_len]
    #
    # # label = 1 if label == 'INFORMATIVE' else 0
    #
    # assert len(token_ids) == max_len
    # # assert len(mask) == max_len

    return {
        'ids': data['index'],
        'brain': data['brain'],
        # 'mask': mask,
        'label': embeding_avg
    }


class Dataset:
    def __init__(self, dataList):
        self.dataList = dataList
        # self.label = label
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, item):
        data = process_data(
            self.dataList[item],
            self.tokenizer,
            self.max_len,
            # self.label[item],
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            # 'mask': torch.tensor(data["mask"], dtype=torch.long),
            'brain': data['brain'],
            'label': data['label'],
        }


class EarlyStopping:
    """
    Early stopping utility
    Source : https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/
    """

    def __init__(self, patience=7, mode="max", delta=0.001,optimizer=None):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.optimizer = optimizer
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
            self.delta = -1.0 * self.delta
        else:
            score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
            # torch.save({'state_dict': model.state_dict(),
            #             'optimizer_state_dict': self.optimizer.state_dict()},
            #            model_path)
        self.val_score = epoch_score


class AverageMeter:
    """
    Computes and stores the average and current value
    Source : https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(data_loader, model, optimizer, device):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    # optimizer.zero_grad()

    for bi, d in enumerate(tk0):
        optimizer.zero_grad()
        x = d['brain']
        t = d['label']

        x = standardization(x, 'train')
        t = standardization(t, 'train')
        x, _, _ = normalization(x)
        t, _, _ = normalization(t)
        x = torch.Tensor(x).float()
        t = torch.Tensor(t).float()

        x = x.to(device, dtype=torch.float)
        t = t.to(device, dtype=torch.float)
        # model.zero_grad()
        # with torch.no_grad():
        y, z = model(x)
        loss = criterion_l2(y.float(), x.float()) + criterion_l1(z.float(), t.float())
        # corr = np.corrcoef(t.cpu().detach().numpy(), z.cpu().detach().numpy())[0, 1]
        # corr_list.append(corr)
        # loss.requres_grad = True
        loss.backward()
        optimizer.step()
        tk0.set_postfix(loss=losses.avg)


# %%
def eval_fn(data_loader, model, device):
    model.eval()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    x_list = []
    y_list = []
    for bi, d in enumerate(tk0):
        x = d['brain']
        t = d['label']
        x = standardization(x, 'valid')
        t = standardization(t, 'valid')
        x, _, _ = normalization(x)
        t, _, _ = normalization(t)
        x = torch.Tensor(x).float()
        t = torch.Tensor(t).float()
        x = x.to(device, dtype=torch.float)
        t = t.to(device, dtype=torch.float)
        # model.zero_grad()
        y, z = model(x.float())
        # loss = criterion(y.float(), x.float())
        loss = criterion_l2(y.float(), x.float()) + criterion_l1(z.float(), t.float())
        # corr = np.corrcoef(t.cpu().detach().numpy(), z.cpu().detach().numpy())[0, 1]
        # corr_list.append(corr)
        print(loss)
        loss.backward()
        tk0.set_postfix(loss=losses.avg)
        x_list.append(x)
        y_list.append(y)

        losses.update(loss.item(), d['ids'].size(0))
        tk0.set_postfix(loss=losses.avg)
    return losses.avg
    # return mean_squared_error(x_list, y_list)

def run(train, val, fold=None):
    train_dataset = Dataset(
        dataList=train
        # brain=train['brain'],
        # label='',
    )

    valid_dataset = Dataset(
        dataList=val
        # brain=val['brain'],
        # label='',
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4,
        drop_last=True,
        shuffle=True
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2,
        drop_last=True,
        shuffle=True
    )
    # with torch.no_grad():
    model = Autoencoder()
    checkpoint_file = os.path.join(config.SAVE_DIR, 'model_' + config._type + '_0.10589077356068985.bin')
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint,False)

    model = nn.DataParallel(model, device_ids=[0,1,2,3])  # multi-GPU
    # cudnn.benchmark = True
    # model = transformers.RobertaForSequenceClassification.from_pretrained(config.MODEL, num_labels=2)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model.to(device)
    # criterion = nn.MSELoss()2
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')  # set up scheduler

    es = EarlyStopping(patience=10, mode="min",optimizer=optimizer)
    # es = EarlyStopping(patience=3, mode="max")

    print('Starting training....')
    losses = []
    for epoch in range(config.EPOCHS):
        print(epoch)
        train_fn(train_data_loader, model, optimizer, device)
        valid_loss = eval_fn(valid_data_loader, model, device)
        losses.append(valid_loss)
        # scheduler.step(valid_loss, epoch)  # update lr if needed
        print(f'Epoch :{epoch + 1} | Validation Loss :{valid_loss}')
        if fold is None:
            es(valid_loss, model, model_path=os.path.join(config.SAVE_DIR, f'model_' + config._type + '_'+str(valid_loss)+'.bin'))
        else:
            es(valid_loss, model, model_path=os.path.join(config.SAVE_DIR, f'model_{fold}.bin'))
        if es.early_stop:
            print('Early stopping')
            break

    print('Predicting for OOF')
    print(losses)


def dataloader_alice():
    brain2txt = json.load(open(f'/Storage/ying/project/brainAE/src/com/model/brain2txt.json', 'r'))
    sentence_df = pd.read_csv(f'/Storage/ying/project/brainAE/src/com/model/alice_sentence.tsv', sep='\t', header=0)
    brain_path = '/Storage/ying/resources/BrainBertTorch/brain/alice/ae_npy/'
    files = os.listdir(brain_path)
    # i = 0
    train_data = []
    val_data = []
    # min_size = 10000000
    for file in files:  # 遍历文件夹
        file_spilt = file.replace('.npy', '').split('_')
        index = int(file_spilt[2])
        user = file_spilt[1]
        brain_data = np.load(brain_path + file)
        name = 'alice_' + str(user) + '_' + str(index) + '.npz'
        sentence_id = int(brain2txt[name].split('-')[2])
        if index < 362 * 0.8:
                train_data.append({
                    'index': index,
                    'participant': user,
                    'brain': brain_data[:46840],
                    'label': sentence_df['sentences'][sentence_id]
                })
        else:
                val_data.append({
                    'index': index,
                    'participant': user,
                    'brain': brain_data[:46840],
                    'label': sentence_df['sentences'][sentence_id]
                })

        # (372, 199662)
    # print("min_size:", min_size)
    return train_data, val_data


def dataloader_pereira():
    train_data = []
    val_data = []
    # brain_path = '/Storage/ying/resources/BrainBertTorch/brain/pereira/npy/AE_training/'
    brain_path = '/Storage/ying/resources/BrainBertTorch/brain/pereira/npy/'
    files = os.listdir(brain_path)
    i = 0
    for file in files:  # 遍历文件夹
        # print(file.replace('.npy', '').split('_'))
        file_spilt = file.replace('.npy', '').split('_')
        index = int(file_spilt[3])
        user = file_spilt[1]
        exp_index = file_spilt[2]
        brain_data = np.load(brain_path + file)
        if exp_index == 'exp1':
            # 180 ->144 train; ->36 val

            # sentence_df = pd.read_csv(Proj_dir+'/resource/exp1_text.tsv', sep='\t', header=0)
            sentence_df = pd.read_csv('/Storage/ying/project/brainAE/resource/exp1_text.tsv', sep='\t', header=0)
            group_boundary = 144
        elif exp_index == 'exp2':
            # sentence_df = pd.read_csv(Proj_dir+'/resource/exp2_text.tsv', sep='\t', header=0)

            sentence_df = pd.read_csv('/Storage/ying/project/brainAE/resource/exp2_text.tsv', sep='\t', header=0)
            # 384 ->308 train; ->76 val
            group_boundary = 308
        elif exp_index == 'exp3':
            # sentence_df = pd.read_csv(Proj_dir+'/resource/exp3_text.tsv', sep='\t', header=0)

            sentence_df = pd.read_csv('/Storage/ying/project/brainAE/resource/exp3_text.tsv', sep='\t', header=0)
            # 243 ->195 train; ->48 val
            group_boundary = 195
        if index < group_boundary:
            train_data.append({
                'index': i,
                'participant': user,
                'brain': brain_data[:46840],
                'label': sentence_df['text'][index]})
            i = i + 1
        else:
            val_data.append({
                'index': i,
                'participant': user,
                'brain': brain_data[:46840],
                'label': sentence_df['text'][index]
            })
            i = i + 1

    # print("min_size:", min_size)
    return train_data, val_data


def dataloader_switch(type):
    if type == 'alice':
        train_, val_ = dataloader_alice()
    elif type == 'pereira':
        train_, val_ = dataloader_pereira()
    return train_, val_


def run_fold(type, fold_idx=None):
    """
      Perform k-fold cross-validation
    """
    seed_all(seed=config.SEED)
    train_, val_ = dataloader_switch(type=type)

    run(train_, val_, fold_idx)


def make_print_to_file(path='.'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import os
    # import config_file as cfg_file
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('day and time:' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # print -> log
    #############################################################
    print(fileName.center(60, '*'))


if __name__ == "__main__":
    start = time.perf_counter()
    make_print_to_file('.')
    run_fold(type='pereira', fold_idx=None)
    print(corr_list)
    end = time.perf_counter()
    time_cost = str((end - start) / 60)
    print("time-cost:", time_cost)
