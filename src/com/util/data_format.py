import numpy as np
from sklearn.preprocessing import StandardScaler
s = StandardScaler()


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

def standardization(data, type):
    shape = data.shape
    if type == 'valid':
        # 验证集 直接用transform
        return s.transform(
            data.numpy().astype(np.float32).reshape(-1, 1)).reshape(shape)
    else:
        return s.fit_transform(
            data.numpy().astype(np.float32).reshape(-1, 1)).reshape(shape)


def normalization(data):
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    # normData = np.zeros(np.shape(data))
    # m = normData.shape[0]
    normData = data - np.tile(minVals, np.shape(data))
    normData = normData / np.tile(ranges, np.shape(data))

    return normData, ranges, minVals


def adjust_learning_rate(learning_rate, learning_rate_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR multiplied by learning_rate_decay(set 0.98, usually) every epoch"""
    learning_rate = learning_rate * (learning_rate_decay ** epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    return learning_rate
