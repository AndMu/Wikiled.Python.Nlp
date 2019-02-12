from os import path
import socket
import os

hostname = socket.gethostname().lower()

if os.name == 'nt':
    TEMP = "C:/Temp/Sentiment"
else:
    TEMP = "Temp/Sentiment"


use_fp16 = False


def set_fp16():
    from keras import backend as K
    K.set_floatx('float16')
    K.set_epsilon(1e-4)
    global use_fp16
    use_fp16 = True


def get_root_by_host():
    if hostname == 'main-pc':
        return 'g:/'
    if hostname == 'alienpc' or hostname == 'hp-z8':
        return 'e:/'
    elif hostname == 'dev-pc':
        return '//storage/monitoring'
    else:
        return 'c:/'


def set_root(root, data_sets='DataSets', lexicons='lexicons'):
    global ROOT_LOCATION
    global DATASETS
    global PROCESSED_LEXICONS
    ROOT_LOCATION = root
    DATASETS = path.join(ROOT_LOCATION, data_sets)
    PROCESSED_LEXICONS = path.join(DATASETS, lexicons, '')


set_root(get_root_by_host())

TRAINING_BATCH = 10
TESTING_BATCH = 10
EMBEDDING_START_INDEX = 1