from os import path
import socket
import os

import tensorflow_core.python.keras.backend as backend

hostname = socket.gethostname().lower()

if os.name == 'nt':
    TEMP = "C:/Temp/Sentiment"
else:
    TEMP = "Temp/Sentiment"


use_fp16 = False

use_special_symbols = True

START = '[START]'
START_ID = 1
END = '[END]'
END_ID = 2
PAD = '[PAD]'
PAD_ID = 0
UNK = '[UNK]'
UNK_ID = 3


def set_special_tags(flag=False):
    global use_special_symbols
    global EMBEDDING_START_INDEX
    use_special_symbols = flag
    if use_special_symbols:
        EMBEDDING_START_INDEX = 4
    else:
        EMBEDDING_START_INDEX = 1


def set_fp16():
    backend.set_floatx('float16')
    backend.set_epsilon(1e-4)
    global use_fp16
    use_fp16 = True


def get_root_by_host():
    if hostname.capitalize() == 'Threadripper':
        return 'e:/'
    else:
        return 'd:/'


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

if use_special_symbols:
    EMBEDDING_START_INDEX = 4


