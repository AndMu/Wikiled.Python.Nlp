from os import path
import socket


hostname = socket.gethostname()
TEMP = "C:/Temp/Sentiment"


def get_root_by_host():
    if hostname.lower() == 'main-pc':
        return 'g:/'
    if hostname.lower() == 'alienpc':
        return 'e:/'
    elif hostname.lower() == 'dev-pc':
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