from os import path
import socket
import os

hostname = socket.gethostname().lower()

if os.name == 'nt':
    TEMP = "C:/Temp/Sentiment"
else:
    TEMP = "Temp/Sentiment"

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