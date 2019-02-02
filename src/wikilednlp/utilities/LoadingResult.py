from os import makedirs, path

from wikilednlp.utilities.NumpyHelper import NumpyDynamic

from wikilednlp.utilities import logger
import numpy as np
from pathlib2 import Path
import pickle
import gzip


class LoadingSingleResult(object):

    def __init__(self, name, x):
        self.name = name
        self.y = None
        self.x = x
        self.block_id = None
        self.text = None


class LoadingResult(object):

    def __init__(self, records, document_type='document'):
        self.document_type = document_type
        self.records = records
        if len(records) == 0:
            raise Exception("No data")

    def save(self, data_path):
        result_file = Path(path.join(data_path, 'all.dat'))
        logger.info('Saving %s', str(result_file))
        with gzip.GzipFile(result_file, 'w') as f:
            pickle.dump(self.records, f)

    @staticmethod
    def load(data_path):
        if not Path(data_path).exists():
            makedirs(data_path)
        result_type = path.basename(path.normpath(data_path))
        result_file = Path(path.join(data_path, 'all.dat'))
        if result_file.exists():
            logger.info('Found created file. Loading %s...', str(result_file))
            with gzip.open(result_file, 'rb') as f:
                data = pickle.load(f)
            logger.info('Using saved data %s with %i records', str(result_file), len(data))
            return LoadingResult(data, result_type)
        else:
            return None

    def get_vectors(self):
        x_vectors = NumpyDynamic(np.object)
        y_class = NumpyDynamic(np.int32)
        length = []
        for record in self.records:
            x_vectors.add(record.x)
            y_class.add(record.y)
            length.append(len(record.x))

        x_data = x_vectors.finalize()
        total = (float(len(length) + 0.1))
        logger.info("get_data %i with average length %6.2f, min: %i and max %i", len(x_data),
                    sum(length) / total, min(length), max(length))
        return x_data, y_class.finalize()
