from os import makedirs, path

from wikilednlp.utilities.NumpyHelper import NumpyDynamic

from wikilednlp.utilities import logger
import numpy as np
from pathlib2 import Path


class LoadingSingleResult(object):

    def __init__(self, name, x):
        self.name = name
        self.y = None
        self.x = x
        self.block_id = None
        self.text = None


class LoadingResult(object):

    def __init__(self, names, y_data, x_data, block_ids=None, document_type='document'):
        self.names = names
        self.y_data = y_data
        self.x_data = x_data
        if block_ids is None:
            self.block_ids = np.zeros(self.x_data.shape)
        else:
            self.block_ids = block_ids
        self.document_type = document_type
        self.original = []

    def save(self, data_path):
        data_file = Path(path.join(data_path, 'data.npy'))
        class_file = Path(path.join(data_path, 'class.npy'))
        name_file = Path(path.join(data_path, 'name.npy'))
        sentences_file = Path(path.join(data_path, 'sentences.npy'))
        logger.info('Saving %s', str(data_file))
        np.save(str(data_file), self.x_data)
        np.save(str(class_file), self.y_data)
        np.save(str(name_file), self.names)
        np.save(str(sentences_file), self.block_ids)

    @staticmethod
    def load(data_path):
        if not Path(data_path).exists():
            makedirs(data_path)
        result_type = path.basename(path.normpath(data_path))

        data_file = Path(path.join(data_path, 'data.npy'))
        class_file = Path(path.join(data_path, 'class.npy'))
        name_file = Path(path.join(data_path, 'name.npy'))
        sentences_file = Path(path.join(data_path, 'sentences.npy'))
        if data_file.exists():
            logger.info('Found created file. Loading %s...', str(data_file))
            data = np.load(str(data_file))
            type_data = np.load(str(class_file))
            names_data = np.load(str(name_file))
            sentence_ids = np.load(str(sentences_file))
            logger.info('Using saved data %s with %i records', str(data_file), len(data))
            return LoadingResult(names_data, type_data, data, sentence_ids, result_type)
        else:
            return None


class LoadingResultDynamic(object):
    def __init__(self, result_type='document'):
        self.x_vectors = NumpyDynamic(np.object)
        self.y_class = NumpyDynamic(np.int32)
        self.names = NumpyDynamic(np.object)
        self.block_ids = NumpyDynamic(np.object)
        self.length = []
        self.result_type = result_type
        self.total = None
        self.original = []

    def add(self, record: LoadingSingleResult):
        if record.y is not None:
            self.x_vectors.add(record.x)
            self.names.add(record.name)
            self.block_ids.add(record.block_id)
            self.y_class.add(record.y)
            self.length.append(len(record.x))
            self.original.append(record)
        else:
            logger.warning('Dropping records without class')

    def finalize(self) -> LoadingResult:
        bock_id = self.block_ids.finalize()
        x_data = self.x_vectors.finalize()
        names_data = self.names.finalize()
        y_data = self.y_class.finalize()

        if len(x_data) == 0:
            raise Exception("No files found")
        self.total = (float(len(self.length) + 0.1))
        logger.info("Loaded %i with average length %6.2f, min: %i and max %i", len(x_data),
                    sum(self.length) / self.total, min(self.length), max(self.length))
        result = LoadingResult(names_data, y_data, x_data, bock_id, self.result_type)
        result.original = self.original
        return result
