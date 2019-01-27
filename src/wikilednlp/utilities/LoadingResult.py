from os import makedirs, path

from wikilednlp.utilities.NumpyHelper import NumpyDynamic

from wikilednlp.utilities import logger
import numpy as np
from pathlib2 import Path


class LoadingSingleResult(object):

    def __init__(self, name, data):
        self.name = name
        self.result_class = None
        self.data = data
        self.block_id = None
        self.text = None


class LoadingResult(object):

    def __init__(self, names, result_classes, data, block_ids=None, document_type='document'):
        self.names = names
        self.result_classes = result_classes
        self.data = data
        if block_ids is None:
            self.block_ids = np.zeros(self.data.shape)
        else:
            self.block_ids = block_ids
        self.document_type = document_type

    def save(self, data_path):
        data_file = Path(path.join(data_path, 'data.npy'))
        class_file = Path(path.join(data_path, 'class.npy'))
        name_file = Path(path.join(data_path, 'name.npy'))
        sentences_file = Path(path.join(data_path, 'sentences.npy'))
        logger.info('Saving %s', str(data_file))
        np.save(str(data_file), self.data)
        np.save(str(class_file), self.result_classes)
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
        self.vectors = NumpyDynamic(np.object)
        self.result_class = NumpyDynamic(np.int32)
        self.file_name = NumpyDynamic(np.object)
        self.sentence_id = NumpyDynamic(np.object)
        self.length = []
        self.result_type = result_type
        self.total = None

    def add(self, record: LoadingSingleResult):
        self.vectors.add(record.data)
        self.file_name.add(record.name)
        self.sentence_id.add(record.block_id)
        if record.result_class is None:
            self.result_class.add(-1)
        else:
            self.result_class.add(record.result_class)
        self.length.append(len(record.data))

    def finalize(self) -> LoadingResult:
        sentence_ids = self.sentence_id.finalize()
        data = self.vectors.finalize()
        names_data = self.file_name.finalize()
        types_data = self.result_class.finalize()

        if len(data) == 0:
            raise Exception("No files found")
        self.total = (float(len(self.length) + 0.1))
        logger.info("Loaded %i with average length %6.2f, min: %i and max %i", len(data),
                    sum(self.length) / self.total, min(self.length), max(self.length))
        return LoadingResult(names_data, types_data, data, sentence_ids, self.result_type)
