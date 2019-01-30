import abc

import logging

from wikilednlp.utilities import LoadingResult
from wikilednlp.utilities.ClassConvertors import ClassConvertor
from wikilednlp.utilities.FileIterators import ClassDataIterator, SingeDataIterator, SemEvalDataIterator, \
    NullDataIterator

logger = logging.getLogger(__name__)


class DataLoader(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, parser, convertor, root):
        self.root = root
        self.convertor = convertor
        self.parser = parser

    @abc.abstractmethod
    def get_iterator(self, data_path, class_iter=True):
        pass

    def get_data(self, data_path, delete=False, class_iter=True) -> LoadingResult:
        logger.info("Loading [%s]...", data_path)
        train_iterator = self.get_iterator(data_path, class_iter)

        if delete:
            train_iterator.delete_cache()

        record = train_iterator.get_data()
        return record


class NullDataLoader(DataLoader):
    def __init__(self, convertor=None):
        if convertor is None:
            convertor = ClassConvertor("Basic", {0: 0, 1: 1})
        super(NullDataLoader, self).__init__(None, convertor, None)

    def get_iterator(self, data_path, class_iter=True):
        return NullDataIterator()


class ImdbDataLoader(DataLoader):

    def get_iterator(self, data_path, class_iter=True):
        if class_iter:
            return ClassDataIterator(self.parser, self.root, data_path)

        return SingeDataIterator(self.parser, self.root, data_path)


class SemEvalDataLoader(DataLoader):

    def __init__(self, parser, convertor, root):
        super(SemEvalDataLoader, self).__init__(parser, convertor, root)
        self.convertor = convertor

    def get_iterator(self, data_path, class_iter=True):
        return SemEvalDataIterator(self.parser, self.root, data_path, self.convertor)
