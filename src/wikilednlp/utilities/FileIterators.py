import abc
import re
import shutil
from pathlib2 import Path
from os import path, walk
from wikilednlp.utilities import Constants, logger
import io
from wikilednlp.utilities.LoadingResult import LoadingResult, LoadingResultDynamic, LoadingSingleResult


class FileIterator(object):
    def __init__(self, source, data_path):
        self.source = source
        self.data_path = data_path

    def __iter__(self) -> LoadingSingleResult:
        logger.info("Loading %s...", self.data_path)

        for (root, dir_names, files) in walk(self.data_path):
            for name in files:
                file_name = path.join(root, name)
                sentence_id = 0
                for text, vector in self.source.get_vector(file_name):
                    if vector is not None:
                        result = LoadingSingleResult(name, vector)
                        result.block_id = sentence_id
                        result.text = text
                        yield result
                        sentence_id += 1


class DataIterator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, source, root, data_path):
        split_result = path.split(data_path)
        if len(split_result) > 1:
            self.name = path.split(data_path)[1]
        else:
            self.name = data_path
        self.source = source
        self.data_path = path.join(root, data_path)
        root_name = path.split(root)[1]
        sub_folder = ''.join(ch for ch in data_path if ch.isalnum())
        self.tag = "document"
        if source.use_sentence:
            self.tag = "sentence"
        self.bin_location = path.join(Constants.TEMP, 'bin', root_name, sub_folder, self.source.word2vec.name, self.tag)

    @abc.abstractmethod
    def __iter__(self) -> LoadingSingleResult:
        pass

    def delete_cache(self):
        if Path(self.bin_location).exists():
            logger.info('Deleting [%s] cache dir', self.bin_location)
            shutil.rmtree(self.bin_location)

    def get_data(self, use_cache=True) -> LoadingResult:

        if use_cache:
            result = LoadingResult.load(self.bin_location)
            if result is not None:
                return result

        dynamic = LoadingResultDynamic(self.tag)
        for record in self:
            dynamic.add(record)

        result = dynamic.finalize()
        result.save(self.bin_location)

        return result


class NullDataIterator(DataIterator):
    def __iter__(self) -> LoadingSingleResult:
        pass


class ClassDataIterator(DataIterator):

    def __iter__(self) -> LoadingSingleResult:
        pos_files = FileIterator(self.source, path.join(self.data_path, 'pos'))
        neg_files = FileIterator(self.source, path.join(self.data_path, 'neg'))

        for record in pos_files:
            record.y = 1
            yield record
        for record in neg_files:
            record.y = 0
            yield record


class SingeDataIterator(DataIterator):

    def __iter__(self) -> LoadingSingleResult:
        pos_files = FileIterator(self.source, self.data_path)
        for record in pos_files:
            if record.y is None:
                record.y = -1
            yield record


class SemEvalFileReader(object):
    def __init__(self, file_name, source, convertor):
        self.file_name = file_name
        self.source = source
        self.convertor = convertor

    def __iter__(self) -> LoadingSingleResult:
        with io.open(self.file_name, 'rt', encoding='utf8') as csv_file:
            logger.info('Loading: %s', self.file_name)
            for line in csv_file:
                row = re.split(r'\t+', line)
                review_id = row[0]
                total_rows = len(row)
                if total_rows >= 3:
                    type_class = self.convertor.is_supported(row[total_rows - 2])
                    if type_class is not None:
                        text = row[total_rows - 1]
                        sentence_id = 0
                        for vector in self.source.get_vector_from_review(text):
                            if vector is not None:
                                result = LoadingSingleResult(review_id, vector)
                                result.block_id = sentence_id
                                result.text = text
                                result.y = type_class
                                yield result
                            else:
                                logger.warning("Vector not found: %s", text)
                            sentence_id += 1


class SemEvalDataIterator(DataIterator):

    def __init__(self, source, root, data_path, convertor):
        super(SemEvalDataIterator, self).__init__(source, root, data_path)
        self.bin_location += convertor.name
        self.convertor = convertor

    def __iter__(self) -> LoadingSingleResult:
        if path.isfile(self.data_path):
            for result in SemEvalFileReader(self.data_path, self.source, self.convertor):
                yield result
        else:
            for (root, dir_names, files) in walk(self.data_path):
                for name in files:
                    file_name = path.join(root, name)
                    for result in SemEvalFileReader(file_name, self.source, self.convertor):
                        yield result



