import numpy as np
from ..utilities import logger
from ..utilities import Utilities


class ClassConvertor(object):

    ignore_error = False

    def __init__(self, name, class_dict):
        self.class_dict = class_dict
        self.classes = np.unique(list(class_dict.values()))
        self.name = "{}_{}".format(name, len(self.classes))

    def total_classes(self):
        return len(self.classes)

    def create_vector(self, y):
        if self.total_classes() > 2:
            return Utilities.make_dual(y, self.total_classes())
        return y

    def is_binary(self):
        return self.total_classes() == 2

    def is_supported(self, y):
        if y not in self.class_dict:
            if not ClassConvertor.ignore_error:
                logger.warning("Value %s not supported", y)
            return None
        return self.class_dict[y]

    def make_single(self, y):
        if self.total_classes() == 2:
            if len(y.shape) > 1 and y.shape[1] > 1:
                return y[:, 1]
            if len(y.shape) > 1 and y.shape[1] == 1:
                return y[:, 0]
            return y
        else:
            return np.amax(y, axis=1)
