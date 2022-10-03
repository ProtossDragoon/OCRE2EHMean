# 내장
import abc
import logging

# 프로젝트
from runtime import GlobalRuntime


class BaseDataLoader(abc.ABC):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abc.abstractmethod
    def load_data(self, image_id, split_char):
        raise NotImplementedError

    def load_data_for_python_runtime(self, image_id, split_char):
        return self.load_data(image_id, split_char)

    def load_data_for_numba_runtime(self, image_id, split_char):
        raise NotImplementedError

    def load_data_for_numpy_runtime(self, image_id, split_char):
        raise NotImplementedError

    def load_data_for_cython_runtime(self, image_id, split_char):
        raise NotImplementedError

    def load_data_for_jax_runtime(self, image_id, split_char):
        raise NotImplementedError

    def __call__(self, image_id, split_char):
        if GlobalRuntime.is_python_runtime():
            return self.load_data_for_python_runtime(image_id, split_char)
        elif GlobalRuntime.is_numba_runtime():
            return self.load_data_for_numba_runtime(image_id, split_char)
        elif GlobalRuntime.is_numpy_runtime():
            return self.load_data_for_numpy_runtime(image_id, split_char)
        elif GlobalRuntime.is_cython_runtime():
            return self.load_data_for_numpy_runtime(image_id, split_char)
        elif GlobalRuntime.is_jax_runtime():
            return self.load_data_for_jax_runtime(image_id, split_char)
        else:
            raise NotImplementedError(f'Not implemented runtime: {GlobalRuntime.runtime}')