# 내장
import abc
import logging

# 프로젝트
import matrix
import runtime
from approach.data_loader_base import BaseDataLoader


class Approach(abc.ABC):
    
    def __init__(self, gts, preds) -> None:
        self.gts = gts
        self.preds = preds
        self.pairs = []
        self._n_calculation = 0
        self.matrix = matrix.Matrix()
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_pair(self, gt, pred):
        self.pairs.append((gt, pred))

    def yield_pair(self):
        for gt, pred in self.pairs:
            yield gt, pred

    def update(self, *args, **kwargs):
        self.matrix.update(*args, **kwargs)

    @classmethod
    def load_data(cls, image_id, split_char) -> tuple:
        data_loader = cls.get_data_loader()
        return data_loader(image_id, split_char)

    @classmethod
    @abc.abstractmethod
    def get_data_loader(cls) -> BaseDataLoader:
        raise NotImplementedError
    

def start(
    image_id: int,
    runtime_: runtime.Runtime,
    approach_: Approach,
    *args, **kwargs,
):
    runtime.GlobalRuntime.set_mode(runtime_)
    gts, preds = approach_.load_data(image_id, split_char=',  ')
    runner = approach_(gts, preds, *args, **kwargs)
    return runner.run()