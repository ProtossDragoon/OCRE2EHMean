# 내장
import abc
import logging

# 프로젝트
import matrix
import runtime
from container.linked_list import Node
from approach.data_loader_base import BaseDataLoader


class Approach(abc.ABC):
    
    def __init__(self, gts, preds) -> None:
        self.gts = gts
        self.preds = preds
        self.pairs = []
        self._n_calculation = 0
        self.matrix = matrix.Matrix()
        self.matrix.n_words_gt = len(gts)
        self.matrix.n_words_pred = len(preds)
        self.logger = logging.getLogger(self.__class__.__name__)

    def is_matched(self, gt: Node, pred: Node):
        self._n_calculation += 1
        if matrix.Calculation.cal_iou(gt.polygon, pred.polygon) > 0.5:
            return True
        else:
            return False

    def add_pair(self, gt: Node, pred: Node):
        self.pairs.append((gt, pred))

    def cal_matrix(self):
        for gt, pred in self.yield_pair():
            if self.matrix.is_end2end_tp(gt, pred, check_iou=False):
                gt.is_connected = True
                pred.is_connected = True
        self.matrix.update(self.preds, self.gts)

    def yield_pair(self):
        for gt, pred in self.pairs:
            yield gt, pred

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