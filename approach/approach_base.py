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
        self.matrix.n_words_gt = len(gts) - self.get_dontcare_cnt()
        self.matrix.n_words_pred = len(preds)
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_dontcare_cnt(self):
        return len([gt for gt in self.gts if gt.is_dontcare()])

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
            # 이미 IoU 가 0.5 이상이라는 전제가 확보되어 있는 상태임.
            if self.matrix.is_end2end_tp(gt, pred, check_iou=False):
                gt.is_connected = True
                pred.is_connected = True
            elif gt.is_dontcare():
                gt.is_connected_dontcare = True
                pred.is_connected_dontcare = True
        self.matrix.update(self.preds, self.gts)

    def yield_pair(self):
        for gt, pred in self.pairs:
            yield gt, pred

    @classmethod
    def load_data(cls, image_id, split_char, *args, **kwargs) -> tuple:
        data_loader = cls.get_data_loader()
        return data_loader(image_id, split_char, *args, **kwargs)

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
    matrix_save_dir = kwargs.pop('matrix_save_dir', None)
    gt_name = kwargs.pop('gt_name', None)
    pred_name = kwargs.pop('pred_name', None)
    runtime.GlobalRuntime.set_mode(runtime_)
    gts, preds = approach_.load_data(image_id, 
                                     split_char=',',
                                     base_dir=kwargs.pop('base_dir', None),
                                     gt_base_dir=kwargs.pop('gt_base_dir', None),
                                     pred_base_dir=kwargs.pop('pred_base_dir', None),
                                     gt_name=gt_name,
                                     pred_name=pred_name,)
    runner = approach_(gts, preds, *args, **kwargs)
    mat = runner.run()
    if matrix_save_dir:
        if gt_name and pred_name:
            assert gt_name == pred_name, f'{gt_name} != {pred_name}'
            fname = f"{gt_name.replace('.txt', '')}.json"
        else:
            fname = f'{image_id}.json'
        mat.save(matrix_save_dir, fname)
    return mat