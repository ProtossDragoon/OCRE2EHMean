# 내장
import os
import time

# 프로젝트
from approach import Approach
from approach.data_loader_base import BaseDataLoader
from container.linked_list import Node, LinkedList
import parse


class NavieApproachDataLoader(BaseDataLoader):

    def load_data_for_numba_runtime(self, image_id, split_char, *args, **kwargs):
        return self.load_data(image_id, split_char, *args, **kwargs)

    def load_data(self, image_id, split_char, base_dir=None): 
        self.logger.debug(f'Start loading data')
        if base_dir is None:
            base_dir = os.path.join(os.getcwd(), 'data', 'preprocessed')
        p = os.path.join(base_dir, f'gt_{image_id}.txt')
        gts = LinkedList()
        with open(p, 'r', encoding='utf-8-sig') as f:
            line = f.readline()
            while line:
                gt = parse.parse_data(line, split_char)
                gt = Node(**gt)
                gts.append(gt)
                line = f.readline()
        p = os.path.join(base_dir, f'pred_{image_id}.txt')
        preds = LinkedList()
        with open(p, 'r', encoding='utf-8-sig') as f:
            line = f.readline()
            while line:
                pred = parse.parse_data(line, split_char)
                pred = Node(**pred)
                preds.append(pred)
                line = f.readline()
        self.logger.debug(f'# of preds: {len(preds)}')
        self.logger.debug(f'# of gts: {len(gts)}')
        return (gts, preds)


class ApproachNaive(Approach):

    @classmethod
    def get_data_loader(cls):
        return NavieApproachDataLoader()

    def __init__(self, gts, preds) -> None:
        super().__init__(gts, preds)
    
    def run(self):
        start = time.time()

        s = time.time()
        intersect_li = []
        for pred in self.preds:
            for gt in self.gts:
                if self.is_matched(gt, pred):
                    intersect_li.append((gt, pred))
        e = time.time()
        self.logger.debug(f'filtering dt: {e-s:.3f}(s)')

        s = time.time()
        for gt, pred in intersect_li:
            if self.is_matched(gt, pred):
                self.add_pair(gt, pred)
        e = time.time()
        self.logger.debug(f'matching dt: {e-s:.3f}(s)')

        s = time.time()
        self.cal_matrix()
        e = time.time()
        self.logger.debug(f'cal_matrix dt: {e-s:.3f}(s)')

        end = time.time()
        self.logger.info(f'total dt: {end-start:.2f}(s) (O(n^2), compared {self._n_calculation} times)')
        self.logger.info(f'Found {len(self.pairs)} pairs. (F1: {self.matrix.f1:.2f})')
        return self.matrix

    def is_intersect(self, pred:Node, gt:Node):
        # case 1
        a = (pred.left_top.x <= gt.right_bottom.x)
        b = (pred.left_top.y <= gt.right_bottom.y)
        # case 2
        c = (pred.right_bottom.x >= gt.left_top.x)
        d = (pred.right_bottom.y >= gt.left_top.y)
        return (a and b) or (c and d)