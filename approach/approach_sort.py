# 내장
import os
import time

# 프로젝트
from approach import Approach
from approach.data_loader_base import BaseDataLoader
from container.linked_list import Node
import parse


class NodeV2(Node):
    def __init__(
        self, 
        polygon:list, 
        left_top:tuple,
        right_bottom:tuple,
        sentence:str,
        **kwargs,
    ):
        super().__init__(polygon, left_top, right_bottom, sentence, **kwargs)
        self.matched_candidates_filtered_by_left = set()
        self.matched_candidates_filtered_by_top = set()
        self.matched_candidates_filtered_by_right = set()
        self.matched_candidates_filtered_by_bottom = set()
        self.matched = set()

    def add_candidate_filtered_by_left(self, node:Node):
        self.matched_candidates_filtered_by_left.add(node)
        if node in self.matched_candidates_filtered_by_top:
            self.matched.add(node)

    def add_candidate_filtered_by_top(self, node:Node):
        self.matched_candidates_filtered_by_top.add(node)
        if node in self.matched_candidates_filtered_by_left:
            self.matched.add(node)

    def add_candidate_filtered_by_right(self, node:Node):
        self.matched_candidates_filtered_by_right.add(node)
        if node in self.matched_candidates_filtered_by_bottom:
            self.matched.add(node)

    def add_candidate_filtered_by_bottom(self, node:Node):
        self.matched_candidates_filtered_by_bottom.add(node)
        if node in self.matched_candidates_filtered_by_right:
            self.matched.add(node)

    def cond_intersect(self):
        a = set(self.matched_candidates_filtered_by_left)
        b = set(self.matched_candidates_filtered_by_top)
        case_1 = a.intersection(b)
        c = set(self.matched_candidates_filtered_by_right)
        d = set(self.matched_candidates_filtered_by_bottom)
        case_2 = c.intersection(d)
        return case_1.union(case_2)


class SortApproachDataLoader(BaseDataLoader):

    def load_data_for_numba_runtime(self, image_id, split_char):
        return self.load_data(image_id, split_char)
        
    def load_data(self, image_id, split_char):
        self.logger.debug(f'Start loading data')
        p = os.path.join(os.getcwd(), 'data', 'preprocessed', f'gt_{image_id}.txt')
        gts = []
        with open(p, 'r') as f:
            line = f.readline()
            while line:
                gt = parse.parse_data(line, split_char)
                gt = NodeV2(**gt)
                gts.append(gt)
                line = f.readline()
        p = os.path.join(os.getcwd(), 'data', 'preprocessed', f'pred_{image_id}.txt')
        preds = []
        with open(p, 'r') as f:
            line = f.readline()
            while line:
                pred = parse.parse_data(line, split_char)
                pred = NodeV2(**pred)
                preds.append(pred)
                line = f.readline()
        self.logger.debug(f'# of preds: {len(preds)}')
        self.logger.debug(f'# of gts: {len(gts)}')
        return (gts, preds)        


class ApproachSort(Approach):
    
    @classmethod
    def get_data_loader(cls):
        return SortApproachDataLoader()

    def __init__(self, gts, preds, run_type:str='sort_preds_first') -> None:
        super().__init__(gts, preds)
        if run_type == 'sort_gts_first':
            self.run = self.run_1
        elif run_type == 'sort_preds_first':
            self.run = self.run_2
        else:
            raise NotImplementedError
    
    def run_1(self):
        start = time.time()
        
        s = time.time()
        gt_sorted_right = self.sorted(self.gts, key=lambda node: node.right_bottom[0])
        gt_sorted_bottom = self.sorted(self.gts, key=lambda node: node.right_bottom[1])
        gt_sorted_left = self.sorted(self.gts, key=lambda node: node.left_top[0])
        gt_sorted_top = self.sorted(self.gts, key=lambda node: node.left_top[1])
        e = time.time()
        self.logger.debug(f'sorting dt: {e-s:.3f}(s)')

        s = time.time()
        self.binary_search_by_sorted_gts(self.preds, gt_sorted_right, cond='left')
        self.binary_search_by_sorted_gts(self.preds, gt_sorted_bottom, cond='top')
        self.binary_search_by_sorted_gts(self.preds, gt_sorted_left, cond='right')
        self.binary_search_by_sorted_gts(self.preds, gt_sorted_top, cond='bottom')
        e = time.time()
        self.logger.debug(f'filtering dt: {e-s:.3f}(s)')

        s = time.time()
        for pred in self.preds:
            assert len(pred.cond_intersect()) == len(pred.matched)
            for e in pred.cond_intersect():
                if self.is_matched(pred, e):
                    self.add_pair(pred, e)
        e = time.time()
        self.logger.debug(f'matching dt: {e-s:.3f}(s)')

        s = time.time()
        self.cal_matrix()
        e = time.time()
        self.logger.debug(f'cal_matrix dt: {e-s:.3f}(s)')

        end = time.time()
        self.logger.info(f'total dt: {end-start:.2f}(s) (O(nlog(n)), compared {self._n_calculation} times')
        self.logger.info(f'Found {len(self.pairs)} pairs. (F1: {self.matrix.f1:.2f})')
        return self.matrix

    def run_2(self):
        start = time.time()

        s = time.time()
        pred_sorted_left = self.sorted(self.preds, key=lambda node: node.left_top[0])
        pred_sorted_top = self.sorted(self.preds, key=lambda node: node.left_top[1])
        pred_sorted_right = self.sorted(self.preds, key=lambda node: node.right_bottom[0])
        pred_sorted_bottom = self.sorted(self.preds, key=lambda node: node.right_bottom[1])
        e = time.time()
        self.logger.debug(f'sorting dt: {e-s:.3f}(s)')

        s = time.time()
        self.binary_search_by_sorted_preds(pred_sorted_left, self.gts, cond='right')
        self.binary_search_by_sorted_preds(pred_sorted_top, self.gts, cond='bottom')
        self.binary_search_by_sorted_preds(pred_sorted_right, self.gts, cond='left')
        self.binary_search_by_sorted_preds(pred_sorted_bottom, self.gts, cond='top')
        e = time.time()
        self.logger.debug(f'filtering dt: {e-s:.3f}(s)')

        s = time.time()
        for gt in self.gts:
            for e in gt.cond_intersect():
                if self.is_matched(e, gt):
                    self.add_pair(e, gt)
        e = time.time()
        self.logger.debug(f'matching dt: {e-s:.3f}(s)')

        s = time.time()
        self.cal_matrix()
        e = time.time()
        self.logger.debug(f'cal_matrix dt: {e-s:.3f}(s)')

        end = time.time()
        self.logger.info(f'total dt: {end-start:.2f}(s) (O(nlog(n)), compared {self._n_calculation} times')
        self.logger.info(f'Found {len(self.pairs)} pairs. (F1: {self.matrix.f1:.2f})')
        return self.matrix

    def sorted(self, nodes, key, reverse=False):
        # TODO: implement your own sorting algorithm by your own container.
        return sorted(nodes, key=key, reverse=reverse)

    def binary_search_by_sorted_gts(self, preds, sorted_gts, cond):

        def condition(pred, gt, cond):
            if cond == 'left':
                return pred.left_top[0] < gt.right_bottom[0]
            elif cond == 'top':
                return pred.left_top[1] < gt.right_bottom[1]
            elif cond == 'right':
                return pred.right_bottom[0] > gt.left_top[0]
            elif cond == 'bottom':
                return pred.right_bottom[1] > gt.left_top[1]

        def fill(pred, gt, cond):
            if cond == 'left':
                pred.add_candidate_filtered_by_left(gt)
            elif cond == 'top':
                pred.add_candidate_filtered_by_top(gt)
            elif cond == 'right':
                pred.add_candidate_filtered_by_right(gt)
            elif cond == 'bottom':
                pred.add_candidate_filtered_by_bottom(gt)

        for pred in preds:
            _l = 0
            _r = len(sorted_gts)
            while _l < _r:
                _mid = (_l + _r) // 2
                if condition(pred, sorted_gts[_mid], cond):
                    _r = _mid
                else:
                    _l = _mid + 1
            for gt in sorted_gts[_l:]:
                fill(pred, gt, cond)

    def binary_search_by_sorted_preds(self, sorted_preds, gts, cond):

        def condition(pred, gt, cond):
            if cond == 'left':
                return pred.left_top[0] < gt.right_bottom[0]
            elif cond == 'top':
                return pred.left_top[1] < gt.right_bottom[1]
            elif cond == 'right':
                return pred.right_bottom[0] > gt.left_top[0]
            elif cond == 'bottom':
                return pred.right_bottom[1] > gt.left_top[1]

        def fill(gt, pred, cond):
            if cond == 'left':
                gt.add_candidate_filtered_by_left(pred)
            elif cond == 'top':
                gt.add_candidate_filtered_by_top(pred)
            elif cond == 'right':
                gt.add_candidate_filtered_by_right(pred)
            elif cond == 'bottom':
                gt.add_candidate_filtered_by_bottom(pred)

        for gt in gts:
            _l = 0
            _r = len(sorted_preds)
            while _l < _r:
                _mid = (_l + _r) // 2
                if condition(gt, sorted_preds[_mid], cond):
                    _l = _mid + 1
                else:
                    _r = _mid
            for pred in sorted_preds[:_l]:
                fill(gt, pred, cond)