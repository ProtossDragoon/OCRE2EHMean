# 내부
import itertools

# 서드파티
import numba
import shapely
import shapely.geometry

# 프로젝트
from runtime import GlobalRuntime
from container.linked_list import Node
from polygon.shapely_polygon import iou as iou_shapely
from polygon.python_polygon import iou as iou_original
from polygon.python_polygon_optimized import iou as iou_optimized


class Calculation():

    @staticmethod
    def transcription_equal(sentence_1, sentence_2):
        if GlobalRuntime.is_python_runtime():
            return Calculation._transcription_equal_python(sentence_1, sentence_2)
        elif GlobalRuntime.is_numba_runtime():
            return Calculation._transcription_equal_numba(sentence_1, sentence_2)
        else:
            raise NotImplementedError(f'Not implemented  {GlobalRuntime.runtime}')

    @staticmethod
    def _transcription_equal_python(sentence_1, sentence_2):
        result = []
        words_1 = sentence_1.split()
        words_2 = sentence_2.split()
        for word_1, word_2 in itertools.zip_longest(words_1, words_2, fillvalue=''):
            if word_1 == word_2:
                result.append(1)
            else:
                result.append(0)
        return result

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _transcription_equal_numba(sentence_1, sentence_2):
        result = []
        words_1 = sentence_1.split()
        words_2 = sentence_2.split()

        # pad '' to the shorter sentence
        if len(words_1) > len(words_2):
            words_2 += [''] * (len(words_1) - len(words_2))
        elif len(words_1) < len(words_2):
            words_1 += [''] * (len(words_2) - len(words_1))

        for word_1, word_2 in zip(words_1, words_2):
            if word_1 == word_2:
                result.append(1)
            else:
                result.append(0)
        return result

    @staticmethod
    def is_iou_tp(polygon1, polygon2):
        if GlobalRuntime.is_python_runtime():
            return Calculation._is_iou_tp_python(polygon1, polygon2)
        elif GlobalRuntime.is_numba_runtime():
            return Calculation._is_iou_tp_numba(polygon1, polygon2)
        else:
            raise NotImplementedError(f'Not implemented  {GlobalRuntime.runtime}')

    @staticmethod
    def _is_iou_tp_python(polygon1, polygon2):
        iou = Calculation.cal_iou(polygon1, polygon2)
        return True if iou > 0.5 else False

    @staticmethod
    def _is_iou_tp_numba(polygon1, polygon2):
        iou = Calculation.cal_iou(polygon1, polygon2)
        return True if iou > 0.5 else False

    @staticmethod
    def _is_iou_tp_cython(polygon1, polygon2):
        raise NotImplementedError

    @staticmethod
    def _is_iou_tp_numpy(polygon1, polygon2):
        raise NotImplementedError

    @staticmethod
    def _is_iou_tp_jax(polygon1, polygon2):
        raise NotImplementedError

    @staticmethod
    def cal_iou(polygon1, polygon2):
        if GlobalRuntime.is_python_runtime():
            return Calculation._cal_iou_python(polygon1, polygon2)
        elif GlobalRuntime.is_numba_runtime():
            return Calculation._cal_iou_numba(polygon1, polygon2)
        else:
            raise NotImplementedError(f'Not implemented  {GlobalRuntime.runtime}')

    @staticmethod
    def _cal_iou_python(polygon1, polygon2):
        if type(polygon1) is shapely.geometry.Polygon:
            return iou_shapely(polygon1, polygon2)
        else:
            return iou_original(polygon1, polygon2)
            
    @staticmethod
    def _cal_iou_numba(polygon1, polygon2):
        return iou_optimized(polygon1, polygon2)

    @staticmethod
    def _cal_iou_cython(polygon1, polygon2):
        raise NotImplementedError

    @staticmethod
    def _cal_iou_numpy(polygon1, polygon2):
        raise NotImplementedError

    @staticmethod
    def _cal_iou_jax(polygon1, polygon2):
        raise NotImplementedError


class Matrix():
    
    DONTCARE_INDICATOR = '###'

    def __init__(self):
        # NOTE: 전체 예측 bbox 의 수
        self.n_words_pred = 0 
        
        # NOTE: 전체 정답 bbox 의 수 - don’t care bbox 의 수
        self.n_words_gt = 0 
        
        # NOTE: (1) IoU > 0.5, (2) 텍스트 일치 → 정답값과 매칭 쌍이 생성된 예측 bbox 의 수
        self.n_words_matched_pred = 0 
        
        # NOTE: (1) IoU > 0.5, (2) 텍스트 일치 → 예측값과 매칭 쌍이 생성된 정답 bbox 의 수
        self.n_words_matched_gt = 0
        
        # NOTE: 매칭 쌍이 생성되지 않은 bbox 중, (1) IoU > 0.5, ~(2) 인 대신 (3) dontcare bbox 에 예측을 제시한 bbox 의 수
        self.n_words_passediou_but_dontcare_pred = 0 
    
    @property
    def precision(self):
        if self.n_words_pred == 0:
            return 1
        # NOTE: 전체 예측 bbox 의 수 - don’t care 가 아닌 예측 bbox 의 수
        n_words_notdontcare = self.n_words_pred - self.n_words_passediou_but_dontcare_pred
        if n_words_notdontcare == 0:
            return 1
        return self.n_words_matched_pred / n_words_notdontcare

    @property
    def recall(self):
        if self.n_words_gt == 0:
            return 0
        return self.n_words_matched_gt / self.n_words_gt

    @property
    def f1(self):
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0
        return 2*p*r / (p+r)
    
    def update(self, preds, gts):
        # preds, gts are list or LinkedList
        for pred in preds:
            if pred.is_connected:
                # pred: iou > 0.5, 
                # sentence is equal
                self.n_words_matched_pred += 1 
            else:
                if pred.is_connected_dontcare:
                    # pred: iou > 0.5,
                    # there is no gt connected to pred
                    # only a iou > 0.5 gt is dontcare
                    self.n_words_passediou_but_dontcare_pred += 1
        for gt in gts:
            if gt.is_connected:
                self.n_words_matched_gt += 1

    def is_end2end_tp(self, pred: Node, gt: Node, check_iou=True):
        iou_tp = True
        result_mask = []
        if check_iou:
            iou_tp = Calculation.is_iou_tp(pred.polygon, gt.polygon)
        if iou_tp:
            result_mask = Calculation.transcription_equal(pred.sentence, gt.sentence)
        if not result_mask:
            # result_mask 가 비어 있으면
            # 문자 영역을 올바르게 찾아내지 못한 것임.
            return False
        else:
            if sum(result_mask) == len(result_mask):
                # 찾아낸 단어들을 하나도 빠짐없이 맞추었다면
                return True
            return False
        
    def is_dontcare(self, pred: Node, gt: Node, check_iou=True):
        iou_tp = True
        if check_iou:
            iou_tp = Calculation.is_iou_tp(pred.polygon, gt.polygon)
        if iou_tp:
            if gt.sentence == Matrix.DONTCARE_INDICATOR:
                return True
            return False

    @classmethod
    def merge(cls, matrix_list: list):
        ret = cls()
        for matrix in matrix_list:
            ret.n_words_gt += matrix.n_words_gt
            ret.n_words_pred += matrix.n_words_pred
            ret.n_words_matched_gt += matrix.n_words_matched_gt
            ret.n_words_matched_pred += matrix.n_words_matched_pred
            ret.n_words_passediou_but_dontcare_pred += matrix.n_words_passediou_but_dontcare_pred
        return ret