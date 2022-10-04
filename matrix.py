# 내부
import itertools

# 서드파티
import numba
import shapely
import shapely.geometry

# 프로젝트
from runtime import GlobalRuntime
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
    def is_true_positive(polygon1, polygon2):
        if GlobalRuntime.is_python_runtime():
            return Calculation._is_true_positive_python(polygon1, polygon2)
        elif GlobalRuntime.is_numba_runtime():
            return Calculation._is_ture_positive_numba(polygon1, polygon2)
        else:
            raise NotImplementedError(f'Not implemented  {GlobalRuntime.runtime}')

    @staticmethod
    def _is_true_positive_python(polygon1, polygon2):
        iou = Calculation.cal_iou(polygon1, polygon2)
        return True if iou > 0.5 else False

    @staticmethod
    def _is_ture_positive_numba(polygon1, polygon2):
        iou = Calculation.cal_iou(polygon1, polygon2)
        return True if iou > 0.5 else False

    @staticmethod
    def _is_true_positive_cython(polygon1, polygon2):
        raise NotImplementedError

    @staticmethod
    def _is_true_positive_numpy(polygon1, polygon2):
        raise NotImplementedError

    @staticmethod
    def _is_true_positive_jax(polygon1, polygon2):
        raise NotImplementedError

    @staticmethod
    def end2end(polygon1, polygon2, sentence_1, sentence_2, check_iou=True):
        if GlobalRuntime.is_python_runtime():
            return Calculation._end2end_python(polygon1, polygon2, sentence_1, sentence_2, check_iou)
        elif GlobalRuntime.is_numba_runtime():
            return Calculation._end2end_numba(polygon1, polygon2, sentence_1, sentence_2, check_iou)
        else:
            raise NotImplementedError(f'Not implemented  {GlobalRuntime.runtime}')

    @staticmethod
    def _end2end_python(polygon1, polygon2, sentence_1, sentence_2, check_iou):
        tp = True
        if check_iou:
            tp = True if Calculation.is_true_positive(polygon1, polygon2) else False
        if tp:
            return Calculation.transcription_equal(sentence_1, sentence_2)
        else:
            return []

    @staticmethod
    def _end2end_numba(polygon1, polygon2, sentence_1, sentence_2, check_iou):
        tp = True
        if check_iou:
            tp = True if Calculation.is_true_positive(polygon1, polygon2) else False
        if tp:
            return Calculation.transcription_equal(sentence_1, sentence_2)
        else:
            return []

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

    def __init__(self):
        self.n_words_pred = 0
        self.n_words_gt = 0
        self.n_words_correct = 0
    
    @property
    def precision(self):
        if self.n_words_pred == 0:
            return 0
        return self.n_words_correct / self.n_words_pred

    @property
    def recall(self):
        if self.n_words_gt == 0:
            return 0
        return self.n_words_correct / self.n_words_gt

    @property
    def f1(self):
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0
        return 2*p*r / (p+r)

    def update(self, polygon_pred, polygon_gt, sentence_pred, sentence_gt, check_iou=True):
        assert type(sentence_gt) is str
        assert type(sentence_pred) is str
        DONT_CARE = ''
        if sentence_gt == DONT_CARE:
            return
        result_mask = Calculation.end2end(polygon_pred, polygon_gt, sentence_pred, sentence_gt, check_iou=check_iou)
        if result_mask:
            self.n_words_pred += len(sentence_pred.split())
            self.n_words_gt += len(sentence_gt.split())
            self.n_words_correct += sum(result_mask)
            
    @classmethod
    def merge(cls, matrix_list: list):
        ret = cls()
        for matrix in matrix_list:
            ret.n_words_correct += matrix.n_words_correct
            ret.n_words_gt += matrix.n_words_gt
            ret.n_words_pred += matrix.n_words_pred
        return ret