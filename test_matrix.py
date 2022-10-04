# 프로젝트
import polygon.shapely_polygon as shapely_polygon
import parse
import matrix
import unittest
import runtime
import log


class TestCalculation(unittest.TestCase):

    def test_iou_calculation_cython(self):
        polygon1 = shapely_polygon.get_polygon((0, 0), (0, 1), (1, 1), (1, 0))
        polygon2 = shapely_polygon.get_polygon((0, 0), (0, 1), (1, 1), (1, 0))
        matrix.Calculation.MODE = runtime.PythonRuntime()
        python_iou = matrix.Calculation.cal_iou(polygon1, polygon2)
        matrix.Calculation.MODE = runtime.CythonRuntime()
        cython_iou = matrix.Calculation.cal_iou(polygon1, polygon2)
        self.assertEqual(python_iou, cython_iou)


class TestIoU(unittest.TestCase):

    def generate_fake_polygon_pair_iou_100(self):
        polygon_pred = shapely_polygon.get_polygon((0, 0), (0, 1), (1, 1), (1, 0))
        polygon_gt = shapely_polygon.get_polygon((0, 0), (0, 1), (1, 1), (1, 0))
        return polygon_pred, polygon_gt

    def generate_fake_polygon_pair_iou_30(self):
        polygon_pred = shapely_polygon.get_polygon((0, 0), (0, 1), (1, 1), (1, 0))
        polygon_gt = shapely_polygon.get_polygon((0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5))
        return polygon_pred, polygon_gt

    def read_pred_data(self, path='./data/mock/pred.txt'):
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                yield parse.parse_data(line)
                line = f.readline()

    def read_gt_data(self, path='./data/mock/gt.txt'):
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                yield parse.parse_data(line)
                line = f.readline()

    def test_text_recognition(self):
        sentence_preds = [
            '안녕', '안녕', '안녕',
            '안녕 친구들', '친구들 안녕', '안녕',
            '친구들 안녕', '친구들 안녕',
        ]
        sentence_gts = [
            '친구들', '안녕', '친구들 안녕',
            '친구들', '잘가 안녕', '안녕 친구들',
            '친구들', '친구들 안녕',
        ]
        expected_li = [
            [0,], [1,], [0, 0,],
            [0, 0,], [0, 1,], [1, 0,],
            [1, 0,], [1, 1,]
        ]
        for (sentence_pred, sentence_gt, expected) in zip(
            sentence_preds, sentence_gts, expected_li):
            ret = matrix.Calculation.text_recognition(sentence_pred, sentence_gt)
            self.assertEqual(ret, expected)
            ret = matrix.Calculation.text_recognition(sentence_gt, sentence_pred)
            self.assertEqual(ret, expected)


if __name__ == '__main__':
    log.set_default_logger('DEBUG')
    unittest.main()