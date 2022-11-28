# 내장
import os

# 프로젝트
import polygon.shapely_polygon as shapely_polygon
import parse
import matrix
import unittest
import runtime
import log
import approach


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

    def test_transcription_equal(self):
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
            ret = matrix.Calculation.transcription_equal(sentence_pred, sentence_gt)
            self.assertEqual(ret, expected)
            ret = matrix.Calculation.transcription_equal(sentence_gt, sentence_pred)
            self.assertEqual(ret, expected)


class TextMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ret_0 = approach.start(
            0,
            runtime.PythonRuntime,
            approach.ApproachNaive,
            base_dir=os.path.join(os.getcwd(), 'data', 'test', 'preprocessed'))
        cls.ret_1 = approach.start(
            1,
            runtime.PythonRuntime,
            approach.ApproachNaive,
            base_dir=os.path.join(os.getcwd(), 'data', 'test', 'preprocessed'))
        cls.ret_2 = approach.start(
            2,
            runtime.PythonRuntime,
            approach.ApproachNaive,
            base_dir=os.path.join(os.getcwd(), 'data', 'test', 'preprocessed'))
        cls.ret_3 = approach.start(
            3,
            runtime.PythonRuntime,
            approach.ApproachNaive,
            base_dir=os.path.join(os.getcwd(), 'data', 'test', 'preprocessed'))
        cls.ret_4 = approach.start(
            4,
            runtime.PythonRuntime,
            approach.ApproachNaive,
            base_dir=os.path.join(os.getcwd(), 'data', 'test', 'preprocessed'))

    def test_e2e_pricision(self):
        with self.subTest('one-to-one'):
            self.assertAlmostEqual(self.ret_0.precision, 1.0)
        with self.subTest('one-to-one'):
            self.assertAlmostEqual(self.ret_1.precision, 2/4)
        with self.subTest('many-to-one'):
            self.assertAlmostEqual(self.ret_2.precision, 2/5)
        with self.subTest('many-to-one'):
            self.assertAlmostEqual(self.ret_3.precision, 1)
        with self.subTest('many-to-many'):
            self.assertAlmostEqual(self.ret_4.precision, 3/4)

    def test_e2e_recall(self):
        with self.subTest('one-to-one'):
            self.assertAlmostEqual(self.ret_0.recall, 1.0)
        with self.subTest('one-to-one'):
            self.assertAlmostEqual(self.ret_1.recall, 2/5)
        with self.subTest('many-to-one'):
            self.assertAlmostEqual(self.ret_2.recall, 1/1)
        with self.subTest('many-to-one'):
            self.assertAlmostEqual(self.ret_3.recall, 0)
        with self.subTest('many-to-many'):
            self.assertAlmostEqual(self.ret_4.recall, 2/3)

    def test_e2e_f1(self):
        with self.subTest('one-to-one'):
            self.assertAlmostEqual(self.ret_0.f1, 1.0)
        with self.subTest('one-to-one'):
            self.assertAlmostEqual(self.ret_1.f1, 4/9)
        with self.subTest('many-to-one'):
            self.assertAlmostEqual(self.ret_2.f1, 2*(2/5)*(1/1)/((2/5)+(1/1)))
        with self.subTest('many-to-one'):
            self.assertAlmostEqual(self.ret_3.f1, 0)
        with self.subTest('many-to-many'):
            self.assertAlmostEqual(self.ret_4.f1, 2*(2/3)*(3/4)/((2/3)+(3/4)))

if __name__ == '__main__':
    log.set_default_logger('DEBUG')
    unittest.main()