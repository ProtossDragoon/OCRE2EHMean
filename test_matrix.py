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

    def test_fn_iou(self):
        for pred, gt in zip(self.read_pred_data(), self.read_gt_data()):
            polygon_pred = shapely_polygon.get_polygon(*pred[:-1])
            polygon_gt = shapely_polygon.get_polygon(*gt[:-1])
            iou = matrix.Calculation.cal_iou(polygon_pred, polygon_gt)
            # print(f'{matrix.is_true_positive(polygon_pred, polygon_gt)} (IoU: {iou:.3f})')
            # self.assertEqual(iou, 1.0)

    def test_fn_end2end(self):
        polygon1, polygon2 = self.generate_fake_polygon_pair_iou_100()
        ret = matrix.Calculation.end2end(polygon1, polygon2, '안녕', '친구들')
        self.assertEqual(ret, [0,])
        ret = matrix.Calculation.end2end(polygon1, polygon2, '안녕', '안녕')
        self.assertEqual(ret, [1,])
        ret = matrix.Calculation.end2end(polygon1, polygon2, '안녕', '친구들 안녕')
        self.assertEqual(ret, [0, 0,])
        ret = matrix.Calculation.end2end(polygon1, polygon2, '안녕 친구들', '친구들')
        self.assertEqual(ret, [0, 0,])
        ret = matrix.Calculation.end2end(polygon1, polygon2, '친구들 안녕', '잘가 안녕')
        self.assertEqual(ret, [0, 1,])
        ret = matrix.Calculation.end2end(polygon1, polygon2, '안녕', '안녕 친구들')
        self.assertEqual(ret, [1, 0,])
        ret = matrix.Calculation.end2end(polygon1, polygon2, '친구들 안녕', '친구들')
        self.assertEqual(ret, [1, 0,])
        ret = matrix.Calculation.end2end(polygon1, polygon2, '친구들 안녕', '친구들 안녕')
        self.assertEqual(ret, [1, 1,])
        polygon1, polygon2 = self.generate_fake_polygon_pair_iou_30()
        ret = matrix.Calculation.end2end(polygon1, polygon2, '안녕', '친구들')
        self.assertEqual(ret, [])
        ret = matrix.Calculation.end2end(polygon1, polygon2, '안녕', '안녕')
        self.assertEqual(ret, [])

    def test_fn_metric(self):
        polygon1, polygon2 = self.generate_fake_polygon_pair_iou_100()
        m = matrix.Matrix()
        m.update(polygon1, polygon2, '안녕', '친구들')
        self.assertEqual(m.precision, 0.0)
        self.assertEqual(m.recall, 0.0)
        self.assertEqual(m.f1, 0.0)
        m = matrix.Matrix()
        m.update(polygon1, polygon2, '안녕', '안녕')
        self.assertEqual(m.precision, 1.0)
        self.assertEqual(m.recall, 1.0)
        self.assertEqual(m.f1, 1.0)
        m = matrix.Matrix()
        m.update(polygon1, polygon2, '안녕', '안녕')
        m.update(polygon1, polygon2, '안녕', '친구들')
        self.assertEqual(m.precision, 0.5)
        self.assertEqual(m.recall, 0.5)
        self.assertEqual(m.f1, 0.5)

    def test_pipeline(self):
        m = matrix.Matrix()
        for pred, gt in zip(self.read_pred_data(), self.read_gt_data()):
            polygon_pred = shapely_polygon.get_polygon(*pred[:-1])
            polygon_gt = shapely_polygon.get_polygon(*gt[:-1])
            m.update(polygon_pred, polygon_gt, pred[-1], gt[-1])
        print(f'Precision: {m.precision:.3f}')
        print(f'Recall: {m.recall:.3f}')
        print(f'F1: {m.f1:.3f}')


if __name__ == '__main__':
    log.set_default_logger('DEBUG')
    unittest.main()