# 내장
import os
import unittest

# 프로젝트
import log
import parse


class TestDataUtil(unittest.TestCase):
    
    def test_parse_data(self):
        split_char = ','

        p = os.path.join(os.path.dirname(__file__), 'data', 'real', 'preprocessed', 'gt_0.txt')
        with open(p, 'r') as f:
            gt_it = iter([
                'CARTOLER', 
                '###',
                '###',
                '###',
                'T',
                '###',
                '###',
                '12',
                '###',
                'PISTONE',
                '###',
                '###',
                '###'])
            line = f.readline()
            while line:
                data_gt = parse.parse_data(line, split_char)
                self.assertEqual(data_gt['sentence'], next(gt_it))
                line = f.readline()
        
        p = os.path.join(os.path.dirname(__file__), 'data', 'real', 'preprocessed', 'pred_0.txt')
        with open(p, 'r') as f:
            prep_it = iter([
                'CARTOLER', 
                '###',
                '###',
                '###',
                'T',
                '###',
                '###',
                '12',
                '###',
                'PISTONE',
                '###',
                '###',
                '###'])
            line = f.readline()
            while line:
                data_prep = parse.parse_data(line, split_char)
                self.assertEqual(data_prep['sentence'], next(prep_it))
                line = f.readline()


if __name__ == '__main__':
    log.set_default_logger('DEBUG')
    unittest.main()