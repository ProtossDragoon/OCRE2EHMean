# 내장
import os
import unittest

# 프로젝트
import log
import parse


class TestDataUtil(unittest.TestCase):
    
    def test_parse_data_extended(self):
        split_char = ',  '

        p = os.path.join(os.path.dirname(__file__), 'data', 'mock', 'gt_extended.txt')
        with open(p, 'r') as f:
            gt_it = iter(['·', '이것은 테스트입니다.', '이거, 잘, 동작하나요?'])
            line = f.readline()
            while line:
                data_gt = parse.parse_data_extended(line, split_char)
                self.assertEqual(data_gt['sentence'], next(gt_it))
                line = f.readline()
        
        p = os.path.join(os.path.dirname(__file__), 'data', 'mock', 'pred_extended.txt')
        with open(p, 'r') as f:
            prep_it = iter(['·', '제품명', ':'])
            line = f.readline()
            while line:
                data_prep = parse.parse_data_extended(line, split_char)
                self.assertEqual(data_prep['sentence'], next(prep_it))
                line = f.readline()

        self.assertEqual(data_gt['image_w'], data_prep['image_w'])
        self.assertEqual(data_gt['image_h'], data_prep['image_h'])


if __name__ == '__main__':
    log.set_default_logger('DEBUG')
    unittest.main()