# 내장
import time
import unittest

# 서드파티
import tqdm

# 프로젝트
import log
import runtime
import approach
from approach import start
from parallel import start_parallel, wait


class RaySpeedTest(unittest.TestCase):
    
    def setUp(self):
        self.image_ids = range(1, 10)
    
    def dt_sigle_core(self, *args, **kwargs) -> float:
        print('Single-core processing start.')
        s = time.time()
        for image_id in tqdm.tqdm(self.image_ids):
            _ = start(
                image_id, *args, **kwargs
            )
        e = time.time()
        time_single_process = e - s
        return time_single_process
    
    def dt_multi_core(self, *args, **kwargs) -> float:
        print('Multi-core processing start.')
        s = time.time()
        promise_li = []
        for image_id in self.image_ids:
            promise_li.append(start_parallel.remote(
                image_id, *args, **kwargs
            ))
        wait(promise_li)
        e = time.time()
        time_multi_process = e - s
        return time_multi_process

    def test_parallel_performance(self):
        with self.subTest('Naive Approach, No Optimization'):
            print('\nNaive Approach, No Optimization')
            time_single_process = self.dt_sigle_core(
                runtime.PythonRuntime, approach.ApproachNaive)
            time_multi_process = self.dt_multi_core(
                runtime.PythonRuntime, approach.ApproachNaive)
            self.assertGreater(time_single_process, time_multi_process)
        with self.subTest('Naive Approach + Optimization'):
            print('\nNaive Approach + Optimization')
            time_single_process = self.dt_sigle_core(
                runtime.NumbaRuntime, approach.ApproachNaive)
            time_multi_process = self.dt_multi_core(
                runtime.NumbaRuntime, approach.ApproachNaive)
            self.assertGreater(time_single_process, time_multi_process)
        with self.subTest('Sort Approach, No Optimization'):
            print('\nSort Approach, No Optimization')
            time_single_process = self.dt_sigle_core(
                runtime.PythonRuntime, approach.ApproachSort)
            time_multi_process = self.dt_multi_core(
                runtime.PythonRuntime, approach.ApproachSort)
            self.assertGreater(time_single_process, time_multi_process)
        with self.subTest('Sort Approach + Optimization'):
            print('\nSort Approach + Optimization')
            time_single_process = self.dt_sigle_core(
                runtime.NumbaRuntime, approach.ApproachSort)
            time_multi_process = self.dt_multi_core(
                runtime.NumbaRuntime, approach.ApproachSort)
            self.assertGreater(time_single_process, time_multi_process)
    
    
if __name__ == '__main__':
    log.set_default_logger('WARNING')
    unittest.main()