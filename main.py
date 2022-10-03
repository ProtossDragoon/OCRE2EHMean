# 내장
import log
import tqdm

# 프로젝트
import approach
import runtime

# 프로젝트
import parallel


if __name__ == '__main__':
    log.set_default_logger('WARNING')
    promise_li = []
    for image_id in tqdm.tqdm(range(0, 30)):
        promise_li.append(parallel.start_parallel.remote(
            image_id,
            runtime.NumbaRuntime, 
            approach.ApproachSort,
        ))
    parallel.wait(promise_li)