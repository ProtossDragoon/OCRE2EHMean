# 내장
import log
import tqdm

# 프로젝트
import approach
import runtime
import matrix

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
    matrix_li = parallel.wait(promise_li)
    metric = matrix.Matrix.merge(matrix_li)
    print(f'Found {metric.n_words_pred} preds and {metric.n_words_gt} gts')
    print(f'Precision:\t {metric.precision:4f}')
    print(f'Recall:   \t {metric.recall:4f}')
    print(f'F1:       \t {metric.f1:4f}')