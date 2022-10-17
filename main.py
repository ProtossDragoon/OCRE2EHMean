# 내장
import log
import tqdm
import re, os, glob

# 프로젝트
import approach
import runtime
import matrix

# 프로젝트
import parallel


def get_number_of_images(dir_path :str = './data/preprocessed'):
    def sort_by_number_inside_of_string(li: list):
        li.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        return li
    gt_files = glob.glob(os.path.join(dir_path, 'gt_*.txt'))
    pred_files = glob.glob(os.path.join(dir_path, 'pred_*.txt'))
    sort_by_number_inside_of_string(gt_files)
    sort_by_number_inside_of_string(pred_files)
    assert int(re.sub(r'[^0-9]', '', gt_files[-1])) == (len(gt_files)-1), f'gt files are not continuous ({gt_files[-1]},+1) != count(gt_files) ({len(gt_files)})'
    assert int(re.sub(r'[^0-9]', '', pred_files[-1])) == (len(pred_files)-1), f'pred files are not continuous ({pred_files[-1]},+1) != count(pred_files) ({len(pred_files)})'
    assert len(gt_files) == len(pred_files)
    return len(gt_files)


if __name__ == '__main__':
    log.set_default_logger('WARNING')
    promise_li = []
    for image_id in tqdm.tqdm(range(0, get_number_of_images())):
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