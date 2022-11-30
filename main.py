# 내장
import log
import tqdm
import re, os, glob
import logging
import argparse

# 프로젝트
import approach
import runtime
import matrix

# 프로젝트
import parallel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_base_dir')
    parser.add_argument('pred_base_dir')
    parser.add_argument('matrix_save_dir')
    parser.add_argument('--logfile_path', default='e2e.log', type=str)
    args = parser.parse_args()
    return args


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


def get_image_names(
    gt_base_dir: str,
    pred_base_dir: str,
):
    # load file names from gt_base_dir
    gt_files = os.listdir(gt_base_dir)
    pred_files = os.listdir(pred_base_dir)
    gt_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    pred_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    assert len(gt_files) == len(pred_files), (
        f'gt_files and pred_files are not same length ({len(gt_files)} != {len(pred_files)})')
    for gt_file, pred_file in (zip(gt_files, pred_files)):
        assert gt_file.split('/')[-1] == pred_file.split('/')[-1], (
            f'gt_file ({gt_file}) and pred_file ({pred_file}) are not matched')
    return zip(gt_files, pred_files)


if __name__ == '__main__':
    args = parse_args()
    log.set_default_logger(level='INFO', logfile_path=args.logfile_path)
    logger = logging.getLogger('e2e_f1')
    logger.info('End-to-End F1 Score 평가 시작')
    promise_li = []
    for i, (gt_file_name, pred_file_name) in enumerate(tqdm.tqdm(get_image_names(
        args.gt_base_dir,
        args.pred_base_dir,
    ))):
        promise_li.append(parallel.start_parallel.remote(
            i,
            runtime.NumbaRuntime, 
            approach.ApproachSort,
            gt_base_dir=args.gt_base_dir,
            pred_base_dir=args.pred_base_dir,
            gt_name=gt_file_name,
            pred_name=pred_file_name,
            matrix_save_dir=args.matrix_save_dir,
        ))
    matrix_li = parallel.wait(promise_li)
    mat = matrix.Matrix.merge(matrix_li)
    logger.info(f'Found {mat.n_words_pred} preds and {mat.n_words_gt} gts')
    logger.info(f'Precision:\t {mat.precision:4f}')
    logger.info(f'Recall:   \t {mat.recall:4f}')
    logger.info(f'F1:       \t {mat.f1:4f}')
    mat.save(args.matrix_save_dir, 'total.json')