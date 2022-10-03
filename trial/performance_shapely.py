# 내장
import time
import pprint

# 프로젝트
import trial.common
from polygon.shapely_polygon import get_polygon


if __name__ == '__main__':
    n_iter = 100000

    print('fn : IoU()')
    it = iter(trial.common.polygon_generator())
    record = {
        'create_poly_1': 0,
        'create_poly_2': 0,
        'get_intersection': 0,
        'get_union': 0,
    }

    global_s = time.time()
    for _ in range(n_iter):
        s = time.time()
        poly_1 = get_polygon(*next(it))
        e = time.time()
        record['create_poly_1'] += e - s

        s = time.time()
        poly_2 = get_polygon(*next(it))
        e = time.time()
        record['create_poly_2'] += e - s

        s = time.time()
        intersection = poly_1.intersection(poly_2).area
        e = time.time()
        record['get_intersection'] += e - s

        s = time.time()
        union = poly_1.intersection(poly_2).area
        e = time.time()
        record['get_union'] += e - s

        iou = intersection / union
    global_e = time.time()
    print(f'plain data\tshapely fn\t: {global_e - global_s:5f}')
    pprint.pprint(record, indent=4)

    print('fn : get_distance()')
    it = iter(trial.common.point_generator())
