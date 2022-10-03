import numpy as np
import time
import numba
import random
import math

# 프로젝트
import trial.common
from polygon.shapely_polygon import get_polygon


def get_distance(p1, p2):
    # 두 점의 거리를 구합니다.
    sub1 = (p2[0] - p1[0])
    sub1 = sub1 ** 2
    sub2 = (p2[1] - p1[1])
    sub2 = sub2 ** 2
    return math.sqrt(sub1 + sub2)


@numba.jit(nopython=True)
def get_distance_numba(p1, p2):
    # 두 점의 거리를 구합니다.
    sub1 = (p2[0] - p1[0])
    sub1 = sub1 ** 2
    sub2 = (p2[1] - p1[1])
    sub2 = sub2 ** 2
    return math.sqrt(sub1 + sub2)

def get_distance_np(p1, p2):
    # 두 점의 거리를 구합니다.
    return np.sqrt(np.sum((p2 - p1) ** 2))

@numba.jit(nopython=True)
def get_distance_np_numba(p1, p2):
    # 두 점의 거리를 구합니다.
    return np.sqrt(np.sum((p2 - p1) ** 2))

def get_polygon_area(polygon: list):
    area = 0
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        area += (p1[0] * p2[1] - p2[0] * p1[1])
    return abs(area) / 2

@numba.jit(nopython=True)
def get_polygon_area_numba(polygon: list):
    area = 0
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        area += (p1[0] * p2[1] - p2[0] * p1[1])
    return abs(area) / 2

def get_polygon_area_np(polygon: np.ndarray):
    a = polygon[:, 0] * np.roll(polygon[:, 1], 1)
    b = np.roll(polygon[:, 0], 1) * polygon[:, 1]
    area = np.sum(a - b)
    return np.abs(area) / 2

@numba.jit(nopython=True)
def get_polygon_area_np_numba(polygon: np.ndarray):
    a = polygon[:, 0] * np.roll(polygon[:, 1], 1)
    b = np.roll(polygon[:, 0], 1) * polygon[:, 1]
    area = np.sum(a - b)
    return np.abs(area) / 2

def get_intersection(line1_p1, line1_p2, line2_p1, line2_p2):
    m1 = (line1_p2[1] - line1_p1[1]) / (line1_p2[0] - line1_p1[0])
    m2 = (line2_p2[1] - line2_p1[1]) / (line2_p2[0] - line2_p1[0])
    b1 = line1_p1[1] - m1 * line1_p1[0]
    b2 = line2_p1[1] - m2 * line2_p1[0]
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    if m1 == m2:
        return ()
    return (x, y)

@numba.jit(nopython=True)
def get_intersection_numba(line1_p1, line1_p2, line2_p1, line2_p2):
    m1 = (line1_p2[1] - line1_p1[1]) / (line1_p2[0] - line1_p1[0])
    m2 = (line2_p2[1] - line2_p1[1]) / (line2_p2[0] - line2_p1[0])
    b1 = line1_p1[1] - m1 * line1_p1[0]
    b2 = line2_p1[1] - m2 * line2_p1[0]
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    if m1 == m2:
        return ()
    return (x, y)


if __name__ == '__main__':

    n = 100000
    print('fn : get_distance()')
    it = iter(trial.common.point_generator())
    numpy_it = iter(trial.common.point_generator('numpy'))

    s = time.time()
    for _ in range(n):
        get_distance(next(it), next(it))
    e = time.time()
    print(f'plain data\tplain fn\t: {e - s:5f}')

    s = time.time()
    for _ in range(n):
        get_distance_numba(next(it), next(it))
    e = time.time()
    print(f'plain data\tplain fn (@)\t: {e - s:5f}')

    s = time.time()
    for _ in range(n):
        get_distance_numba(next(numpy_it), next(numpy_it))
    e = time.time()
    print(f'numpy data\tplain fn (@)\t: {e - s:5f}')

    s = time.time()
    for _ in range(n):
        get_distance_np(next(numpy_it), next(numpy_it))
    e = time.time()
    print(f'numpy data\tnumpy fn\t: {e - s:5f}')

    s = time.time()
    for _ in range(n):
        get_distance_np_numba(next(numpy_it), next(numpy_it))
    e = time.time()
    print(f'numpy data\tnumpy fn (@)\t: {e - s:5f}')

    print('fn : get_polygon_area()')
    it = iter(trial.common.polygon_generator())
    numpy_it = iter(trial.common.polygon_generator('numpy'))

    s = time.time()
    for _ in range(n):
        get_polygon(*next(it)).area
    e = time.time()
    print(f'plain data\tshapely fn\t: {e - s:5f}')

    s = time.time()
    for _ in range(n):
        get_polygon_area(next(it))
    e = time.time()
    print(f'plain data\tplain fn\t: {e - s:5f}')

    s = time.time()
    for _ in range(n):
        get_polygon_area_numba(next(it))
    e = time.time()
    print(f'plain data\tplain fn (@)\t: {e - s:5f}')

    s = time.time()
    for _ in range(n):
        get_polygon_area_np(next(numpy_it))
    e = time.time()
    print(f'numpy data\tplain fn (@)\t: {e - s:5f}')

    s = time.time()
    for _ in range(n):
        get_polygon_area_np(next(numpy_it))
    e = time.time()
    print(f'numpy data\tnumpy fn\t: {e - s:5f}')

    s = time.time()
    for _ in range(n):
        get_polygon_area_np_numba(next(numpy_it))
    e = time.time()
    print(f'numpy data\tnumpy fn (@)\t: {e - s:5f}')

    print('fn : get_intersection()')
    it = iter(trial.common.point_generator())
    numpy_it = iter(trial.common.point_generator('numpy'))

    s = time.time()
    for _ in range(n):
        get_distance(next(it), next(it))
    e = time.time()
    print(f'plain data\tplain fn\t: {e - s:5f}')

    s = time.time()
    for _ in range(n):
        get_distance_numba(next(it), next(it))
    e = time.time()
    print(f'plain data\tplain fn (@)\t: {e - s:5f}')