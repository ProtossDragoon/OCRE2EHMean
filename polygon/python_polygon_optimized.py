import numba


@numba.jit(cache=True, nopython=True)
def hg(p: tuple) -> tuple:
    return (p[0], p[1], 1)


@numba.jit(cache=True, nopython=True)
def cross_2d(
    p1: tuple, 
    p2: tuple, 
    p3: tuple,
) -> float:
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])


@numba.jit(cache=True, nopython=True)
def cross_3d(
    p1: tuple,
    p2: tuple,
    p3: tuple,
) -> float:
    return (
        (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]),
        (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]),
        (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]),
    )


@numba.jit(cache=True, nopython=True)
def is_point_on_segment(
    point: tuple,
    segment: tuple,
) -> bool:
    """ccw 알고리즘을 이용해 점이 선분 위에 놓여 있는지 확인합니다.

    Args:
        point (tuple): 선분 위에 존재하는지 확인할 점
        segment (tuple): 점을 포함하는지 확인할 선분

    Returns:
        bool: 선분 위에 존재하는 경우 True 를 반환
    """
    p1, p2 = segment
    ret = cross_2d(p1, p2, point)
    EPSILON = 1e-6
    if ret < EPSILON:
        condition1 = min(p1[0], p2[0]) <= point[0] <= max(p1[0], p2[0])
        condition2 = min(p1[1], p2[1]) <= point[1] <= max(p1[1], p2[1])
        if condition1 and condition2:
            return True
        else:
            return False
    else:    
        return False


@numba.jit(cache=True, nopython=True)
def does_two_segments_intersect(
    segment1: tuple,
    segment2: tuple,
) -> bool:
    """ccw 알고리즘을 이용해 두 선분이 교차하는지 확인합니다.
    두 선분이 겹치는 부분의 길이를 측정할 수 있거나,
    한 점에서만 겹치는 상황에서는 교차하는 것으로 간주합니다.

    Args:
        segment1 (tuple): 교차하는지 확인할 선분
        segment2 (tuple): 교차하는지 확인할 선분

    Returns:
        bool: 교차하는 경우 True 를 반환
    """
    p1, p2 = segment1
    p3, p4 = segment2
    ret1 = cross_2d(p1, p2, p3) * cross_2d(p1, p2, p4)
    ret2 = cross_2d(p3, p4, p1) * cross_2d(p3, p4, p2)
    if ret1 > 0 or ret2 > 0:
        # 두 선분이 자명하게 교차하지 않는 경우
        return False
    elif ret1 == 0 and ret2 == 0:
        # 네 점이 하나의 직선 위에 있는 경우에 해당합니다.
        # 네 점 중 겹치는 부분이 있는지 검사합니다.
        cond1 = is_point_on_segment(p1, segment2)
        cond2 = is_point_on_segment(p2, segment2)
        cond3 = is_point_on_segment(p3, segment1)
        cond4 = is_point_on_segment(p4, segment1)
        if cond1 or cond2 or cond3 or cond4:
            return True
        else:
            return False
    elif ret1 == 0 or ret2 == 0:
        # 다음 케이스들을 포함합니다.
        if ret1 == 0:
            # 점 p1, p2, p3 가 한 직선 위에 있거나, p1, p2, p4 가 한 직선 위에 있는 경우
            if is_point_on_segment(p3, segment1):
                return True # 평행하지 않은 두 선분이 한 점에서만 겹치는 경우
            elif is_point_on_segment(p4, segment1):
                return True # 평행하지 않은 두 선분이 한 점에서만 겹치는 경우
            else:
                return False # (주의) 두 선분 중 한 선분의 연장선이 다른 선분의 끝점을 포함하는 경우
        else: # ret2 == 0
            # 점 p3, p4, p1 가 한 직선 위에 있거나, p3, p4, p2 가 한 직선 위에 있는 경우
            if is_point_on_segment(p1, segment2):
                return True # 평행하지 않은 두 선분이 한 점에서만 겹치는 경우
            elif is_point_on_segment(p2, segment2):
                return True # 평행하지 않은 두 선분이 한 점에서만 겹치는 경우
            else:
                return False # (주의) 두 선분 중 한 선분의 연장선이 다른 선분의 끝점을 포함하는 경우
    else:
        return True


@numba.jit(cache=True, nopython=True)
def is_point_in_polygon(
    polygon: list, 
    point: tuple
) -> bool:
    """점이 사각형 내부에 놓여 있는지 확인합니다.

    Args:
        polygon (list): 점을 포함하는지 확인할 사각형
        point (tuple): 내부에 존재하는지 확인할 점

    Returns:
        bool: 내부에 존재하는 경우 True 를 반환
    """
    # 1. polygon 의 가장 왼쪽, 가장 오른쪽, 가장 위쪽, 가장 아래쪽의 점을 구합니다.
    min_x = min(polygon[0][0], polygon[1][0], polygon[2][0], polygon[3][0])
    max_x = max(polygon[0][0], polygon[1][0], polygon[2][0], polygon[3][0])
    min_y = min(polygon[0][1], polygon[1][1], polygon[2][1], polygon[3][1])
    max_y = max(polygon[0][1], polygon[1][1], polygon[2][1], polygon[3][1])
    # 2. point 가 polygon 의 사각형 내부에 존재하는지 확인합니다.
    if not (min_x < point[0] < max_x and min_y < point[1] < max_y):
        return False
    # 3. point 가 polygon 의 경계선 위에 존재하는지 확인합니다.
    #    경계선 위에 존재하는 경우 False 를 반환합니다.
    n_vertex = 4 # 우리는 사각형만 다룹니다.
    for i in range(n_vertex):
        p1, p2 = polygon[i], polygon[(i + 1) % n_vertex]
        segment = (p1, p2)
        if is_point_on_segment(point, segment):
            return False
    # 4. point 가 polygon 의 경계선 위에 존재하지 않는 경우,
    #    point 를 기준으로 polygon 의 경계선과 교차하는 선분의 개수를 세어
    #    홀수인지 짝수인지 확인합니다.
    #    홀수인 경우 True 를 반환합니다.
    count = 0
    for i in range(n_vertex):
        p1, p2 = polygon[i], polygon[(i + 1) % n_vertex]
        segment1 = (p1, p2)
        segment2 = (point, (max_x + 1, point[1]))
        if does_two_segments_intersect(segment1, segment2):
            count += 1
    return count % 2 == 1


@numba.jit(cache=True, nopython=True)
def get_points_in_polygon(
    polygon_a: list, 
    polygon_b: list,
) -> list:
    """두 사각형 A, B의 꼭지점들 중
    A의 내부에 위치하는 B의 꼭짓점을 구합니다.
    접해 있는 꼭짓점은 포함하지 않습니다.

    Args:
        polygon_a (list): 꼭짓점을 포함하는 사각형
        polygon_b (list): 꼭짓점을 포함하는 사각형

    Returns:
        list: 사각형 A에 포함되어 있는 B의 꼭짓점들
    """
    ret = []
    # 다각형 B의 꼭짓점들을 순회합니다.
    # n_vertex = len(polygon_b)
    n_vertex = 4 # 우리는 사각형만 다룹니다.
    for i in range(n_vertex):
        if is_point_in_polygon(polygon_a, polygon_b[i]):
            ret.append(polygon_b[i])
    return ret


@numba.jit(cache=True, nopython=True)
def get_intersection_points_between_segments(
    segment1: tuple,
    segment2: tuple,
) -> list:
    ret = []
    if not does_two_segments_intersect(segment1, segment2):
        return ret
    else:
        p1, p2 = segment1
        p3, p4 = segment2
        p1, p2, p3, p4 = hg(p1), hg(p2), hg(p3), hg(p4)
        O = (0, 0, 0)
        v_a = cross_3d(O, p1, p2)
        v_b = cross_3d(O, p3, p4)
        v = cross_3d(O, v_a, v_b)
        if (v[0] + v[1] + v[2] == 0):
            line = sorted([p1, p2, p3, p4]) # 'key' is not supported (numba)
            # 한 점에서 만나는지를 검사합니다.
            dx = line[-1][0] - line[0][0]
            dy = line[-1][1] - line[0][1]
            dx_segment1 = abs(p1[0] - p2[0]) + abs(p3[0] - p4[0])
            dy_segment2 = abs(p1[1] - p2[1]) + abs(p3[1] - p4[1])
            if dx == dx_segment1 and dy == dy_segment2:
                ret.append((line[1][0], line[1][1]))
                return ret
            else:
                ret.append((line[1][0], line[1][1]))
                ret.append((line[2][0], line[2][1]))
                return ret
        else:
            v = (v[0] / v[2], v[1] / v[2])
            ret.append(v)
            return ret


@numba.jit(cache=True, nopython=True)
def get_intersection_points_between_polygons(
    polygon1: list, 
    polygon2: list,
) -> set:
    ret = set()
    for i in range(len(polygon1)):
        p1, p2 = polygon1[i], polygon1[(i + 1) % len(polygon1)]
        for j in range(len(polygon2)):
            p3, p4 = polygon2[j], polygon2[(j + 1) % len(polygon2)]
            points_intersection = get_intersection_points_between_segments((p1, p2), (p3, p4))
            for point in points_intersection:
                ret.add(point)
    return ret


@numba.jit(cache=True, nopython=True)
def sort_points_ccw_square(
    points: list
) -> list:
    if len(points) != 4:
        raise NotImplementedError
    ret = []
    ret.append(points[0])
    cross_01_02 = cross_2d(points[0], points[1], points[2])
    cross_01_03 = cross_2d(points[0], points[1], points[3])
    cross_02_03 = cross_2d(points[0], points[2], points[3])
    if cross_01_02 > 0:
        # 반시계 방향은 01 -> 02 입니다.
        ret.append(points[1])
        ret.append(points[2])
        if cross_01_03 > 0:
            # 반시계 방향은 01 -> 03 입니다.
            if cross_02_03 > 0:
                # 반시계 방향은 01 -> 02 -> 03 입니다.
                ret.append(points[3])
            else:
                # 반시계 방향은 01 -> 03 -> 02 입니다.
                ret.insert(2, points[3])
        else:
            # 반시계 방향은 03 -> 01 -> 02 입니다.
            ret.insert(1, points[3])
    else:
        # 반시계 방향은 02 -> 01 입니다.
        ret.append(points[2])
        ret.append(points[1])
        if cross_01_03 > 0:
            # 반시계 방향은 02 -> 01 -> 03 입니다.
            ret.append(points[3])
        else:
            # 반시계 방향은 03 -> 01 입니다.
            if cross_02_03 > 0:
                # 반시계 방향은 02 -> 03 -> 01 입니다.
                ret.insert(2, points[3])
            else:
                # 반시계 방향은 03 -> 02 -> 01 입니다.
                ret.insert(1, points[3])
    return ret


@numba.jit(cache=True, nopython=True)
def sort_points_ccw(
    points: list
) -> list:
    ret = []
    ret.append(points[0])
    ret.append(points[1])
    # 삽입 정렬
    for i in range(1, len(points)):
        for j in range(0, len(ret)):
            if cross_2d(ret[j], points[i], ret[(j + 1) % len(ret)]) > 0:
                ret.insert(j + 1, points[i])
                break
    return ret


@numba.jit(cache=True, nopython=True)
def get_intersection_polygon(
    polygon1: list, 
    polygon2: list,
) -> list:
    points_intersection = get_intersection_points_between_polygons(polygon1, polygon2)
    points_polygon2_in_polygon1 = get_points_in_polygon(polygon1, polygon2)
    points_polygon1_in_polygon2 = get_points_in_polygon(polygon2, polygon1)
    polygon_intersection = (
        [(float(x[0]), float(x[1])) for x in list(points_intersection)]
        + [(float(x[0]), float(x[1])) for x in points_polygon2_in_polygon1]
        + [(float(x[0]), float(x[1])) for x in points_polygon1_in_polygon2]
    )
    if len(polygon_intersection) < 3:
        polygon_intersection.clear()
        return polygon_intersection
    polygon_intersection = sort_points_ccw(polygon_intersection)
    return polygon_intersection


@numba.jit(cache=True, nopython=True)
def get_area(
    polygon: list
) -> float:
    ret = 0
    for i in range(len(polygon)):
        p1, p2 = polygon[i], polygon[(i + 1) % len(polygon)]
        ret += p1[0] * p2[1] - p1[1] * p2[0]
    return abs(ret) / 2


@numba.jit(cache=True, nopython=True)
def iou(
    polygon1: list,
    polygon2: list,
) -> float:
    a = get_area(polygon1)
    b = get_area(polygon2)
    polygon_intersection = get_intersection_polygon(polygon1, polygon2)
    a_and_b = get_area(polygon_intersection)
    if a + b - a_and_b == 0:
        return 0.
    return a_and_b / (a + b - a_and_b)