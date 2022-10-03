# 내부
import random
import unittest
import time

# 서드파티
import numpy as np

# 프로젝트
from trial.common import point_generator, polygon_generator
from polygon.python_polygon import cross_2d as cross_2d_original
from polygon.python_polygon_optimized import cross_2d as cross_2d_optimized
from polygon.python_polygon import is_point_on_segment as is_point_on_segment_original
from polygon.python_polygon_optimized import is_point_on_segment as is_point_on_segment_optimized
from polygon.python_polygon import does_two_segments_intersect as does_two_segments_intersect_original
from polygon.python_polygon_optimized import does_two_segments_intersect as does_two_segments_intersect_optimized
from polygon.python_polygon import is_point_in_polygon as is_point_in_polygon_original
from polygon.python_polygon_optimized import is_point_in_polygon as is_point_in_polygon_optimized
from polygon.python_polygon import get_points_in_polygon as get_points_in_polygon_original
from polygon.python_polygon_optimized import get_points_in_polygon as get_points_in_polygon_optimized
from polygon.python_polygon import get_intersection_points_between_segments as get_intersection_points_between_segments_original
from polygon.python_polygon_optimized import get_intersection_points_between_segments as get_intersection_points_between_segments_optimized
from polygon.python_polygon import get_intersection_points_between_polygons as get_intersection_points_between_polygons_original
from polygon.python_polygon_optimized import get_intersection_points_between_polygons as get_intersection_points_between_polygons_optimized
from polygon.python_polygon import sort_points_ccw_square as sort_points_ccw_square_original
from polygon.python_polygon_optimized import sort_points_ccw_square as sort_points_ccw_square_optimized
from polygon.python_polygon import sort_points_ccw as sort_points_ccw_original
from polygon.python_polygon_optimized import sort_points_ccw as sort_points_ccw_optimized
from polygon.python_polygon import get_intersection_polygon as get_intersection_polygon_original
from polygon.python_polygon_optimized import get_intersection_polygon as get_intersection_polygon_optimized
from polygon.python_polygon import get_area as get_area_original
from polygon.python_polygon_optimized import get_area as get_area_optimized
from polygon.python_polygon import iou as iou_original
from polygon.python_polygon_optimized import iou as iou_optimized


class TestPurePolygonOptimized(unittest.TestCase):
    def test_cross(self):
        p1 = np.array([1, 1])
        p2 = np.array([2, 2])
        p3 = np.array([3, 3])
        self.assertEqual(cross_2d_optimized(p1, p2, p3), np.cross(p2 - p1, p3 - p1))

        p1 = np.array([1, 1])
        p2 = np.array([2, 2])
        p3 = np.array([3, 4])
        self.assertEqual(cross_2d_optimized(p1, p2, p3), np.cross(p2 - p1, p3 - p1))

        p1 = np.array([1, 1])
        p2 = np.array([2, 2])
        p3 = np.array([4, 3])
        self.assertEqual(cross_2d_optimized(p1, p2, p3), np.cross(p2 - p1, p3 - p1))

        p1 = np.array([1, 1])
        p2 = np.array([2, 2])
        p3 = np.array([4, 4])
        self.assertEqual(cross_2d_optimized(p1, p2, p3), np.cross(p2 - p1, p3 - p1))

        p1 = np.array([1, 1])
        p2 = np.array([2, 2])
        p3 = np.array([3, 2])
        self.assertEqual(cross_2d_optimized(p1, p2, p3), np.cross(p2 - p1, p3 - p1))

        p1 = np.array([1, 1])
        p2 = np.array([2, 2])
        p3 = np.array([2, 3])
        self.assertEqual(cross_2d_optimized(p1, p2, p3), np.cross(p2 - p1, p3 - p1))
                
    def test_is_point_on_segment(self):
        p_0_0 = (0, 0)
        p_0_1 = (0, 1)
        p_1_0 = (1, 0)
        p_1_1 = (1, 1)
        p_2_2 = (2, 2)
        self.assertEqual(is_point_on_segment_optimized(p_0_1, (p_2_2, p_1_1)), False) # 외부
        self.assertEqual(is_point_on_segment_optimized(p_1_0, (p_1_1, p_2_2)), False) # 외부
        self.assertEqual(is_point_on_segment_optimized(p_0_0, (p_0_0, p_0_1)), True) # 끝점
        self.assertEqual(is_point_on_segment_optimized(p_0_1, (p_0_0, p_0_1)), True) # 끝점
        self.assertEqual(is_point_on_segment_optimized(p_1_1, (p_0_0, p_2_2)), True) # 포함
        self.assertEqual(is_point_on_segment_optimized(p_1_1, (p_2_2, p_0_0)), True) # 포함

    def test_does_two_segments_intersect(self):
        p_0_0 = (0, 0)
        p_0_1 = (0, 1)
        p_0_2 = (0, 2)
        p_1_0 = (1, 0)
        p_1_1 = (1, 1)
        p_2_0 = (2, 0)
        p_2_2 = (2, 2)
        p_3_3 = (3, 3)
        self.assertEqual(does_two_segments_intersect_optimized((p_0_0, p_0_1), (p_1_0, p_1_1)), False) # 평행
        self.assertEqual(does_two_segments_intersect_optimized((p_1_0, p_1_1), (p_0_0, p_0_1)), False) # 평행
        self.assertEqual(does_two_segments_intersect_optimized((p_0_0, p_1_1), (p_0_0, p_1_1)), True) # 포개짐, 완벽히 동일한 선분
        self.assertEqual(does_two_segments_intersect_optimized((p_0_0, p_1_1), (p_1_1, p_0_0)), True) # 포개짐, 완벽히 동일한 선분
        self.assertEqual(does_two_segments_intersect_optimized((p_0_0, p_2_2), (p_1_1, p_0_0)), True) # 포개짐
        self.assertEqual(does_two_segments_intersect_optimized((p_2_2, p_0_0), (p_0_0, p_1_1)), True) # 포개짐
        self.assertEqual(does_two_segments_intersect_optimized((p_2_2, p_3_3), (p_1_1, p_2_2)), True) # 포개짐
        self.assertEqual(does_two_segments_intersect_optimized((p_0_2, p_2_0), (p_1_1, p_2_2)), True) # 한 선분의 끝점이 다른 선분과 겹침
        self.assertEqual(does_two_segments_intersect_optimized((p_0_0, p_1_1), (p_1_1, p_2_2)), True) # 한 선분의 끝점이 다른 선분의 끝점과 겹침
        self.assertEqual(does_two_segments_intersect_optimized((p_0_0, p_0_1), (p_0_2, p_1_1)), False) # 서로 닿지는 않지만, 한 선분의 연장선이 다른 선분의 끝점을 지나감
        self.assertEqual(does_two_segments_intersect_optimized((p_0_0, p_0_1), (p_1_1, p_2_2)), False) # 서로 닿지도 않고 한 선분의 연장선이 다른 선분의 끝점을 지나가지도 않는 경우
        self.assertEqual(does_two_segments_intersect_optimized((p_0_0, p_3_3), (p_2_0, p_0_2)), True) # 누가 봐도 교차하는 경우

    def test_is_point_in_polygon(self):
        polygon = [(0, 0), (1, 0), (2, 2), (0, 1)]
        # 꼭짓점
        point = (2, 2)
        self.assertEqual(is_point_in_polygon_optimized(polygon, point), False)
        # 꼭짓점
        point = (1, 0)
        self.assertEqual(is_point_in_polygon_optimized(polygon, point), False)
        # 모서리 중점
        point = (0.5, 0)
        self.assertEqual(is_point_in_polygon_optimized(polygon, point), False)
        # 내부
        point = (0.5, 0.5)
        self.assertEqual(is_point_in_polygon_optimized(polygon, point), True)
        # 외부, polygon bbox 내부
        point = (0.5, 1.5)
        self.assertEqual(is_point_in_polygon_optimized(polygon, point), False)
        point = (1, 1.5)
        self.assertEqual(is_point_in_polygon_optimized(polygon, point), False)
        # 외부, polygon bbox 외부
        point = (2.5, 2.5)
        self.assertEqual(is_point_in_polygon_optimized(polygon, point), False)

    def test_get_points_in_polygon(self):
        # case 1
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 3), (4, 3), (4, 5), (1, 5)]
        self.assertEqual(
            set(get_points_in_polygon_optimized(polygon1, polygon2)),
            set([]))
        # case 2
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 3), (5, 3), (5, 5), (1, 5)]
        self.assertEqual(
            set(get_points_in_polygon_optimized(polygon1, polygon2)),
            set([]))
        # case 3
        polygon1 = [(0, 0), (4, 0), (4, 3), (0, 3)]
        polygon2 = [(4, 0), (5, 0), (5, 3), (4, 3)]
        self.assertEqual(
            set(get_points_in_polygon_optimized(polygon1, polygon2)),
            set([]))
        # case 4
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 0), (4, 0), (4, 2), (1, 2)]
        self.assertEqual(
            set(get_points_in_polygon_optimized(polygon1, polygon2)),
            set([(1, 2), (4, 2)])
        )
        # case 5
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(0, 0), (3, 0), (3, 2), (0, 2)]
        self.assertEqual(
            set(get_points_in_polygon_optimized(polygon1, polygon2)),
            set([(3, 2)])
        )
        # case 6
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(4, 0), (5, 0), (5, 3), (4, 3)]
        self.assertEqual(
            set(get_points_in_polygon_optimized(polygon1, polygon2)),
            set([])
        )
        # case 7
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        self.assertEqual(
            set(get_points_in_polygon_optimized(polygon1, polygon2)),
            set([])
        )        
        # case 8
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 0), (4, 0), (4, 5), (1, 5)]
        self.assertEqual(
            set(get_points_in_polygon_optimized(polygon1, polygon2)),
            set([])
        )
        # case 9
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(0, 0), (3, 0), (3, 4), (0, 4)]
        self.assertEqual(
            set(get_points_in_polygon_optimized(polygon1, polygon2)),
            set([])
        )
        # case 10
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 1), (3, 1), (3, 2), (1, 2)]
        self.assertEqual(
            set(get_points_in_polygon_optimized(polygon1, polygon2)),
            set([(1, 1), (3, 1), (3, 2), (1, 2)])
        )
        # case 11
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 1), (3, 1), (3, 5), (1, 5)]
        self.assertEqual(
            set(get_points_in_polygon_optimized(polygon1, polygon2)),
            set([(1, 1), (3, 1)])
        )
        # case 12
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(5, 3), (6, 4), (5, 5), (4, 4)]
        self.assertEqual(
            set(get_points_in_polygon_optimized(polygon1, polygon2)),
            set([])
        )
        # case 13
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(2, 0), (4, 1), (2, 2), (0, 1)]
        self.assertEqual(
            set(get_points_in_polygon_optimized(polygon1, polygon2)),
            set([(4, 1), (2, 2)])
        )
        # case 14
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(2, 2), (3, 4), (2, 6), (1, 4)]
        self.assertEqual(
            set(get_points_in_polygon_optimized(polygon1, polygon2)),
            set([(2, 2)])
        )

    def test_get_intersection_points_between_segments(self):
        segment1 = [(0, 0), (3, 3)]
        segment2 = [(3, 1), (5, 3)]
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment1, segment2)), set([]))
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment2, segment1)), set([]))
        segment2 = [(4, 4), (6, 6)]
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment1, segment2)), set([]))
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment2, segment1)), set([]))
        segment2 = [(3, 3), (5, 5)]
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment1, segment2)), set([(3, 3)]))
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment2, segment1)), set([(3, 3)]))
        segment2 = [(2, 2), (5, 5)]
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment1, segment2)), set([(2, 2), (3, 3)]))
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment2, segment1)), set([(2, 2), (3, 3)]))
        segment2 = [(1, 1), (2, 2)]
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment1, segment2)), set([(1, 1), (2, 2)]))
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment2, segment1)), set([(1, 1), (2, 2)]))
        segment2 = [(1, 1), (3, 3)]
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment1, segment2)), set([(1, 1), (3, 3)]))
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment2, segment1)), set([(1, 1), (3, 3)]))
        segment2 = [(0, 0), (3, 3)]
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment1, segment2)), set([(0, 0), (3, 3)]))
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment2, segment1)), set([(0, 0), (3, 3)]))
        segment2 = [(4, 2), (4, 5)]
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment1, segment2)), set([]))
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment2, segment1)), set([]))
        segment2 = [(4, 1), (4, 4)]
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment1, segment2)), set([]))
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment2, segment1)), set([]))
        segment2 = [(3, 0), (3, 3)]
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment1, segment2)), set([(3, 3)]))
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment2, segment1)), set([(3, 3)]))
        segment2 = [(2, 0), (2, 4)]
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment1, segment2)), set([(2, 2)]))
        self.assertEqual(set(get_intersection_points_between_segments_optimized(segment2, segment1)), set([(2, 2)]))

    def test_get_intersection_points_between_polygons(self):
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 3), (4, 3), (4, 5), (1, 5)]
        self.assertEqual(
            set(get_intersection_points_between_polygons_optimized(polygon1, polygon2)), 
            set([(1, 3), (4, 3)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 3), (5, 3), (5, 5), (1, 5)]
        self.assertEqual(
            set(get_intersection_points_between_polygons_optimized(polygon1, polygon2)), 
            set([(1, 3), (5, 3)]))
        polygon1 = [(0, 0), (4, 0), (4, 3), (0, 3)]
        polygon2 = [(4, 0), (5, 0), (5, 3), (4, 3)]
        self.assertEqual(
            set(get_intersection_points_between_polygons_optimized(polygon1, polygon2)), 
            set([(4, 0), (4, 3)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 0), (4, 0), (4, 2), (1, 2)]
        self.assertEqual(
            set(get_intersection_points_between_polygons_optimized(polygon1, polygon2)), 
            set([(1, 0), (4, 0)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(0, 0), (3, 0), (3, 2), (0, 2)]
        self.assertEqual(
            set(get_intersection_points_between_polygons_optimized(polygon1, polygon2)), 
            set([(0, 0), (3, 0), (0, 2)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(4, 0), (5, 0), (5, 3), (4, 3)]
        self.assertEqual(
            set(get_intersection_points_between_polygons_optimized(polygon1, polygon2)), 
            set([(4, 0), (4, 3), (5, 0), (5, 3)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        self.assertEqual(
            set(get_intersection_points_between_polygons_optimized(polygon1, polygon2)), 
            set([(0, 0), (5, 0), (5, 3), (0, 3)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 0), (4, 0), (4, 5), (1, 5)]
        self.assertEqual(
            set(get_intersection_points_between_polygons_optimized(polygon1, polygon2)), 
            set([(1, 0), (4, 0), (1, 3), (4, 3)]))                
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(0, 0), (3, 0), (3, 4), (0, 4)]
        self.assertEqual(
            set(get_intersection_points_between_polygons_optimized(polygon1, polygon2)), 
            set([(0, 0), (3, 0), (0, 3), (3, 3)]))                
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 1), (3, 1), (3, 2), (1, 2)]
        self.assertEqual(
            set(get_intersection_points_between_polygons_optimized(polygon1, polygon2)), 
            set([]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 1), (3, 1), (3, 5), (1, 5)]
        self.assertEqual(
            set(get_intersection_points_between_polygons_optimized(polygon1, polygon2)), 
            set([(1, 3), (3, 3)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(5, 3), (6, 4), (5, 5), (4, 4)]
        self.assertEqual(
            set(get_intersection_points_between_polygons_optimized(polygon1, polygon2)), 
            set([(5, 3)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(2, 0), (4, 1), (2, 2), (0, 1)]
        self.assertEqual(
            set(get_intersection_points_between_polygons_optimized(polygon1, polygon2)), 
            set([(2, 0), (0, 1)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(2, 2), (3, 4), (2, 6), (1, 4)]
        self.assertEqual(
            set(get_intersection_points_between_polygons_optimized(polygon1, polygon2)), 
            set([(1.5, 3), (2.5, 3)]))

    def test_sort_points_ccw_square(self):
        # 직사각형
        # 직사각형의 좌하단을 기준삼아 정렬
        li = [(0, 0), (0, 3), (5, 3), (5, 0)]
        polygon = [(0, 0), (5, 0), (5, 3), (0, 3)]
        self.assertEqual(sort_points_ccw_square_optimized(li), polygon)
        li = [(1, 3), (4, 5), (4, 3), (1, 5)]
        polygon = [(1, 3), (4, 3), (4, 5), (1, 5)]
        self.assertEqual(sort_points_ccw_square_optimized(li), polygon)
        li = [(1, 3), (5, 5), (1, 5), (5, 3)]
        polygon = [(1, 3), (5, 3), (5, 5), (1, 5)]
        self.assertEqual(sort_points_ccw_square_optimized(li), polygon)
        # 직사각형
        # 직사각형의 우하단을 기준삼아 정렬
        li = [(4, 0), (0, 3), (0, 0), (4, 3)]
        polygon = [(4, 0), (4, 3), (0, 3), (0, 0)]
        self.assertEqual(sort_points_ccw_square_optimized(li), polygon)
        li = [(4, 0), (4, 2), (1, 0), (1, 2)]
        polygon = [(4, 0), (4, 2), (1, 2), (1, 0)]
        self.assertEqual(sort_points_ccw_square_optimized(li), polygon)
        # 직사각형
        # 직사각형의 우상단을 기준삼아 정렬
        li = [(3, 2), (0, 0), (3, 0), (0, 2)]
        polygon = [(3, 2), (0, 2), (0, 0), (3, 0)]
        self.assertEqual(sort_points_ccw_square_optimized(li), polygon)
        li = [(5, 3), (4, 3), (5, 0), (4, 0)]
        polygon = [(5, 3), (4, 3), (4, 0), (5, 0)]
        self.assertEqual(sort_points_ccw_square_optimized(li), polygon)
        # 직사각형
        # 직사각형의 좌상단을 기준삼아 정렬
        li = [(1, 6), (4, 0), (4, 6), (1, 0)]
        polygon = [(1, 6), (1, 0), (4, 0), (4, 6)]
        self.assertEqual(sort_points_ccw_square_optimized(li), polygon)
        li = [(0, 4), (0, 0), (3, 4), (3, 0)]
        polygon = [(0, 4), (0, 0), (3, 0), (3, 4)]
        self.assertEqual(sort_points_ccw_square_optimized(li), polygon)
        li = [(1, 2), (3, 1), (3, 2), (1, 1)]
        polygon = [(1, 2), (1, 1), (3, 1), (3, 2)]
        self.assertEqual(sort_points_ccw_square_optimized(li), polygon)
        li = [(1, 6), (3, 1), (1, 1), (3, 6)]
        polygon = [(1, 6), (1, 1), (3, 1), (3, 6)]
        self.assertEqual(sort_points_ccw_square_optimized(li), polygon)

    def test_sort_points_ccw(self):
        polygons = [
            # 직사각형의 좌하단을 기준삼아 정렬
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(1, 3), (4, 3), (4, 5), (1, 5)],
            [(1, 3), (5, 3), (5, 5), (1, 5)],
            # 직사각형의 우하단을 기준삼아 정렬
            [(4, 0), (4, 3), (0, 3), (0, 0)],
            [(4, 0), (4, 2), (1, 2), (1, 0)],
            # 직사각형의 우상단을 기준삼아 정렬
            [(3, 2), (0, 2), (0, 0), (3, 0)],
            [(5, 3), (4, 3), (4, 0), (5, 0)],
            # 직사각형의 좌상단을 기준삼아 정렬
            [(1, 6), (1, 0), (4, 0), (4, 6)],
            [(0, 4), (0, 0), (3, 0), (3, 4)],
            [(1, 2), (1, 1), (3, 1), (3, 2)],
            [(1, 6), (1, 1), (3, 1), (3, 6)],
            # 삼각형
            [(3, 2), (4, 3), (2, 3)],
            # 오각형
            [(3, 0), (5, 2), (4, 3), (2, 3), (1, 2)],
            # 육각형
            [(3., 0.), (5., 1+1/3), (5., 2+2/3), (4.5, 3.), (1.5, 3.), (0., 2.)],
            # 칠각형
            [(2, 1), (4, 1), (5, 2), (5, 4), (4, 5), (2, 5), (0, 3)],
            # 팔각형
            [(2, 1), (4, 1), (5, 2), (5, 4), (4, 5), (2, 5), (1, 4), (1, 2)],            
        ]
        for polygon in polygons:
            todo = polygon.copy()[1:]
            random.shuffle(todo)
            ret = sort_points_ccw_optimized([polygon[0]] + todo)
            self.assertEqual(ret, polygon)

    def test_get_intersection_polygon(self):
        polygons1 = [
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (4, 0), (4, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 1), (5, 1), (5, 5), (0, 5)],
            [(1, 1), (5, 1), (5, 5), (1, 5)],
        ]
        polygons2 = [
            [(1, 3), (4, 3), (4, 5), (1, 5)],
            [(1, 3), (5, 3), (5, 5), (1, 5)],
            [(4, 0), (5, 0), (5, 3), (4, 3)],
            [(1, 0), (4, 0), (4, 2), (1, 2)],
            [(0, 0), (3, 0), (3, 2), (0, 2)],
            [(4, 0), (5, 0), (5, 3), (4, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(1, 0), (4, 0), (4, 5), (1, 5)],
            [(0, 0), (3, 0), (3, 4), (0, 4)],
            [(1, 1), (3, 1), (3, 2), (1, 2)],
            [(1, 1), (3, 1), (3, 5), (1, 5)],
            [(5, 3), (6, 4), (5, 5), (4, 4)],
            [(2, 0), (4, 1), (2, 2), (0, 1)],
            [(2, 2), (3, 4), (2, 6), (1, 4)],
            [(2, 3), (3, 4), (2, 5), (1, 4)],
            [(5, 3), (6, 4), (5, 5), (4, 4)],
            [(3, 2), (5, 4), (3, 6), (1, 4)],
            [(3, 1), (5, 3), (3, 5), (1, 3)],
            [(3, 0), (5, 2), (3, 4), (1, 2)],
            [(4, 0), (6, 2), (4, 4), (2, 2)],
            [(3, 0), (6, 2), (3, 4), (0, 2)],
            [(3, 0), (6, 3), (3, 6), (0, 3)],
            [(3, 0), (6, 3), (3, 6), (0, 3)],
        ]
        polygons_intersection = [
            [],
            [],
            [],
            [(1, 0), (4, 0), (4, 2), (1, 2)],
            [(0, 0), (3, 0), (3, 2), (0, 2)],
            [(4, 0), (5, 0), (5, 3), (4, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(1, 0), (4, 0), (4, 3), (1, 3)],
            [(0, 0), (3, 0), (3, 3), (0, 3)],
            [(1, 1), (3, 1), (3, 2), (1, 2)],
            [(1, 1), (3, 1), (3, 3), (1, 3)],
            [],
            [(2, 0), (4, 1), (2, 2), (0, 1)],
            [(2, 2), (2.5, 3), (1.5, 3)],
            [],
            [],
            [(3, 2), (4, 3), (2, 3)], # 삼각형
            [(3, 1), (5, 3), (1, 3)], # 삼각형
            [(3, 0), (5, 2), (4, 3), (2, 3), (1, 2)], # 오각형
            [(4, 0), (5, 1), (5, 3), (3, 3), (2, 2)], # 오각형
            [(3, 0), (5, 1+1/3), (5, 2+2/3), (4.5, 3), (1.5, 3), (0, 2)], # 육각형
            [(2, 1), (4, 1), (5, 2), (5, 4), (4, 5), (2, 5), (0, 3)], # 칠각형
            [(2, 1), (4, 1), (5, 2), (5, 4), (4, 5), (2, 5), (1, 4), (1, 2)], # 팔각형
        ]
        for polygon1, polygon2, polygon_intersection in zip(
            polygons1, polygons2, polygons_intersection):
            with self.subTest(
                polygon1=polygon1, 
                polygon2=polygon2,
                gt=polygon_intersection,
            ):
                # 개수가 동일한지 검사
                self.assertEqual(
                    len(get_intersection_polygon_optimized(polygon1, polygon2)),
                    len(polygon_intersection)
                )
                # 동일한 요소들이 존재하는지 검사
                self.assertEqual(
                    set(get_intersection_polygon_optimized(polygon1, polygon2)),
                    set(polygon_intersection)
                )
                # 순서가 맞는지 검사
                ret = get_intersection_polygon_optimized(polygon1, polygon2)
                if ret:
                    is_sorted = False
                    for i in range(len(ret)):
                        rolled = np.roll(np.array(ret), i, axis=0)
                        polygon_intersection = [list(e) for e in polygon_intersection]
                        polygon_intersection = np.array(polygon_intersection)
                        if np.all(rolled == polygon_intersection):
                            is_sorted = True
                            break
                    self.assertTrue(is_sorted)
    
    def test_get_area(self):
        polygons_and_areas = [
            ([(1, 3), (4, 3), (4, 5), (1, 5)], 6),
            ([(1, 3), (5, 3), (5, 5), (1, 5)], 8),
            ([(4, 0), (5, 0), (5, 3), (4, 3)], 3),
            ([(1, 0), (4, 0), (4, 2), (1, 2)], 6),
            ([(0, 0), (3, 0), (3, 2), (0, 2)], 6),
            ([(4, 0), (5, 0), (5, 3), (4, 3)], 3 ),
            ([(0, 0), (5, 0), (5, 3), (0, 3)], 15),
            ([(1, 0), (4, 0), (4, 5), (1, 5)], 15),
            ([(0, 0), (3, 0), (3, 4), (0, 4)], 12),
            ([(1, 1), (3, 1), (3, 2), (1, 2)], 2),
            ([(1, 1), (3, 1), (3, 5), (1, 5)], 8),
            ([(5, 3), (6, 4), (5, 5), (4, 4)], 2),
            ([(2, 0), (4, 1), (2, 2), (0, 1)], 4),
            ([(2, 2), (3, 4), (2, 6), (1, 4)], 4),
            ([(2, 3), (3, 4), (2, 5), (1, 4)], 2),
            ([(5, 3), (6, 4), (5, 5), (4, 4)], 2),
            ([(3, 2), (5, 4), (3, 6), (1, 4)], 8),
            ([(3, 1), (5, 3), (3, 5), (1, 3)], 8),
            ([(3, 0), (5, 2), (3, 4), (1, 2)], 8),
            ([(4, 0), (6, 2), (4, 4), (2, 2)], 8),
            ([(3, 0), (6, 2), (3, 4), (0, 2)], 12),
            ([(3, 0), (6, 3), (3, 6), (0, 3)], 18),
        ]
        for polygon, area in polygons_and_areas:
            with self.subTest(polygon=polygon):
                self.assertEqual(get_area_optimized(polygon), area)
            with self.subTest(polygon=polygon[::-1]):
                self.assertEqual(get_area_optimized(polygon[::-1]), area)

    def test_iou(self):
        polygons1 = [
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (4, 0), (4, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(0, 1), (5, 1), (5, 5), (0, 5)],
            [(1, 1), (5, 1), (5, 5), (1, 5)],
        ]
        polygons2 = [
            [(1, 3), (4, 3), (4, 5), (1, 5)],
            [(1, 3), (5, 3), (5, 5), (1, 5)],
            [(4, 0), (5, 0), (5, 3), (4, 3)],
            [(1, 0), (4, 0), (4, 2), (1, 2)],
            [(0, 0), (3, 0), (3, 2), (0, 2)],
            [(4, 0), (5, 0), (5, 3), (4, 3)],
            [(0, 0), (5, 0), (5, 3), (0, 3)],
            [(1, 0), (4, 0), (4, 5), (1, 5)],
            [(0, 0), (3, 0), (3, 4), (0, 4)],
            [(1, 1), (3, 1), (3, 2), (1, 2)],
            [(1, 1), (3, 1), (3, 5), (1, 5)],
            [(5, 3), (6, 4), (5, 5), (4, 4)],
            [(2, 0), (4, 1), (2, 2), (0, 1)],
            [(2, 2), (3, 4), (2, 6), (1, 4)],
            [(2, 3), (3, 4), (2, 5), (1, 4)],
            [(5, 3), (6, 4), (5, 5), (4, 4)],
            [(3, 2), (5, 4), (3, 6), (1, 4)],
            [(3, 1), (5, 3), (3, 5), (1, 3)],
            [(3, 0), (5, 2), (3, 4), (1, 2)],
            [(4, 0), (6, 2), (4, 4), (2, 2)],
            [(3, 0), (6, 2), (3, 4), (0, 2)],
            [(3, 0), (6, 3), (3, 6), (0, 3)],
            [(3, 0), (6, 3), (3, 6), (0, 3)],
        ]
        polygons_intersection_area = [
            0/(15+6-0), 0/(15+8-0), 0/(12+3-0), 
            6/(15+6-6), 6/(15+6-6), 3/(15+3-3), 15/(15+15-15), 
            9/(15+15-9), 9/(15+12-9), 2/(15+2-2), 4/(15+8-4), 
            0/(15+2-0), 4/(15+4-4), 0.5/(15+4-0.5), 
            0/(15+2-0), 0/(15+2-0), 1/(15+8-1), 4/(15+8-4), 
            7/(15+8-7), 6/(15+8-6), (10-1/6)/(15+12-(10-1/6)), 15/(20+18-15), 14/(16+18-14),
        ]
        assert len(polygons1) == len(polygons2) == len(polygons_intersection_area)
        for polygon1, polygon2, intersection_area in zip(
            polygons1, polygons2, polygons_intersection_area):
            with self.subTest(polygon1=polygon1, polygon2=polygon2):
                self.assertAlmostEqual(
                    iou_optimized(polygon1, polygon2),
                    intersection_area)
                

class TestPurePolygonOptimizedPerformance(unittest.TestCase):
    def test_cross(self):
        n = 10000
        print(f'\ncross() speed test (iter: {n})')
        with self.subTest('pure python speed'):
            gen1 = point_generator()
            gen2 = point_generator()
            gen3 = point_generator()
            total_time_ori = 0
            for idx in range(n):
                p1 = next(gen1)
                p2 = next(gen2)
                p3 = next(gen3)
                s = time.time()
                cross_2d_original(p1, p2, p3)
                e = time.time()
                if idx != 0:
                    total_time_ori += e - s
            print(f'original\t: {total_time_ori:.5f} sec')
        with self.subTest('numba optimized python speed'):
            gen1 = point_generator('numpy')
            gen2 = point_generator('numpy')
            gen3 = point_generator('numpy')
            total_time_opt = 0
            for _ in range(n):
                p1 = next(gen1)
                p2 = next(gen2)
                p3 = next(gen3)
                s = time.time()
                cross_2d_optimized(p1, p2, p3)
                e = time.time()
                if idx != 0:
                    total_time_opt += e - s
            print(f'optimized\t: {total_time_opt:.5f} sec')
            e = time.time()
        self.assertGreater(total_time_ori, total_time_opt)

    def test_is_point_on_segment(self):
        n = 10000
        print(f'\nis_point_on_segment() speed test (iter: {n})')
        gen1 = point_generator()
        gen2 = point_generator()
        gen3 = point_generator()
        with self.subTest('pure python speed'):
            total_time_ori = 0
            for idx in range(n):
                p1 = next(gen1)
                p2 = next(gen2)
                p3 = next(gen3)
                s = time.time()
                is_point_on_segment_original(p1, (p2, p3))
                e = time.time()
                if idx != 0:
                    total_time_ori += e - s
            print(f'original\t: {total_time_ori:.5f} sec')
        with self.subTest('numba optimized python speed'):
            s = time.time()
            total_time_opt = 0
            for idx in range(n):
                p1 = next(gen1)
                p2 = next(gen2)
                p3 = next(gen3)
                s = time.time()
                is_point_on_segment_optimized(p1, (p2, p3))
                e = time.time()
                if idx != 0:
                    total_time_opt += e - s
            print(f'optimized\t: {total_time_opt:.5f} sec')
            e = time.time()
        self.assertGreater(total_time_ori, total_time_opt)

    def test_does_two_segments_intersect(self):
        n = 10000
        print(f'\ndoes_two_segments_intersect() speed test (iter: {n})')
        gen1 = point_generator()
        gen2 = point_generator()
        gen3 = point_generator()
        gen4 = point_generator()
        with self.subTest('pure python speed'):
            total_time_ori = 0
            for idx in range(n):
                p1 = next(gen1)
                p2 = next(gen2)
                p3 = next(gen3)
                p4 = next(gen4)
                s = time.time()
                does_two_segments_intersect_original((p1, p2), (p3, p4))
                e = time.time()
                if idx != 0:
                    total_time_ori += e - s
            print(f'original\t: {total_time_ori:.5f} sec')
        with self.subTest('numba optimized python speed'):
            s = time.time()
            total_time_opt = 0
            for idx in range(n):
                p1 = next(gen1)
                p2 = next(gen2)
                p3 = next(gen3)
                p4 = next(gen4)
                s = time.time()
                does_two_segments_intersect_optimized((p1, p2), (p3, p4))
                e = time.time()
                if idx != 0:
                    total_time_opt += e - s
            print(f'optimized\t: {total_time_opt:.5f} sec')
            e = time.time()
        self.assertGreater(total_time_ori, total_time_opt)

    def test_is_point_in_polygon(self):
        n = 10000
        print(f'\nis_point_in_polygon() speed test (iter: {n})')
        with self.subTest('pure python speed'):
            gen_polygon = polygon_generator()
            gen_point = point_generator()
            total_time_ori = 0
            for idx in range(n):
                polygon = next(gen_polygon)
                point = next(gen_point)
                s = time.time()
                _ = is_point_in_polygon_original(polygon, point)
                e = time.time()
                if idx != 0:
                    total_time_ori += e - s
            print(f'original\t: {total_time_ori:.5f} sec')
        with self.subTest('numba optimized python speed'):
            gen_polygon = polygon_generator('numpy')
            gen_point = point_generator('numpy')
            total_time_opt = 0
            for idx in range(n):
                polygon = next(gen_polygon)
                point = next(gen_point)
                s = time.time()
                _ = is_point_in_polygon_optimized(polygon, point)
                e = time.time()
                if idx != 0:
                    total_time_opt += e - s
            print(f'optimized\t: {total_time_opt:.5f} sec *')
            print('* 시간을 측정하기 위해 polygon 생성 시뮬레이션을 사용하는데, '
                  'is_point_in_polygon_optimized 함수 내부의 분기에 의한 프로그램 흐름이 '
                  '복잡한 연산을 수행하지 않는 경우로 실행되는 polygon 을 입력받을 수 있기 때문에 '
                  '테스트가 예상보다 훨씬 빠르게 작동할수도 있습니다.')
        self.assertGreater(total_time_ori, total_time_opt)

    def test_get_points_in_polygon(self):
        n = 10000
        print(f'\nget_points_in_polygon() speed test (iter: {n})')
        with self.subTest('pure python speed'):
            gen1 = polygon_generator()
            gen2 = polygon_generator()
            total_time_ori = 0
            for idx in range(n):
                polygon1 = next(gen1)
                polygon2 = next(gen2)
                s = time.time()
                get_points_in_polygon_original(polygon1, polygon2)
                e = time.time()
                if idx != 0:
                    total_time_ori += e - s
            print(f'original\t: {total_time_ori:.5f} sec')
        with self.subTest('umba optimized python speed'):
            gen1 = polygon_generator('numpy')
            gen2 = polygon_generator('numpy')
            total_time_opt = 0
            for idx in range(n):
                polygon1 = next(gen1)
                polygon2 = next(gen2)
                s = time.time()
                get_points_in_polygon_optimized(polygon1, polygon2)
                e = time.time()
                if idx != 0:
                    total_time_opt += e - s
            print(f'optimized\t: {total_time_opt:.5f} sec *')
            print('* 시간을 측정하기 위해 polygon 생성 시뮬레이션을 사용하는데, '
                  'get_points_in_polygon_optimized 함수 내부의 분기에 의한 프로그램 흐름이 '
                  '복잡한 연산을 수행하지 않는 경우로 실행되는 polygon 을 입력받을 수 있기 때문에 '
                  '테스트가 예상보다 훨씬 빠르게 작동할수도 있습니다.')
        self.assertGreater(total_time_ori, total_time_opt)

    def test_get_intersection_points_between_segments(self):
        n = 10000
        print(f'\nget_intersection_points_between_segments() speed test (iter: {n})')
        with self.subTest('pure python speed'):
            gen1 = point_generator()
            gen2 = point_generator()
            gen3 = point_generator()
            gen4 = point_generator()
            total_time_ori = 0
            for idx in range(n):
                segment_1 = (next(gen1), next(gen3))
                segment_2 = (next(gen2), next(gen4))
                s = time.time()
                get_intersection_points_between_segments_original(segment_1, segment_2)
                e = time.time()
                if idx != 0:
                    total_time_ori += e - s
            print(f'original\t: {total_time_ori:.5f} sec')
        with self.subTest('numba optimized python speed'):
            gen1 = point_generator('numpy')
            gen2 = point_generator('numpy')
            gen3 = point_generator('numpy')
            gen4 = point_generator('numpy')
            total_time_opt = 0
            for idx in range(n):
                segment_1 = (next(gen1), next(gen3))
                segment_2 = (next(gen2), next(gen4))
                s = time.time()
                get_intersection_points_between_segments_optimized(segment_1, segment_2)
                e = time.time()
                if idx != 0:
                    total_time_opt += e - s
            print(f'optimized\t: {total_time_opt:.5f} sec *')
            print('* 시간을 측정하기 위해 segment 생성 시뮬레이션을 사용하는데, '
                  'get_intersection_points_between_segments_optimized 함수 내부의 분기에 의한 프로그램 흐름이 '
                  '복잡한 연산을 수행하지 않는 경우로 실행되는 segment 을 입력받을 수 있기 때문에 '
                  '테스트가 예상보다 훨씬 빠르게 작동할수도 있습니다.')
        self.assertGreater(total_time_ori, total_time_opt)

    def test_get_intersection_points_between_polygons(self):
        n = 5000
        print(f'\nget_intersection_points_between_polygons() speed test (iter: {n})')
        with self.subTest('pure python speed'):
            gen1 = polygon_generator()
            gen2 = polygon_generator()
            total_time_ori = 0
            for idx in range(n):
                polygon1 = next(gen1)
                polygon2 = next(gen2)
                s = time.time()
                get_intersection_points_between_polygons_original(polygon1, polygon2)
                e = time.time()
                if idx != 0:
                    total_time_ori += e - s
            print(f'original\t: {total_time_ori:.5f} sec')
        with self.subTest('numba optimized python speed'):
            gen1 = polygon_generator('numpy')
            gen2 = polygon_generator('numpy')
            total_time_opt = 0
            for idx in range(n):
                polygon1 = next(gen1)
                polygon2 = next(gen2)
                s = time.time()
                get_intersection_points_between_polygons_optimized(polygon1, polygon2)
                e = time.time()
                if idx != 0:
                    total_time_opt += e - s
            print(f'optimized\t: {total_time_opt:.5f} sec *')
            print('* 시간을 측정하기 위해 segment 생성 시뮬레이션을 사용하는데, '
                  'get_intersection_points_between_segments_optimized 함수 내부의 분기에 의한 프로그램 흐름이 '
                  '복잡한 연산을 수행하지 않는 경우로 실행되는 segment 을 입력받을 수 있기 때문에 '
                  '테스트가 예상보다 훨씬 빠르게 작동할수도 있습니다.')
        self.assertGreater(total_time_ori, total_time_opt)

    def test_sort_points_ccw_square(self):
        n = 10000
        print(f'\ntest_sort_points_ccw_square_speed() speed test (iter: {n})')
        with self.subTest('pure python speed'):
            gen = polygon_generator()
            total_time_ori = 0
            for idx in range(n):
                polygon = next(gen)
                s = time.time()
                sort_points_ccw_square_original(polygon)
                e = time.time()
                if idx != 0:
                    total_time_ori += e - s
            print(f'original\t: {total_time_ori:.5f} sec')
        with self.subTest('numba optimized python speed'):
            gen = polygon_generator('numpy')
            total_time_opt = 0
            for idx in range(n):
                polygon = next(gen)
                s = time.time()
                sort_points_ccw_square_optimized(polygon)
                e = time.time()
                if idx != 0:
                    total_time_opt += e - s
            print(f'optimized\t: {total_time_opt:.5f} sec')

    def test_sort_points_ccw(self):
        n = 100000
        print(f'\ntest_sort_points_ccw_speed() speed test (iter: {n})')
        with self.subTest('pure python speed'):
            gen = polygon_generator()
            total_time_ori = 0
            for idx in range(n):
                polygon = next(gen)
                random.shuffle(polygon)
                s = time.time()
                sort_points_ccw_original(polygon)
                e = time.time()
                if idx != 0:
                    total_time_ori += e - s
            print(f'original\t: {total_time_ori:.5f} sec')
        with self.subTest('numba optimized python speed'):
            gen = polygon_generator('numpy')
            total_time_opt = 0
            for idx in range(n):
                polygon = next(gen)
                random.shuffle(polygon)
                s = time.time()
                sort_points_ccw_optimized(polygon)
                e = time.time()
                if idx != 0:
                    total_time_opt += e - s
            print(f'optimized\t: {total_time_opt:.5f} sec')

    def test_get_intersection_polygon(self):
        n = 10000
        print(f'\nget_intersection_polygon() speed test (iter: {n})')
        with self.subTest('pure python speed'):
            gen1 = polygon_generator()
            gen2 = polygon_generator()
            total_time_ori = 0
            for idx in range(n):
                polygon1 = next(gen1)
                polygon2 = next(gen2)
                s = time.time()
                _ = get_intersection_polygon_original(polygon1, polygon2)
                e = time.time()
                if idx != 0:
                    total_time_ori += e - s
            print(f'original\t: {total_time_ori:.5f} sec')
        with self.subTest('numba optimized python speed'):
            gen1 = polygon_generator('numpy')
            gen2 = polygon_generator('numpy')
            total_time_opt = 0
            for idx in range(n):
                polygon1 = next(gen1)
                polygon2 = next(gen2)
                s = time.time()
                _ = get_intersection_polygon_optimized(polygon1, polygon2)
                e = time.time()
                if idx != 0:
                    total_time_opt += e - s
            print(f'optimized\t: {total_time_opt:.5f} sec')
        self.assertGreater(total_time_ori, total_time_opt)
        
    def test_get_area(self):
        n = 10000
        print(f'\nget_area() speed test (iter: {n})')
        with self.subTest('pure python speed'):
            gen = polygon_generator()
            total_time_ori = 0
            for idx in range(n):
                polygon = next(gen)
                s = time.time()
                _ = get_area_original(polygon)
                e = time.time()
                if idx != 0:
                    total_time_ori += e - s
            print(f'original\t: {total_time_ori:.5f} sec')
        with self.subTest('numba optimized python speed'):
            gen = polygon_generator('numpy')
            total_time_opt = 0
            for idx in range(n):
                polygon = next(gen)
                s = time.time()
                _ = get_area_optimized(polygon)
                e = time.time()
                if idx != 0:
                    total_time_opt += e - s
            print(f'optimized\t: {total_time_opt:.5f} sec')
        self.assertGreater(total_time_ori, total_time_opt)

    def test_iou(self):
        n = 10000
        print(f'\niou() speed test (iter: {n})')
        with self.subTest('pure python speed'):
            gen1 = polygon_generator()
            gen2 = polygon_generator()
            total_time_ori = 0
            for idx in range(n):
                polygon1 = next(gen1)
                polygon2 = next(gen2)
                s = time.time()
                _ = iou_original(polygon1, polygon2)
                e = time.time()
                if idx != 0:
                    total_time_ori += e - s
            print(f'original\t: {total_time_ori:.5f} sec')
        with self.subTest('numba optimized python speed'):
            gen1 = polygon_generator('numpy')
            gen2 = polygon_generator('numpy')
            total_time_opt = 0
            for idx in range(n):
                polygon1 = next(gen1)
                polygon2 = next(gen2)
                s = time.time()
                _ = iou_optimized(polygon1, polygon2)
                e = time.time()
                if idx != 0:
                    total_time_opt += e - s
            print(f'optimized\t: {total_time_opt:.5f} sec')
        self.assertGreater(total_time_ori, total_time_opt)        

if __name__ == '__main__':
    unittest.main()