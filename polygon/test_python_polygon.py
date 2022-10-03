# 내부
import random
import unittest

# 서드파티
import numpy as np

# 프로젝트
from polygon.python_polygon import cross_2d
from polygon.python_polygon import is_point_on_segment
from polygon.python_polygon import does_two_segments_intersect
from polygon.python_polygon import is_point_in_polygon
from polygon.python_polygon import get_points_in_polygon
from polygon.python_polygon import get_intersection_points_between_segments
from polygon.python_polygon import get_intersection_points_between_polygons
from polygon.python_polygon import sort_points_ccw_square
from polygon.python_polygon import sort_points_ccw
from polygon.python_polygon import get_intersection_polygon
from polygon.python_polygon import get_area
from polygon.python_polygon import iou

class TestPurePolygon(unittest.TestCase):
    def test_cross(self):
        p1 = np.array([1, 1])
        p2 = np.array([2, 2])
        p3 = np.array([3, 3])
        self.assertEqual(cross_2d(p1, p2, p3), np.cross(p2 - p1, p3 - p1))

        p1 = np.array([1, 1])
        p2 = np.array([2, 2])
        p3 = np.array([3, 4])
        self.assertEqual(cross_2d(p1, p2, p3), np.cross(p2 - p1, p3 - p1))

        p1 = np.array([1, 1])
        p2 = np.array([2, 2])
        p3 = np.array([4, 3])
        self.assertEqual(cross_2d(p1, p2, p3), np.cross(p2 - p1, p3 - p1))

        p1 = np.array([1, 1])
        p2 = np.array([2, 2])
        p3 = np.array([4, 4])
        self.assertEqual(cross_2d(p1, p2, p3), np.cross(p2 - p1, p3 - p1))

        p1 = np.array([1, 1])
        p2 = np.array([2, 2])
        p3 = np.array([3, 2])
        self.assertEqual(cross_2d(p1, p2, p3), np.cross(p2 - p1, p3 - p1))

        p1 = np.array([1, 1])
        p2 = np.array([2, 2])
        p3 = np.array([2, 3])
        self.assertEqual(cross_2d(p1, p2, p3), np.cross(p2 - p1, p3 - p1))
                
    def test_is_point_on_segment(self):
        p_0_0 = (0, 0)
        p_0_1 = (0, 1)
        p_1_0 = (1, 0)
        p_1_1 = (1, 1)
        p_2_2 = (2, 2)
        self.assertEqual(is_point_on_segment(p_0_1, (p_2_2, p_1_1)), False) # 외부
        self.assertEqual(is_point_on_segment(p_1_0, (p_1_1, p_2_2)), False) # 외부
        self.assertEqual(is_point_on_segment(p_0_0, (p_0_0, p_0_1)), True) # 끝점
        self.assertEqual(is_point_on_segment(p_0_1, (p_0_0, p_0_1)), True) # 끝점
        self.assertEqual(is_point_on_segment(p_1_1, (p_0_0, p_2_2)), True) # 포함
        self.assertEqual(is_point_on_segment(p_1_1, (p_2_2, p_0_0)), True) # 포함

    def test_does_two_segments_intersect(self):
        p_0_0 = (0, 0)
        p_0_1 = (0, 1)
        p_0_2 = (0, 2)
        p_1_0 = (1, 0)
        p_1_1 = (1, 1)
        p_2_0 = (2, 0)
        p_2_2 = (2, 2)
        p_3_3 = (3, 3)
        self.assertEqual(does_two_segments_intersect((p_0_0, p_0_1), (p_1_0, p_1_1)), False) # 평행
        self.assertEqual(does_two_segments_intersect((p_1_0, p_1_1), (p_0_0, p_0_1)), False) # 평행
        self.assertEqual(does_two_segments_intersect((p_0_0, p_1_1), (p_0_0, p_1_1)), True) # 포개짐, 완벽히 동일한 선분
        self.assertEqual(does_two_segments_intersect((p_0_0, p_1_1), (p_1_1, p_0_0)), True) # 포개짐, 완벽히 동일한 선분
        self.assertEqual(does_two_segments_intersect((p_0_0, p_2_2), (p_1_1, p_0_0)), True) # 포개짐
        self.assertEqual(does_two_segments_intersect((p_2_2, p_0_0), (p_0_0, p_1_1)), True) # 포개짐
        self.assertEqual(does_two_segments_intersect((p_2_2, p_3_3), (p_1_1, p_2_2)), True) # 포개짐
        self.assertEqual(does_two_segments_intersect((p_0_2, p_2_0), (p_1_1, p_2_2)), True) # 한 선분의 끝점이 다른 선분과 겹침
        self.assertEqual(does_two_segments_intersect((p_0_0, p_1_1), (p_1_1, p_2_2)), True) # 한 선분의 끝점이 다른 선분의 끝점과 겹침
        self.assertEqual(does_two_segments_intersect((p_0_0, p_0_1), (p_0_2, p_1_1)), False) # 서로 닿지는 않지만, 한 선분의 연장선이 다른 선분의 끝점을 지나감
        self.assertEqual(does_two_segments_intersect((p_0_0, p_0_1), (p_1_1, p_2_2)), False) # 서로 닿지도 않고 한 선분의 연장선이 다른 선분의 끝점을 지나가지도 않는 경우
        self.assertEqual(does_two_segments_intersect((p_0_0, p_3_3), (p_2_0, p_0_2)), True) # 누가 봐도 교차하는 경우

    def test_is_point_in_polygon(self):
        polygon = [(0, 0), (1, 0), (2, 2), (0, 1)]
        # 꼭짓점
        point = (2, 2)
        self.assertEqual(is_point_in_polygon(polygon, point), False)
        # 꼭짓점
        point = (1, 0)
        self.assertEqual(is_point_in_polygon(polygon, point), False)
        # 모서리 중점
        point = (0.5, 0)
        self.assertEqual(is_point_in_polygon(polygon, point), False)
        # 내부
        point = (0.5, 0.5)
        self.assertEqual(is_point_in_polygon(polygon, point), True)
        # 외부, polygon bbox 내부
        point = (0.5, 1.5)
        self.assertEqual(is_point_in_polygon(polygon, point), False)
        point = (1, 1.5)
        self.assertEqual(is_point_in_polygon(polygon, point), False)
        # 외부, polygon bbox 외부
        point = (2.5, 2.5)
        self.assertEqual(is_point_in_polygon(polygon, point), False)

    def test_get_points_in_polygon(self):
        # case 1
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 3), (4, 3), (4, 5), (1, 5)]
        self.assertEqual(
            set(get_points_in_polygon(polygon1, polygon2)),
            set([]))
        # case 2
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 3), (5, 3), (5, 5), (1, 5)]
        self.assertEqual(
            set(get_points_in_polygon(polygon1, polygon2)),
            set([]))
        # case 3
        polygon1 = [(0, 0), (4, 0), (4, 3), (0, 3)]
        polygon2 = [(4, 0), (5, 0), (5, 3), (4, 3)]
        self.assertEqual(
            set(get_points_in_polygon(polygon1, polygon2)),
            set([]))
        # case 4
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 0), (4, 0), (4, 2), (1, 2)]
        self.assertEqual(
            set(get_points_in_polygon(polygon1, polygon2)),
            set([(1, 2), (4, 2)])
        )
        # case 5
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(0, 0), (3, 0), (3, 2), (0, 2)]
        self.assertEqual(
            set(get_points_in_polygon(polygon1, polygon2)),
            set([(3, 2)])
        )
        # case 6
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(4, 0), (5, 0), (5, 3), (4, 3)]
        self.assertEqual(
            set(get_points_in_polygon(polygon1, polygon2)),
            set([])
        )
        # case 7
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        self.assertEqual(
            set(get_points_in_polygon(polygon1, polygon2)),
            set([])
        )        
        # case 8
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 0), (4, 0), (4, 6), (1, 6)]
        self.assertEqual(
            set(get_points_in_polygon(polygon1, polygon2)),
            set([])
        )
        # case 9
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(0, 0), (3, 0), (3, 4), (0, 4)]
        self.assertEqual(
            set(get_points_in_polygon(polygon1, polygon2)),
            set([])
        )
        # case 10
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 1), (3, 1), (3, 2), (1, 2)]
        self.assertEqual(
            set(get_points_in_polygon(polygon1, polygon2)),
            set([(1, 1), (3, 1), (3, 2), (1, 2)])
        )
        # case 11
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 1), (3, 1), (3, 6), (1, 6)]
        self.assertEqual(
            set(get_points_in_polygon(polygon1, polygon2)),
            set([(1, 1), (3, 1)])
        )
        # case 12
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(5, 3), (6, 4), (5, 5), (4, 4)]
        self.assertEqual(
            set(get_points_in_polygon(polygon1, polygon2)),
            set([])
        )
        # case 13
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(2, 0), (4, 1), (2, 2), (0, 1)]
        self.assertEqual(
            set(get_points_in_polygon(polygon1, polygon2)),
            set([(4, 1), (2, 2)])
        )
        # case 14
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(2, 2), (3, 4), (2, 6), (1, 4)]
        self.assertEqual(
            set(get_points_in_polygon(polygon1, polygon2)),
            set([(2, 2)])
        )

    def test_get_intersection_points_between_segments(self):
        segment1 = [(0, 0), (3, 3)]
        segment2 = [(3, 1), (5, 3)]
        self.assertEqual(set(get_intersection_points_between_segments(segment1, segment2)), set([]))
        self.assertEqual(set(get_intersection_points_between_segments(segment2, segment1)), set([]))
        segment2 = [(4, 4), (6, 6)]
        self.assertEqual(set(get_intersection_points_between_segments(segment1, segment2)), set([]))
        self.assertEqual(set(get_intersection_points_between_segments(segment2, segment1)), set([]))
        segment2 = [(3, 3), (5, 5)]
        self.assertEqual(set(get_intersection_points_between_segments(segment1, segment2)), set([(3, 3)]))
        self.assertEqual(set(get_intersection_points_between_segments(segment2, segment1)), set([(3, 3)]))
        segment2 = [(2, 2), (5, 5)]
        self.assertEqual(set(get_intersection_points_between_segments(segment1, segment2)), set([(2, 2), (3, 3)]))
        self.assertEqual(set(get_intersection_points_between_segments(segment2, segment1)), set([(2, 2), (3, 3)]))
        segment2 = [(1, 1), (2, 2)]
        self.assertEqual(set(get_intersection_points_between_segments(segment1, segment2)), set([(1, 1), (2, 2)]))
        self.assertEqual(set(get_intersection_points_between_segments(segment2, segment1)), set([(1, 1), (2, 2)]))
        segment2 = [(1, 1), (3, 3)]
        self.assertEqual(set(get_intersection_points_between_segments(segment1, segment2)), set([(1, 1), (3, 3)]))
        self.assertEqual(set(get_intersection_points_between_segments(segment2, segment1)), set([(1, 1), (3, 3)]))
        segment2 = [(0, 0), (3, 3)]
        self.assertEqual(set(get_intersection_points_between_segments(segment1, segment2)), set([(0, 0), (3, 3)]))
        self.assertEqual(set(get_intersection_points_between_segments(segment2, segment1)), set([(0, 0), (3, 3)]))
        segment2 = [(4, 2), (4, 5)]
        self.assertEqual(set(get_intersection_points_between_segments(segment1, segment2)), set([]))
        self.assertEqual(set(get_intersection_points_between_segments(segment2, segment1)), set([]))
        segment2 = [(4, 1), (4, 4)]
        self.assertEqual(set(get_intersection_points_between_segments(segment1, segment2)), set([]))
        self.assertEqual(set(get_intersection_points_between_segments(segment2, segment1)), set([]))
        segment2 = [(3, 0), (3, 3)]
        self.assertEqual(set(get_intersection_points_between_segments(segment1, segment2)), set([(3, 3)]))
        self.assertEqual(set(get_intersection_points_between_segments(segment2, segment1)), set([(3, 3)]))
        segment2 = [(2, 0), (2, 4)]
        self.assertEqual(set(get_intersection_points_between_segments(segment1, segment2)), set([(2, 2)]))
        self.assertEqual(set(get_intersection_points_between_segments(segment2, segment1)), set([(2, 2)]))

    def test_get_intersection_points_between_polygons(self):
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 3), (4, 3), (4, 5), (1, 5)]
        self.assertEqual(
            set(get_intersection_points_between_polygons(polygon1, polygon2)), 
            set([(1, 3), (4, 3)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 3), (5, 3), (5, 5), (1, 5)]
        self.assertEqual(
            set(get_intersection_points_between_polygons(polygon1, polygon2)), 
            set([(1, 3), (5, 3)]))
        polygon1 = [(0, 0), (4, 0), (4, 3), (0, 3)]
        polygon2 = [(4, 0), (5, 0), (5, 3), (4, 3)]
        self.assertEqual(
            set(get_intersection_points_between_polygons(polygon1, polygon2)), 
            set([(4, 0), (4, 3)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 0), (4, 0), (4, 2), (1, 2)]
        self.assertEqual(
            set(get_intersection_points_between_polygons(polygon1, polygon2)), 
            set([(1, 0), (4, 0)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(0, 0), (3, 0), (3, 2), (0, 2)]
        self.assertEqual(
            set(get_intersection_points_between_polygons(polygon1, polygon2)), 
            set([(0, 0), (3, 0), (0, 2)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(4, 0), (5, 0), (5, 3), (4, 3)]
        self.assertEqual(
            set(get_intersection_points_between_polygons(polygon1, polygon2)), 
            set([(4, 0), (4, 3), (5, 0), (5, 3)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        self.assertEqual(
            set(get_intersection_points_between_polygons(polygon1, polygon2)), 
            set([(0, 0), (5, 0), (5, 3), (0, 3)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 0), (4, 0), (4, 6), (1, 6)]
        self.assertEqual(
            set(get_intersection_points_between_polygons(polygon1, polygon2)), 
            set([(1, 0), (4, 0), (1, 3), (4, 3)]))                
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(0, 0), (3, 0), (3, 4), (0, 4)]
        self.assertEqual(
            set(get_intersection_points_between_polygons(polygon1, polygon2)), 
            set([(0, 0), (3, 0), (0, 3), (3, 3)]))                
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 1), (3, 1), (3, 2), (1, 2)]
        self.assertEqual(
            set(get_intersection_points_between_polygons(polygon1, polygon2)), 
            set([]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(1, 1), (3, 1), (3, 6), (1, 6)]
        self.assertEqual(
            set(get_intersection_points_between_polygons(polygon1, polygon2)), 
            set([(1, 3), (3, 3)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(5, 3), (6, 4), (5, 5), (4, 4)]
        self.assertEqual(
            set(get_intersection_points_between_polygons(polygon1, polygon2)), 
            set([(5, 3)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(2, 0), (4, 1), (2, 2), (0, 1)]
        self.assertEqual(
            set(get_intersection_points_between_polygons(polygon1, polygon2)), 
            set([(2, 0), (0, 1)]))
        polygon1 = [(0, 0), (5, 0), (5, 3), (0, 3)]
        polygon2 = [(2, 2), (3, 4), (2, 6), (1, 4)]
        self.assertEqual(
            set(get_intersection_points_between_polygons(polygon1, polygon2)), 
            set([(1.5, 3), (2.5, 3)]))

    def test_sort_points_ccw_square(self):
        # 직사각형
        # 직사각형의 좌하단을 기준삼아 정렬
        li = [(0, 0), (0, 3), (5, 3), (5, 0)]
        polygon = [(0, 0), (5, 0), (5, 3), (0, 3)]
        self.assertEqual(sort_points_ccw_square(li), polygon)
        li = [(1, 3), (4, 5), (4, 3), (1, 5)]
        polygon = [(1, 3), (4, 3), (4, 5), (1, 5)]
        self.assertEqual(sort_points_ccw_square(li), polygon)
        li = [(1, 3), (5, 5), (1, 5), (5, 3)]
        polygon = [(1, 3), (5, 3), (5, 5), (1, 5)]
        self.assertEqual(sort_points_ccw_square(li), polygon)
        # 직사각형
        # 직사각형의 우하단을 기준삼아 정렬
        li = [(4, 0), (0, 3), (0, 0), (4, 3)]
        polygon = [(4, 0), (4, 3), (0, 3), (0, 0)]
        self.assertEqual(sort_points_ccw_square(li), polygon)
        li = [(4, 0), (4, 2), (1, 0), (1, 2)]
        polygon = [(4, 0), (4, 2), (1, 2), (1, 0)]
        self.assertEqual(sort_points_ccw_square(li), polygon)
        # 직사각형
        # 직사각형의 우상단을 기준삼아 정렬
        li = [(3, 2), (0, 0), (3, 0), (0, 2)]
        polygon = [(3, 2), (0, 2), (0, 0), (3, 0)]
        self.assertEqual(sort_points_ccw_square(li), polygon)
        li = [(5, 3), (4, 3), (5, 0), (4, 0)]
        polygon = [(5, 3), (4, 3), (4, 0), (5, 0)]
        self.assertEqual(sort_points_ccw_square(li), polygon)
        # 직사각형
        # 직사각형의 좌상단을 기준삼아 정렬
        li = [(1, 6), (4, 0), (4, 6), (1, 0)]
        polygon = [(1, 6), (1, 0), (4, 0), (4, 6)]
        self.assertEqual(sort_points_ccw_square(li), polygon)
        li = [(0, 4), (0, 0), (3, 4), (3, 0)]
        polygon = [(0, 4), (0, 0), (3, 0), (3, 4)]
        self.assertEqual(sort_points_ccw_square(li), polygon)
        li = [(1, 2), (3, 1), (3, 2), (1, 1)]
        polygon = [(1, 2), (1, 1), (3, 1), (3, 2)]
        self.assertEqual(sort_points_ccw_square(li), polygon)
        li = [(1, 6), (3, 1), (1, 1), (3, 6)]
        polygon = [(1, 6), (1, 1), (3, 1), (3, 6)]
        self.assertEqual(sort_points_ccw_square(li), polygon)

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
            ret = sort_points_ccw([polygon[0]] + todo)
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
            [(1, 0), (4, 0), (4, 6), (1, 6)],
            [(0, 0), (3, 0), (3, 4), (0, 4)],
            [(1, 1), (3, 1), (3, 2), (1, 2)],
            [(1, 1), (3, 1), (3, 6), (1, 6)],
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
                    len(get_intersection_polygon(polygon1, polygon2)),
                    len(polygon_intersection)
                )
                # 동일한 요소들이 존재하는지 검사
                self.assertEqual(
                    set(get_intersection_polygon(polygon1, polygon2)),
                    set(polygon_intersection)
                )
                # 순서가 맞는지 검사
                ret = get_intersection_polygon(polygon1, polygon2)
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
                self.assertEqual(get_area(polygon), area)
            with self.subTest(polygon=polygon[::-1]):
                self.assertEqual(get_area(polygon[::-1]), area)
                
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
                    iou(polygon1, polygon2),
                    intersection_area)
    

if __name__ == '__main__':
    unittest.main()