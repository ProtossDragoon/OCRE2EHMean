import shapely.geometry


def sort_by_linear_ring(points):
    return [shapely.geometry.Point(point) for point in points]

def get_polygon(p1, p2, p3, p4):
    p1, p2, p3, p4 = sort_by_linear_ring([p1, p2, p3, p4])
    polygon = shapely.geometry.Polygon([p1, p2, p3, p4])
    return polygon

def iou(
    polygon1: shapely.geometry.Polygon,
    polygon2: shapely.geometry.Polygon,
) -> float:
    return polygon1.intersection(polygon2).area / polygon1.union(polygon2).area