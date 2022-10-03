import pygeos

def get_polygon(p1, p2, p3, p4):
    polygon = pygeos.polygons([p1, p2, p3, p4])
    return polygon