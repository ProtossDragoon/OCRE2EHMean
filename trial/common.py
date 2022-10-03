import numpy as np
import random
import shapely
import shapely.geometry


def point_generator(type='python'):
    if type == 'numpy':
        while True:
            point = np.array([random.randint(0, 15), random.randint(0, 15)])
            yield point
    elif type == 'shapely':
        while True:
            point = np.array([random.randint(0, 15), random.randint(0, 15)])
            yield shapely.geometry.Point(point)
    else:
        while True:
            point = (random.randint(0, 15), random.randint(0, 15))
            yield point

def polygon_generator(type='python'):
    if type == 'numpy':
        while True:
            polygon = np.array([
                [0, random.randint(3, 5)],
                [10, random.randint(0, 5)],
                [15, random.randint(10, 15)],
                [5, random.randint(15, 20)]
            ])
            yield polygon
    elif type == 'shapely':
        while True:
            polygon = np.array([
                [0, random.randint(3, 5)],
                [10, random.randint(0, 5)],
                [15, random.randint(10, 15)],
                [5, random.randint(15, 20)]
            ])
            yield shapely.geometry.Polygon(polygon)
    else:
        while True:
            polygon = [
                (0, random.randint(3, 5)),
                (10, random.randint(0, 5)),
                (15, random.randint(10, 15)),
                (5, random.randint(15, 20))
            ]
            yield polygon