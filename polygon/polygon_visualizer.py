# 서드파티
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon
import matplotlib.patches
from matplotlib.patches import Polygon as MatPlotlibPolygon
from matplotlib.legend_handler import HandlerTuple

# 프로젝트
from polygon.python_polygon import get_area


class PolygonVisualizer():
    def __init__(self, image_w, image_h, title='Polygon Visualizer'):
        self.polygons = []
        self.image_w = image_w
        self.image_h = image_h
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, image_w)
        self.ax.set_ylim(0, image_h)
        self.ax.grid()
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.invert_yaxis()
        self.ax.set_title(title)

    def draw(self, polygon, color='b'):
        if type(polygon) in [list, np.ndarray]:
            self.polygons.append({
                'shape': polygon,
                'area': get_area(polygon),
                'color': color
            })
        elif type(polygon) is ShapelyPolygon:
            self.polygons.append({
                'shape': list(polygon.exterior.coords),
                'area': polygon.area,
                'color': color
            })
        plt.gca().add_patch(
            MatPlotlibPolygon(
                self.polygons[-1]['shape'], 
                fill=True,
                facecolor=color, 
                edgecolor=color))

    def show(self):
        patches_color = []
        for polygon in self.polygons:
            color = matplotlib.patches.Patch(facecolor=polygon['color']) 
            patches_color.append(color)
        plt.legend(
            handles=patches_color,
            labels=[x['area'] for x in self.polygons],
            handler_map={list: HandlerTuple(ndivide=None, pad=0)},
        )
        plt.show()