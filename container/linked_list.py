# 내장
import logging

# 서드파티
import numpy as np

# 프로젝트
from runtime import GlobalRuntime
import polygon.shapely_polygon as shapely_polygon

WARNING_ONCE = True


class Node():
    def __init__(
        self, 
        polygon:list, 
        left_top:tuple,
        right_bottom:tuple,
        sentence:str,
        use_shapely_type_polygon:bool = False,
        **kwargs,
    ):
        global WARNING_ONCE
        self.logger = logging.getLogger(self.__class__.__name__)
        if polygon:
            if GlobalRuntime.is_numba_runtime():
                if use_shapely_type_polygon and WARNING_ONCE:
                    self.logger.warning("Numba runtime does not support shapely type polygon. Automatically change to numpy type polygon.")
                    WARNING_ONCE = False
                self.polygon = np.array([list(point) for point in polygon])
            elif GlobalRuntime.is_python_runtime():
                if use_shapely_type_polygon:
                    self.polygon = shapely_polygon.get_polygon(*polygon)
                else:
                    self.polygon = polygon
        self.left_top = left_top
        self.right_bottom = right_bottom
        self.sentence = sentence
        self.prev = None
        self.next = None


class LinkedList():
    def __init__(
        self,
        image_w:int = None,
        image_h:int = None,
    ) -> None:
        self.size = 0
        self.head = Node([], (), (), 'head')
        self.tail = Node([], (), (), 'tail')
        self.head.next = self.tail
        self.tail.prev = self.head
        self.image_w = image_w
        self.image_h = image_h

    def __len__(self):
        return self.size

    def __iter__(self):
        self.iter_current = self.head.next
        return self

    def __next__(self) -> Node:
        if self.iter_current is not self.tail:
            ret = self.iter_current
            self.iter_current = self.iter_current.next
            return ret
        else:
            raise StopIteration

    def append(self, node: Node):
        self.add_last(node)

    def add_last(self, node: Node):
        self.link_to(self.tail.prev, node)

    def add_first(self, node: Node):
        self.link_to(self.head, node)

    def link_to(self, prev_node: Node, new_node: Node):
        self.size +=1
        next_node = prev_node.next
        next_node.prev = new_node
        prev_node.next = new_node
        new_node.prev = prev_node
        new_node.next = next_node