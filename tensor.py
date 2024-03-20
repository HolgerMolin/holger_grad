from __future__ import annotations
from typing import List, Tuple

class Value:
    def __init__(self, value: int | float, _children=()) -> Value:
        self.value = value
        self.grad = 0
        self._backward = lambda: None
        self._children = set(_children)
    
    def __add__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value + other.value, (self, other))

        def _backward():
            other.grad += out.grad
            self.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value * other.value, (self, other))

        def _backward():
            other.grad += out.grad * self.value
            self.grad  += out.grad * other.value
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Haven't figured out pow compute graphs yet"
        out = Value(self.value ** other, (self,))

        def _backward():
            self.grad += ((other - 1) * (self.value ** (other-1))) * out.grad
        out._backward = _backward
        
        return out

    def backward(self) -> None:

        self.grad = 1
        topo = []
        visited = set()

        def build_compute_graph(root: Value) -> None:
            #topological order of nodes in compute graph
            if root not in visited:
                visited.add(root)
                for child in root._children:
                    build_compute_graph(child)
                topo.append(root)
        build_compute_graph(self)

        for node in reversed(topo):
            node._backward()

    def __str__(self):
        return f"Value({self.value}:grad={self.grad})"
    
    def __repr__(self) -> str:
        return f"Value({self.value}:grad={self.grad})"

class Tensor:
    def __init__(self, elements: List[Value] | List[Value]) -> Tensor:
        self.elements = elements
        self.shape = (len(elements),)
    
    def reshape(self, shape: Tuple[int]) -> Tensor:
        pass



def vstack():
    pass
