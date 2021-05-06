#!/usr/bin/env python
# coding: utf-8

# ### Graph search

# In[3]:


# Dependencies
# !pip3 install chart_studio


# In[4]:


import chart_studio.plotly as py
import plotly.graph_objs as go
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:99% !important; }</style>"))

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
init_notebook_mode(connected=False)


# In[5]:


from enum import IntEnum
import math
import numpy as np


# In[6]:


def create_rotation_matrix(yaw):
    T = np.zeros((len(yaw), 2, 2))
    T[:, 0, 0] = np.cos(yaw)
    T[:, 0, 1] = -np.sin(yaw)
    T[:, 1, 0] = np.sin(yaw)
    T[:, 1, 1] = np.cos(yaw)

    return T
    
class Layer():
    class Id(IntEnum):
        X = 0
        Y = 1
        YAW = 2
        COST = 3
        PARENT = 4
        SIZE = 5

    def __init__(self, N=None, nodes=None):
        assert (N is None) ^ (nodes is None)
        if N is not None:
            self.nodes = np.zeros((N, Layer.Id.SIZE))
        if nodes is not None:
            assert nodes.shape[1] == Layer.Id.SIZE
            self.nodes = nodes
        
    @property
    def x(self):
        return self.nodes[:, Layer.Id.X]
    
    @property
    def y(self):
        return self.nodes[:, Layer.Id.Y]
    
    @property
    def yaw(self):
        return self.nodes[:, Layer.Id.YAW]
    
    @property
    def cost(self):
        return self.nodes[:, Layer.Id.COST]
    
    @property
    def parent(self):
        return self.nodes[:, Layer.Id.PARENT]
    
    @property
    def N(self):
        return self.nodes.shape[0]
    
    @property
    def M(self):
        return self.nodes.shape[1]
    
    
def arc_primitive(c, ds):
    if c == 0:
        return 0, ds, 0
    else:
        dyaw = c * ds
        return dyaw, 1 / c * math.sin(dyaw), 1 / c * (1 - math.cos(dyaw))


class Goal:
    def __init__(self, x, y, threshold):
        self.x = x
        self.y = y
        self.threshold = threshold


class Graph(list):
    def nodes_num(self):
        nodes = 0
        for layer in self:
            nodes += layer.N
        return nodes


def search(curvature_primitives=[-0.2, 0., 0.2], ds=0.5, tree_depth=6, sparse=True, goal=None):
    graph = Graph()
    graph.append(Layer(1))
    
    for i in range(tree_depth):
        X_c = graph[-1]
        X_n = _make_step(X_c, ds, curvature_primitives)
        if sparse:
            X_n = _sparsify(X_n)

        graph.append(X_n)

        goal_id = _goal_reached(X_n, goal)
        if goal_id is not None:
            return graph, _restore_path(graph, goal_id)

    return graph, None


def _make_step(X_c, ds, curvature_primitives):
    N = X_c.N
    X_n = Layer(N * len(curvature_primitives))

    for i, c in enumerate(curvature_primitives):
        # assumme instant change of curvature and movement along circle
        dyaw, dx, dy = arc_primitive(c, ds)
        shift = np.array([dx, dy])

        yaw_c = X_c.yaw
        T = create_rotation_matrix(yaw_c)

        X_n.x[i * N : (i + 1) * N] = X_c.x + T[:, 0] @ shift
        X_n.y[i * N : (i + 1) * N] = X_c.y + T[:, 1] @ shift
        X_n.yaw[i * N : (i + 1) * N] = yaw_c + dyaw
        X_n.parent[i * N : (i + 1) * N] = np.arange(N)
        X_n.cost[i * N : (i + 1) * N] = X_c.cost + c**2
        
    # road constraints
    X_n.nodes = X_n.nodes[X_n.y < 2]
    X_n.nodes = X_n.nodes[X_n.y > -2]

    return X_n


def _sparsify(layer, min_nodes=10, step_x=0.1, step_y=0.1,step_yaw=0.01):
    if layer.N < min_nodes:
        return layer

    def node_to_key(x, y, yaw):
        return (round(x / step_x), round(y / step_y), round(yaw / step_yaw))
    d = {}
    for i in range(layer.N):
        key = node_to_key(layer.x[i], layer.y[i], layer.yaw[i])
        if key in d:
            d[key] = min(d[key], (layer.cost[i], i))
        else:
            d[key] = (layer.cost[i], i)
    indx = list(map(lambda value: value[1][1], d.items()))
    layer.nodes = layer.nodes[indx]

    return layer


def _goal_reached(layer, goal):
    if goal is None:
        return None
    reached = ((layer.x - goal.x) ** 2 + (layer.y - goal.y) ** 2 < goal.threshold ** 2)
    if np.any(reached):
        cost = reached.astype(float) * layer.cost + (~reached).astype(float) * np.max(layer.cost)
        return np.argmin(cost)

def _restore_path(graph, i):
    path = Graph()
    for j in range(len(graph)):
        layer = graph[-j - 1]
        path.append(Layer(nodes=np.copy(layer.nodes[i:i+1])))
        i = int(layer.parent[i])

        # fix parent linkage
        path[-1].parent[:] = 0

    path.reverse()
    return path


# In[7]:


def plot_graph(graph, plot_edges=False):
    # huge number of nodes could freeze browser
    assert graph.nodes_num() < 200000
    
    data = []
    for layer in graph:
        data.append(go.Scatter(
            x=layer.x,
            y=layer.y,
            mode="markers"))

    if plot_edges:
        # huge number of nodes could freeze browser
        assert graph.nodes_num() < 4000
        for prev_layer, layer in zip(graph[:-1], graph[1:]):
            N = prev_layer.N
            for i in range(layer.N):
                data.append(go.Scatter(
                    x=[prev_layer.x[int(layer.parent[i])], layer.x[i]],
                    y=[prev_layer.y[int(layer.parent[i])], layer.y[i]],
                    mode="lines"))

    layout = go.Layout(
        height=600,
        yaxis={
            "scaleanchor": "x",
        },
    )
    figure = go.Figure(data=data, layout=layout)
    iplot(figure, filename="graph")


# In[8]:


tiny_graph, path = search(tree_depth=6, sparse=False, goal=Goal(3, 0, 0.2))

print('The number of nodes in the graph is {}'.format(tiny_graph.nodes_num()))

plot_graph(tiny_graph, plot_edges=True)


# In[9]:


tiny_graph, path = search(tree_depth=6)

print('The number of nodes in the graph is {}'.format(tiny_graph.nodes_num()))

plot_graph(tiny_graph, plot_edges=True)


# In[10]:


huge_graph, path = search(tree_depth=20, goal=Goal(9, 1.5, 0.2))

print('The number of nodes in the graph is {}'.format(huge_graph.nodes_num()))

plot_graph(huge_graph, plot_edges=False)
plot_graph(path, plot_edges=True)


# In[ ]:




