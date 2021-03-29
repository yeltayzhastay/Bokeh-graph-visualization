import networkx as nx
import matplotlib.pyplot as plt
import numpy

from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                        MultiLine, Plot, Range1d, WheelZoomTool, PanTool, )
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx

import pandas as pd


def bohek_plot_graph(self, adjust_nodesize):
    pos = nx.spring_layout(self.graph)




    nodesize = [self.graph.nodes[i]['size']/adjust_nodesize for i in self.graph.nodes()]

    edge_mean = numpy.mean([self.graph.edges[(i[0], i[1])]['weight'] for i in self.graph.edges()])
    edge_std_dev = numpy.std([self.graph.edges[(i[0], i[1])]['weight'] for i in self.graph.edges()])
    edgewidth = [((self.graph.edges[(i[0], i[1])]['weight'] + edge_mean)/edge_std_dev) for i in self.graph.edges()]

    adjusted_node_size = dict([(node[0], nsize) for node, nsize in zip(nx.degree(self.graph), nodesize)])
    nx.set_node_attributes(self.graph, name='adjusted_node_size', values=adjusted_node_size)

    adjusted_edge_size = dict([((i[0], i[1]), j)  for i, j in zip(self.graph.edges(), edgewidth)])
    nx.set_edge_attributes(self.graph, name='adjusted_edge_size', values=adjusted_edge_size)



    plot = Plot(plot_width=800, plot_height=800,
                x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1), toolbar_location="below")
                
    plot.title.text = "Graph Interaction Demonstration"

    node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("size", "@size")])
    plot.add_tools(node_hover_tool, PanTool(), WheelZoomTool())
    plot.toolbar.active_scroll = plot.select_one(WheelZoomTool)

    graph_renderer = from_networkx(self.graph, pos, scale=1, center=(0, 0))

    graph_renderer.node_renderer.glyph = Circle(size='adjusted_node_size', fill_color=Spectral4[0])
    graph_renderer.edge_renderer.glyph = MultiLine(line_alpha=0.8, line_width='adjusted_edge_size')
    plot.renderers.append(graph_renderer)

    output_file("interactive_graphs.html")
    show(plot)

df = pd.read_csv('corpora1602.csv')
df = df[['original_text', 'label']]

n_tops = 20 # неше сөз шығу керегі


# get_gram(text, n = бұл граммнын саны, 1 - униграмм, 2 - биграм, 3 - триграм).head(n_tops*10)

# Экстремист текст ушин
unigram_df_t = get_gram(df.loc[df['label'] == 1].original_text.tolist(), 1).head(n_tops*10)
print ("\n\nUnigram extremistical : \n", unigram_df_t.head(n_tops))

# Нейтральный текст ушин
unigram_df_n = get_gram(df.loc[df['label'] == 0].original_text.tolist(), 1).head(n_tops*10)
print ("\n\nUnigram neutral : \n", unigram_df_n.head(n_tops))