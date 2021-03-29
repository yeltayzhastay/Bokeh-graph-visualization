import pandas as pd
import numpy as np

import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (16, 12)
mpl.rcParams['axes.grid'] = False


import networkx as nx
import matplotlib.pyplot as plt
import numpy

from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                        MultiLine, Plot, Range1d, WheelZoomTool, PanTool, )
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx

import pandas as pd




def get_gram(data_list, n):
    vectorizer = CountVectorizer(ngram_range =(n, n))
    #vectorizer = TfidfVectorizer(ngram_range = (n,n))
    X = vectorizer.fit_transform(data_list)
    features = (vectorizer.get_feature_names())
    sums = X.sum(axis = 0)
    data1 = []
    for col, term in enumerate(features):
        data1.append( (term, sums[0,col] ))
    ranking = pd.DataFrame(data1, columns = ['term','value'])
    words = (ranking.sort_values('value', ascending = False))
    return words


def plotting_ngram(gram_df, n):
    gram = pd.DataFrame(gram_df[:n])
    d = gram.set_index('term').T.to_dict('rank')[:1]
    G = nx.Graph()
    for k, v in d[0].items():
        G.add_node(k.split()[0], size=(v * 10))
        G.add_node(k.split()[1], size=(v * 10))
        G.add_edge(k.split()[0], k.split()[1], weight=(v * 10))
    return G

def bohek_plot_graph(graph):
    pos = nx.spring_layout(graph, k=1)
    nodesize = [12 for i in graph.nodes()]
    adjusted_node_size = dict([(node[0], nsize) for node, nsize in zip(nx.degree(graph), nodesize)])
    nx.set_node_attributes(graph, name='adjusted_node_size', values=adjusted_node_size)
    plot = Plot(plot_width=800, plot_height=800,
                x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1), toolbar_location="below")
    plot.title.text = "Graph Interaction Demonstration"
    node_hover_tool = HoverTool(tooltips=[("index", "@index")])
    plot.add_tools(node_hover_tool, PanTool(), WheelZoomTool())
    plot.toolbar.active_scroll = plot.select_one(WheelZoomTool)
    graph_renderer = from_networkx(graph, pos, scale=1, center=(0, 0))
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
bigram_df_t = get_gram(df.loc[df['label'] == 1].original_text.tolist(), 2).head(n_tops*10)
print ("\n\nUnigram extremistical : \n", bigram_df_t.head(n_tops))

# Нейтральный текст ушин
bigram_df_n = get_gram(df.loc[df['label'] == 0].original_text.tolist(), 2).head(n_tops*10)
print ("\n\nUnigram neutral : \n", bigram_df_n.head(n_tops))


bohek_plot_graph(plotting_ngram(bigram_df_n, n_tops))