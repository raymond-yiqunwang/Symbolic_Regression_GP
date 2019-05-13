import pandas as pd
from IPython.display import Image
import pydotplus

data = pd.read_csv('./forest0.csv', index_col=None, header=0, sep=';')
graph = data['tree'][0]
graph = pydotplus.graphviz.graph_from_dot_data(graph)
Image(graph.create_png())
