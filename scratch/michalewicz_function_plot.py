import plotly.graph_objects as go

import numpy as np

# Make data.
X = np.linspace(0, np.pi, 250)
Y = np.linspace(0, np.pi, 250)
x, y = np.meshgrid(X, Y)


Z = (
    np.sin(y) * np.exp((1 - np.cos(x)) ** 2)
    + np.cos(x) * np.exp((1 - np.sin(y)) ** 2)
    + (x - y) ** 2
)

fig = go.Figure(data=[go.Surface(z=-Z, x=x, y=y)])

fig.update_layout(
    title="Michalewicz Function",
    autosize=False,
    width=1280,
    height=720,
    margin=dict(l=65, r=50, b=65, t=90),
)

fig.show()
