import plotly.graph_objects as go

import numpy as np

# Make data.
X = np.linspace(-2, 0, 100)
Y = np.linspace(-2, 0, 100)
x, y = np.meshgrid(X, Y)

Z = -(
    np.sum([i * np.cos((i - 1) * x + i) for i in range(1, 6)], axis=0)
    * np.sum([j * np.cos((j + 1) * y + j) for j in range(1, 6)], axis=0)
    + (x + 1.42513) ** 2
    + (y + 0.80032) ** 2
)


fig = go.Figure(data=[go.Surface(z=-Z, x=x, y=y)])

fig.update_layout(
    title="Levy05 Function",
    autosize=False,
    width=1280,
    height=720,
    margin=dict(l=65, r=50, b=65, t=90),
)

fig.show()
