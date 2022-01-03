import plotly.graph_objects as go

import numpy as np

# Make data.
X = np.linspace(3.0, 5.0, 1000)
Y = np.linspace(3.0, 5.0, 1000)
# X = np.array([3, 4, 5])
# Y = np.array([3, 4, 5])
x, y = np.meshgrid(X, Y)

m = 5
a = [3, 5, 2, 1, 7]
b = [5, 2, 1, 4, 9]
c = [1, 2, 5, 2, 3]

Z = np.sum(
    [
        c[i]
        * np.exp(-1 / np.pi * ((x - a[i]) ** 2 + (y - b[i]) ** 2))
        * np.cos(np.pi * ((x - a[i]) ** 2 + (y - b[i]) ** 2))
        for i in range(m)
    ],
    axis=0,
)

fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])

fig.update_layout(
    title="Langermann Function",
    autosize=False,
    width=1280,
    height=720,
    margin=dict(l=65, r=50, b=65, t=90),
)

fig.show()
