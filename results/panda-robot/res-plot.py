import matplotlib.pyplot as plt
import numpy as np

q = np.linspace(1 / 3, 1, 100)
q_y = np.zeros_like(q) + 0.65
r = np.linspace(-1, 1, 100)
r_y = np.zeros_like(r)
plt.plot(q, r_y)
plt.plot(q_y, r)
plt.plot(4 / 6, -1, "go")
plt.plot(5 / 6, -0.9, "go")
plt.plot(1, -2 / 3, "go")
plt.grid()
plt.show()
