import numpy as np

from utils.mpl_toolkits_patch import ensure_pip_namespace_first

ensure_pip_namespace_first()

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Create grid
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the main surface (translucent)
ax.plot_surface(X, Y, Z, alpha=0.3, color='peachpuff', edgecolor='black', lw=0.5)

# Create the "filled volume" by masking Z values above epsilon
epsilon = 1
Z_filled = np.copy(Z)
Z_filled[Z > epsilon] = np.nan
ax.plot_surface(X, Y, Z_filled, color='royalblue', alpha=0.8)

plt.savefig("")
plt.show()
