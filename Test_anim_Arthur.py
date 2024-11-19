import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



psi_matrix = np.load('psi_matrix.npy')
# Préparer la figure
fig, ax = plt.subplots()
line, = ax.plot(np.linspace(0, 20, psi_matrix.shape[1]), np.abs(psi_matrix[0])**2)
ax.set_ylim(0, 1)
ax.set_title("Évolution de |ψ(x, t)|²")
ax.set_xlabel("x")
ax.set_ylabel("|ψ(x, t)|²")

# Fonction d'animation
def update(frame):
    line.set_ydata(np.abs(psi_matrix[frame])**2)
    return line,

# Créer l'animation
ani = FuncAnimation(fig, update, frames=range(psi_matrix.shape[0]), interval=50)
plt.show()
