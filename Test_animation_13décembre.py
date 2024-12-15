import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.linalg as la
from time import process_time
from Parameters import *
from Hydrogen import *
from Field import *
from matplotlib.animation import FuncAnimation
from Solve import *
import matplotlib.cm as cm

# Vérifiez si psi fonctionne
print("Lancement de psi()")

wavefunction, x, t, A = psi([-200, 4096, -10, 100000])
print("wavefunction shape:", np.shape(wavefunction))
print("x shape:", np.shape(x))
print("t shape:", np.shape(t))

fig, ax = plt.subplots()
print("Checkpoint: plt.subplots() OK")
line, = ax.plot(x, abs(wavefunction[0]), color='k')
ax.set_xlim(-20, 20)
ax.set_ylim(0, np.max(abs(wavefunction)))
ax.set_title("Evolution de la fonction d'onde")


def update(frame):
    try:
        line.set_ydata(abs(wavefunction[frame]))
    except IndexError as e:
        print(f"Erreur d'indice à frame={frame}: {e}")
    return line,


ani = FuncAnimation(fig, update, frames=range(len(t)), blit=True, interval=100)
plt.show()
