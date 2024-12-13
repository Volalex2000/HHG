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
try:
    wavefunction, x, t, A = psi([-200, 4096, -50, 200000])
    print("wavefunction shape:", np.shape(wavefunction))
    print("x shape:", np.shape(x))
    print("t shape:", np.shape(t))
except Exception as e:
    print(f"Erreur dans psi(): {e}")
    exit()

def coupure(p):
    return np.exp(-(p/(2*200))**4)

try:
    fig, ax = plt.subplots()
    print("Checkpoint: plt.subplots() OK")
    line, = ax.plot(x, coupure(x)*abs(wavefunction[0])**2, color='k')
    ax.set_xlim(-200, 200)
    ax.set_ylim(0, np.max(abs(wavefunction)))
    ax.set_title("Evolution de la fonction d'onde")
except Exception as e:
    print(f"Erreur avec plt.subplots() ou les premières configurations de la figure : {e}")
    exit()

def update(frame):
    try:
        line.set_ydata(abs(wavefunction[frame]))
    except IndexError as e:
        print(f"Erreur d'indice à frame={frame}: {e}")
    return line,

try:
    ani = FuncAnimation(fig, update, frames=range(len(t)), blit=True, interval=1)
    plt.show()
except Exception as e:
    print(f"Erreur avec l'animation FuncAnimation : {e}")
