import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.linalg as la
from time import process_time
from Parameters import *
from Hydrogen import *
from Field import *

#Import des biblothèques utiles

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
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

wavefunction, x, t = psi([-200, 4096, -200, 300000])

def coupure(p):
    return np.exp(-(p/(2*200))**4)

fig, ax = plt.subplots()
line, = ax.plot(x, coupure(x)*abs(wavefunction[0])**2, color='k')
ax.set_xlim(-200, 200)
ax.set_ylim(0, np.max(abs(wavefunction)))
ax.set_title("Evolution de la fonction d'onde")

def update(frame):
    line.set_ydata(abs(wavefunction[frame]))
    return line,


ani = FuncAnimation(fig, update, frames=range(len(t)), blit=True, interval=1)
plt.show()

