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

wavefunction, x, t = psi([-200, 4096, -10, 9999])

fig, ax = plt.subplots()
line, = ax.plot(x, abs(wavefunction[0]), color='k')
ax.set_xlim(-50, 50)
ax.set_ylim(0, np.max(abs(wavefunction)))
ax.set_title("Evolution de la fonction d'onde")

def update(frame):
    line.set_ydata(abs(wavefunction[frame]))
    return line,

ani = FuncAnimation(fig, update, frames=range(len(t)), blit=True, interval=10)
plt.show()

