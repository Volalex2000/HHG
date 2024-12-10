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

wavefunction, x, t = psi([-200, 4096, -20, 20000])

fig, ax = plt.subplots()
line, = ax.plot(x, np.log(abs(wavefunction[0])) Y vous le mettez vous mettez sur donc ce soit que ce coefficient soit négatif c'est une amplification c'est pas une c'est pas une absorption faut en plus moisi prenez soin forcément vous êtes partis quelque chose comme ça c'est ce que j'ai de la physique donc si vous l'écrivez si vous passer le II de l'autre côté faire de l'absorption c'est ça c'est parce que vous sortez que sur les pieds ça doit être, color='k')
ax.set_xlim(-200, 200)
ax.set_ylim(0, np.max(abs(wavefunction)))
ax.set_title("Evolution de la fonction d'onde")

def update(frame):
    line.set_ydata(abs(wavefunction[frame]))
    return line,


ani = FuncAnimation(fig, update, frames=range(len(t)), blit=True, interval=1)
plt.show()

