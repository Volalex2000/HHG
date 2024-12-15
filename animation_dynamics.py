import numpy as np
import matplotlib.pyplot as plt
from parameters import *
from hydrogen import *
from field import *
from matplotlib.animation import FuncAnimation
from solver import *

# Check if psi function works
print("Launching psi()")

# Generate wavefunction, spatial grid, time grid, and vector potential
wavefunction, x, t, A = psi([-200, 4096, -200, 200000])
print("wavefunction shape:", np.shape(wavefunction))
print("x shape:", np.shape(x))
print("t shape:", np.shape(t))

# Replace infinite and NaN values in the wavefunction
wavefunction[np.isinf(wavefunction)] = 1
wavefunction[np.isnan(wavefunction)] = 1

# Create a figure and axis for the plot
fig, ax = plt.subplots()
print("Checkpoint: plt.subplots() OK")

# Initialize the plot line
line, = ax.plot(x, abs(wavefunction[0]), color='k')
ax.set_yscale('log')
ax.set_ylim(1e-10, 1e2)
ax.set_xlim(-200, 200)
ax.set_title("Evolution of the wavefunction")

# Update function for animation
def update(frame):
    try:
        line.set_ydata(abs(wavefunction[frame]))
    except IndexError as e:
        print(f"Index error at frame={frame}: {e}")
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=range(len(t)), blit=True, interval=1)

# Display the plot
plt.show()
