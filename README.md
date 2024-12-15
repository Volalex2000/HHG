Here is the **README.md** file content for your set of files, written in Markdown format:

---

# **High-Harmonic Generation (HHG) Simulation**

This repository contains a numerical simulation of **High-Harmonic Generation (HHG)** in the extreme ultraviolet domain using the **Time-Dependent Schrödinger Equation (TDSE)** and wavelet analysis for time-frequency characteristics.

---

## **Project Description**

The simulation solves the TDSE using the **Crank-Nicolson method** to propagate the wavefunction of a hydrogen-like atom in a strong laser field. The project includes:
1. **Calculation of the wavefunction** during interaction with laser pulses.
2. **Time-frequency analysis** using the **wavelet transform** to extract harmonic spectra.
3. **Visualization** of wavefunction dynamics and HHG spectrum.

---

## **Structure of the Repository**

- **main.ipynb**: Main Jupyter Notebook that runs the entire simulation workflow.
- **solver.py**: Implements the **Crank-Nicolson method** to solve the TDSE.
- **hydrogen.py**: Contains the hydrogen atom model, including:
   - Electrostatic potential
   - Ground state energy
   - Ground state wavefunction
- **field.py**: Defines laser pulse classes:
   - **Pulse**: Single Gaussian laser pulse.
   - **MultiPulse**: Combination of multiple harmonics.
- **parameters.py**: Contains predefined laser pulse parameters (duration, amplitude, frequency, phase) inspired by the **YanPengPhysRev** article.
- **wavelet_visualization.py**: Tools for wavelet-based time-frequency analysis and visualization of HHG spectra.
- **animation_dynamics.py**: Creates animations of the evolving wavefunction.

---

## **Dependencies**

To run this project, install the following Python libraries:

```bash
pip install numpy matplotlib scipy tqdm
```

Optional: Run on **Google Colab** for GPU acceleration.

---

## **Usage**

1. Clone the repository:
   ```bash
   git clone https://github.com/username/HHG_simulation.git
   cd HHG_simulation
   ```

2. Open the **main.ipynb** Jupyter Notebook:
   - Set up laser parameters and grid configurations.
   - Run the Crank-Nicolson solver to compute the wavefunction.
   - Analyze the wavefunction using the **wavelet transform**.

3. Generate and visualize results:
   - **Wavefunction Animation**: View the time evolution of the wavefunction.
   - **Harmonic Spectrum**: Analyze the HHG spectrum using `wavelet_visualization.py`.

---

## **Example Workflow**

The following code snippet demonstrates running the simulation using the default parameters:

```python
from solver import psi

# Solve the TDSE for the given grid and parameters
wavefunction, x, t, A = psi(set_x_t=[-200, 4096, -200, 200000])

# Visualize the time-frequency characteristics
from wavelet_visualization import imshow_time_frequency_characteristics
imshow_time_frequency_characteristics(A)
```

---

## **Results**

1. **Wavefunction Dynamics**:
   - The wavefunction evolves under the influence of a shaped laser pulse.
   - Reflections at boundaries are mitigated using an absorbing potential.

2. **Harmonic Spectrum**:
   - The wavelet transform provides the **harmonic spectrum** in the frequency domain.
   - The **cutoff energy** is calculated and displayed as a reference line.

3. **Time-Frequency Characteristics**:
   - A smooth supercontinuum plateau is visualized, suitable for attosecond pulse generation.

---

## **References**

1. Yan Peng et al., *"Pulse shaping to generate an XUV supercontinuum in the high-order harmonic plateau region"*  
   [Phys. Rev. A 78, 033821 (2008)].

2. Numerical methods inspired by: *"Quantum Mechanics of Strong-Field Interactions"*.

---

## **Contact**

For questions or collaboration, contact:  
**Arthur Gontier** - [arthur.gontier@polytechnique.edu](mailto:arthur.gontier@polytechnique.edu)
**Aleksandr Volkov** - [aleksandr.volkov@polytechnique.edu](mailto:aleksandr.volkov@polytechnique.edu)

---

Let me know if you'd like further refinements!