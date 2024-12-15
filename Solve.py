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
from tqdm import tqdm

import jax
import jax.numpy as jnp
from datetime import datetime

class CrankNicolson:

    
    def set_grid(self, x_min, x_max, n_x, t_min, t_max, n_t):

        self.x_min, self.x_max, self.n_x = x_min, x_max, n_x
        self.t_min, self.t_max, self.n_t = t_min, t_max, n_t
        self.x_pts, self.delta_x = np.linspace(x_min, x_max, n_x, retstep=True, endpoint=False)
        self.t_pts, self.delta_t = np.linspace(t_min, t_max, n_t, retstep=True, endpoint=False)
        
    def set_parameters(self, f):
        self.f = f

    def out_signal_calculation(self, t, psi):
        dV_dx = np.gradient(self.f(1, t), self.x_pts)
        a = np.trapz(-psi * dV_dx[np.newaxis,:] * np.conj(psi), self.x_pts)
        a = np.real(a)
        return a

    def wavelet_trasform(self, t, psi):
        X = self.scales[np.newaxis,:] * (t - self.t_pts[:,np.newaxis])
        wavelet = np.sqrt(self.scales[np.newaxis,:] / self.tau) * np.exp(-X**2 / (2 * self.tau**2) + 1j * X)
        delta_A = self.out_signal_calculation(t, psi)[:,np.newaxis] * wavelet * self.delta_t
        return delta_A

    # @staticmethod
    # @jax.jit
    # def wavelet_trasform(t, psi, f, t_pts, x_pts):

    #     FW = 0.057
    #     max_harm_order = 100
    #     tau = 620.4
    #     scales = FW * np.arange(1, max_harm_order)
    #     scales = jnp.asarray(scales)
    #     t_pts = jnp.asarray(t_pts)
    #     t = jnp.asarray(t)
    #     psi = jnp.asarray(psi)

    #     @jax.jit
    #     def out_signal_calculation(t, psi, f, x_pts):
    #         # Compute the gradient of f(1, t) with respect to x_pts
    #         dV_dx = jax.grad(lambda x: f(1, x))(t)  # Using JAX's auto-diff

    #         # Compute the integrand: -psi * dV_dx * conj(psi)
    #         integrand = -psi * dV_dx * jnp.conj(psi)

    #         # Trapezoidal integration over self.x_pts
    #         dx = x_pts[1] - x_pts[0]  # Assuming uniform spacing
    #         integral = jnp.sum(integrand, axis=-1) * dx

    #         # Return the real part of the integral
    #         return jnp.real(integral)

    #     @jax.jit
    #     # Define the wavelet function for one combination of t_pts and scales
    #     def wavelet_element(t_pt, scale):
    #         X = scale * (t - t_pt)
    #         return jnp.sqrt(scale / tau) * jnp.exp(-X**2 / (2 * tau**2) + 1j * X) * out_signal_calculation(t, psi, f, x_pts)

    #     # Vectorize across t_pts and scales
    #     wavelet_vectorized = jax.vmap(jax.vmap(wavelet_element, in_axes=(None, 0)), in_axes=(0, None))
    #     return np.array(wavelet_vectorized(t_pts, scales))


    def solve(self, psi_init, test=True, sparse=True, boundary_conditions=('dirichlet','dirichlet')):
            
        sig = (1j * self.delta_t) / (4 * self.delta_x**2)
        
        
        # Figure the data type
        data_type = type(sig*psi_init[0])
        
        self.psi_matrix = np.zeros([self.n_t, self.n_x], dtype=data_type)

        # Init fot A

        self.FW = 0.057
        self.max_harm_order = 100
        self.tau = 620.4
        self.scales = self.FW * np.arange(1, self.max_harm_order, 2)
        self.A = np.zeros([self.n_t, len(self.scales)], dtype=data_type)

        # Using sparse matrices and specialized tridiagonal solver speeds up the calculations
        if sparse:
            
            A = self._fillA_sp(sig, self.n_x, data_type)
            B = self._fillB_sp(sig, self.n_x, data_type)
            # Set boundary conditions
            for b in [0,1]:
                if boundary_conditions[b] == 'dirichlet':
                    # u(x,t) = 0
                    A[1,-b] = 1.0
                    A[2*b,1-3*b] = 0.0
                    B[-b,-b] = 0.0
                    B[-b,1-3*b] = 0.0
                elif boundary_conditions[b] == 'neumann':
                    # u'(x,t) = 0
                    A[2*b,1-3*b] = -2*sig
                    B[-b,1-3*b] = 2*sig
                    
            # Propagate
            psi = psi_init
            for n in tqdm(range(self.n_t)):
                t = self.t_min + n*self.delta_t
                self.psi_matrix[n,:] = psi
                self.psi_matrix[n,:]*= np.exp(-(self.x_pts/(2*200))**8)
                fpsi = self.f(psi,t)
                if n==0: fpsi_old = fpsi
                psi = la.solve_banded((1,1),A, B.dot(psi) - 1j*self.delta_t * (1.5 * fpsi - 0.5 * fpsi_old),\
                                    check_finite=False)
                fpsi_old = fpsi

                # Calculate A
                if not test:
                    self.A += self.wavelet_trasform(t, psi)
                #CrankNicolson.wavelet_trasform(t, psi, self.f, self.t_pts, self.x_pts)
                

        else:
            
            A = self._make_tridiag(sig, self.n_x, data_type)
            B = self._make_tridiag(-sig, self.n_x, data_type)

            # Set boundary conditions
            for b in [0,1]:
                if boundary_conditions[b] == 'dirichlet':
                    # u(x,t) = 0
                    A[-b,-b] = 1.0
                    A[-b,1-3*b] = 0.0
                    B[-b,-b] = 0.0
                    B[-b,1-3*b] = 0.0
                
                elif boundary_conditions[b] == 'neumann':
                    # u'(x,t) = 0
                    A[-b,1-3*b] = -2*sig
                    B[-b,1-3*b] = 2*sig

            # Propagate
            psi = psi_init
            for n in range(self.n_t):
                self.psi_matrix[n,:] = psi
                self.psi_matrix[n,:]*= np.exp(-(self.x_pts/(2*200))**8)
                fpsi = self.f(psi,t)
                if n==0: fpsi_old = fpsi
                psi = la.solve(A, B.dot(psi) - 1j*self.delta_t * (1.5 * fpsi - 0.5 * fpsi_old))
                fpsi_old = fpsi

                # Calculate A
                if not test:
                    self.A += self.wavelet_trasform(t, psi)
                #CrankNicolson.wavelet_trasform(t, psi, self.f, self.t_pts, self.x_pts)
                

    def get_final_psi(self):
        
        return self.psi_matrix[-1,:].copy()
        
    def _make_tridiag(self, sig, n, data_type):
    
        M = np.diagflat(np.full(n, (1+2*sig), dtype=data_type)) + \
            np.diagflat(np.full(n-1, -(sig), dtype=data_type), 1) + \
            np.diagflat(np.full(n-1, -(sig), dtype=data_type), -1)

        return M
    
    def _fillA_sp(self, sig, n, data_type):
        """Returns a tridiagonal matrix in compact form ab[1+i-j,j]=a[i,j]"""
        
        A = np.zeros([3,n], dtype=data_type) # A has three diagonals and size n
        A[0] = -(sig) # superdiagonal
        A[1] = 1+2*sig # diagonal
        A[2] = -(sig) # subdiagonal
        return A

    def _fillB_sp(self, sig, n, data_type):
        """Returns a tridiagonal sparse matrix in csr-form"""
        
        _o = np.ones(n, dtype=data_type)
        supdiag = (sig)*_o[:-1]
        diag = (1-2*sig)*_o
        subdiag = (sig)*_o[:-1]
        return scipy.sparse.diags([supdiag, diag, subdiag], [1,0,-1], (n,n), format="csr")
    
       


def psi(set_x_t = None, test = True):
    """
    Solves the time-dependent Schrödinger equation for a given potential and field using the Crank-Nicolson method.
    Parameters:
    set_x_t (tuple, optional): A tuple containing the grid and time parameters in the form (x_min, nx, t_min, nt).
                                If None, default values based on the field parameters will be used.
                                - x_min: Minimum value of the spatial grid.
                                - nx: Number of spatial grid points.
                                - t_min: Minimum value of the time grid.
                                - nt: Number of time grid points.
    Returns:
    tuple: A tuple containing the following elements:
            - psi_matrix (numpy.ndarray): The matrix of the wavefunction values at each grid point and time step.
            - x_pts (numpy.ndarray): The spatial grid points.
            - t_pts (numpy.ndarray): The time grid points.
    """

    crank = CrankNicolson()
    
    # Def du champ laser simple
    param_single_pulse = pars_YanPengPhysRevA_78_033821()[0]
    Field_single_pulse = Field.Pulse(param_single_pulse)
    
    def Field_test(t):
        return Field_single_pulse(t, 'Real')
    
    # Def du potentiel 
    atom = Hydrogen()
    

    
    # Those will not change
    if set_x_t is None:
        l = ((1 / (2 * Field_single_pulse.w)) ** 2) * 0.5 * Field_single_pulse.a
        x_min, x_max, nx = -30 * l, 30 * l, 600
        t_min, t_max, nt = -100, 100., 100
    else:
        x_min = set_x_t[0]
        x_max = -set_x_t[0]
        nx = set_x_t[1]
        t_min = set_x_t[2]
        t_max = -set_x_t[2]
        nt = set_x_t[3]
    
    # Paramétrisation du solveur
    crank.set_grid(x_min, x_max, nx, t_min, t_max, nt)
    X = crank.x_pts
    
    
    def Potentiel_test(x):
        return atom.potential(x) + abs_cos18_potential(x, x_max, alpha=1)
    
    def f(u, t):
        return Potentiel_test(X) * u - X * u * Field_test(t)
    
    crank.set_parameters(f)
    psi_init = atom.ground_state_wavefunction(X)
    
    crank.solve(psi_init, test=test)

    current_time = datetime.now()
    np.save(f'results/A_{current_time.strftime("%Y-%m-%d_%H-%M-%S")}.npy', crank.A)
    np.save(f'results/psi_{current_time.strftime("%Y-%m-%d_%H-%M-%S")}.npy', crank.psi_matrix)
    np.save(f'results/x_pts_{current_time.strftime("%Y-%m-%d_%H-%M-%S")}.npy', crank.x_pts)
    np.save(f'results/t_pts_{current_time.strftime("%Y-%m-%d_%H-%M-%S")}.npy', crank.t_pts)
    
    return crank.psi_matrix, crank.x_pts, crank.t_pts, crank.A



