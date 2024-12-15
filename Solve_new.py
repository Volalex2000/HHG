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

from datetime import datetime

class CrankNicolson:

    
    def set_grid(self, x_min, x_max, n_x, t_min, t_max, n_t):

        self.x_min, self.x_max, self.n_x = x_min, x_max, n_x
        self.t_min, self.t_max, self.n_t = t_min, t_max, n_t
        self.x_pts, self.delta_x = np.linspace(x_min, x_max, n_x, retstep=True, endpoint=False)
        self.t_pts, self.delta_t = np.linspace(t_min, t_max, n_t, retstep=True, endpoint=False)
        
    def set_parameters(self, f):
        self.f = f

    def out_signal_calculation(self, N):
        def F(t): return self.f(1, t)
        F_v = np.vectorize(F, signature='()->(n)')
        t = self.t_pts[::N]
        n = len(self.x_pts)
        xa = self.x_pts[n//2 - n//8 : n//2 + n//8]
        V = F_v(t)[:, n//2 - n//8 : n//2 + n//8]
        dV_dx = np.gradient(V, xa, axis=1)
        ps = self.psi_matrix[::N, n//2 - n//8 : n//2 + n//8]
        a = np.trapz(-ps * dV_dx * np.conj(ps), xa, axis=1)
        a = np.real(a)
        print(a.shape)
        return a

    def wavelet_trasform(self):
        N = 40
        a = self.out_signal_calculation(N)
        t = self.t_pts[::N]
        A = np.zeros((len(self.scales), len(t)), dtype=complex)

        for i in tqdm(range(len(self.scales)), desc="Wavelet Transform", position=1, leave=True):
            X = self.scales[i] * (t[np.newaxis,:] - t[:,np.newaxis])
            Int = a[:,np.newaxis] * np.sqrt(self.scales[i] / self.tau) * np.exp(-X**2 / (2 * self.tau**2) + 1j * X)
            A[i,:] = np.trapz(Int, t, axis=0)
        
        return A

    def solve(self, psi_init, test=True, sparse=True, boundary_conditions=('dirichlet','dirichlet')):
            
        sig = (1j * self.delta_t) / (4 * self.delta_x**2)
        
        
        # Figure the data type
        data_type = type(sig*psi_init[0])
        
        self.psi_matrix = np.zeros([self.n_t, self.n_x], dtype=data_type)

        # Init fot A

        self.FW = 0.057
        self.max_harm_order = 120
        self.tau = 620.4
        self.scales = self.FW * np.arange(1, self.max_harm_order, 1)
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
            for n in tqdm(range(self.n_t), desc="Time Propagation", position=0, leave=True):
                t = self.t_min + n*self.delta_t
                self.psi_matrix[n,:] = psi
                self.psi_matrix[n,:]*= np.exp(-(self.x_pts/(100))**4)
                fpsi = self.f(psi,t)
                if n==0: fpsi_old = fpsi
                psi = la.solve_banded((1,1),A, B.dot(psi) - 1j*self.delta_t * (1.5 * fpsi - 0.5 * fpsi_old),\
                                    check_finite=False)
                fpsi_old = fpsi
                

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
            for n in tqdm(range(self.n_t), desc="Time Propagation", position=0, leave=True):
                self.psi_matrix[n,:] = psi
                self.psi_matrix[n,:]*= np.exp(-(self.x_pts/(100))**4)
                fpsi = self.f(psi,t)
                if n==0: fpsi_old = fpsi
                psi = la.solve(A, B.dot(psi) - 1j*self.delta_t * (1.5 * fpsi - 0.5 * fpsi_old))
                fpsi_old = fpsi

        self.A = self.wavelet_trasform()
                

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
    
       


def psi_new(set_x_t = None, n_of_exp = 0):
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
    param_single_pulse = pars_YanPengPhysRevA_78_033821()[n_of_exp]
    if n_of_exp == 0:
        Field_single_pulse = Field.Pulse(param_single_pulse)
    else:
        Field_single_pulse = Field.MultiPulse(param_single_pulse)
    
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
        return atom.potential(x) + abs_cos18_potential(x, x_max, alpha=50)
    
    def f(u, t):
        return Potentiel_test(X) * u - X * u * Field_test(t)
    
    crank.set_parameters(f)
    psi_init = atom.ground_state_wavefunction(X)
    
    crank.solve(psi_init)

    current_time = datetime.now()
    np.save(f'results/A_{current_time.strftime("%Y-%m-%d_%H-%M-%S")}.npy', crank.A)
    # np.save(f'results/psi_{current_time.strftime("%Y-%m-%d_%H-%M-%S")}.npy', crank.psi_matrix)
    # np.save(f'results/x_pts_{current_time.strftime("%Y-%m-%d_%H-%M-%S")}.npy', crank.x_pts)
    # np.save(f'results/t_pts_{current_time.strftime("%Y-%m-%d_%H-%M-%S")}.npy', crank.t_pts)
    
    return crank.psi_matrix, crank.x_pts, crank.t_pts, crank.A



