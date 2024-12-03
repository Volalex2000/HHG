import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.linalg as la
from time import process_time
from Parameters import *
from Hydrogen import *
from Field import *

class CrankNicolson:
    
    def set_grid(self, x_min, x_max, n_x, t_min, t_max, n_t):

        self.x_min, self.x_max, self.n_x = x_min, x_max, n_x
        self.t_min, self.t_max, self.n_t = t_min, t_max, n_t
        self.x_pts, self.delta_x = np.linspace(x_min, x_max, n_x, retstep=True, endpoint=False)
        self.t_pts, self.delta_t = np.linspace(t_min, t_max, n_t, retstep=True, endpoint=False)
        
    def set_parameter(self,f):
        
        self.f = f
    
    def _make_tridiag(self, sig, n, data_type):
        
        M = np.diagflat(np.full(n, (1+2*sig), dtype=data_type)) +\
            np.diagflat(np.full(n-1, (-sig), dtype=data_type), 1) +\
            np.diagflat(np.full(n-1, (-sig), dtype=data_type), -1)
        
        return M
    
    def solve(self, psi_init, sparse=True):
        
        sig = (1j*self.delta_t)/(4*(self.delta_x)**2)
        data_type = type(psi_init[0]*sig)
        
        self.psi_matrix = np.zeros([self.n_t, self.n_x],dtype=data_type)
        
        
        A = self._make_tridiag(sig, self.n_x,data_type)
        B = self._make_tridiag(-sig, self.n_x, data_type)
        
        for i in [0,1]:
            A[1,-i] = 1.0
            A[2*i,1-3*i] = 0.0
            B[-i,-i] = 0.0
            B[-i,1-3*i] = 0.0
            
        psi = psi_init
        for k in range(self.n_t):
            t1 = process_time()
            t = self.t_min + k*self.delta_t
            self.psi_matrix[k,:] = psi
            fpsi = self.f(psi,t)
            if k==0: fpsi_old = fpsi
            psi = la.solve(A, B.dot(psi) - 1j* self.delta_t * (1.5 * fpsi - 0.5 * fpsi_old))
            
            # Normalizing the wave function
            norm = np.sqrt(np.sum(np.abs(psi)**2) * self.delta_x)
            psi = psi / norm
            
            fpsi_old = fpsi
            t2 = process_time()
            print("Time for step ",k," : ",t2-t1)
            
        
    
    def get_final_psi(self):
        return self.psi_matrix[-1,:].copy()


def psi():
    crank = CrankNicolson()
    
    # Def du champ laser simple
    param_single_pulse = pars_YanPengPhysRevA_78_033821()[0]
    Field_single_pulse = Field.Pulse(param_single_pulse)
    
    def Field_test(t):
        return Field_single_pulse(t, 'Real')
    
    # Def du potentiel 
    atom = Hydrogen()
    
    def Potentiel_test(x):
        return atom.potential(x)
    
    # Those will not change
    l = ((1 / (2 * Field_single_pulse.w)) ** 2) * 0.5 * Field_single_pulse.a
    x_min, x_max, nx = -30 * l, 30 * l, 600
    t_min, t_max, nt = -100, 100., 100
    
    # Paramétrisation du solveur
    crank.set_grid(x_min, x_max, nx, t_min, t_max, nt)
    X = crank.x_pts
    
    def f(u, t):
        return Potentiel_test(X) * u - X * u * Field_test(t)
    
    crank.set_parameter(f)
    psi_init = atom.ground_state_wavefunction(X)
    crank.solve(psi_init)
    
    return crank.psi_matrix
