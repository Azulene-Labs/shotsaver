# Functions for the 'Extrapolation Half-Orthogonalizion' algo


import math
from scipy.linalg import expm
import numpy as np
from functools import reduce
import openfermion
import scipy
import scipy.sparse
import scipy.sparse.linalg
from openfermion import QubitOperator

import itertools
from functools import reduce
import numpy.linalg
import os

import scipy
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import expm, expm_multiply


from functools import reduce


from openfermion.ops.operators import (FermionOperator, QubitOperator,
                                       BosonOperator, QuadOperator)
from openfermion.ops.representations import (DiagonalCoulombHamiltonian,
                                             PolynomialTensor)
from openfermion.transforms.opconversions import normal_ordered
from openfermion.utils.indexing import up_index, down_index
from openfermion.utils.operator_utils import count_qubits, is_hermitian

# Make global definitions.
identity_csc = scipy.sparse.identity(2, format='csc', dtype=complex)
pauli_x_csc = scipy.sparse.csc_matrix([[0., 1.], [1., 0.]], dtype=complex)
pauli_y_csc = scipy.sparse.csc_matrix([[0., -1.j], [1.j, 0.]], dtype=complex)
pauli_z_csc = scipy.sparse.csc_matrix([[1., 0.], [0., -1.]], dtype=complex)
q_raise_csc = (pauli_x_csc - 1.j * pauli_y_csc) / 2.
q_lower_csc = (pauli_x_csc + 1.j * pauli_y_csc) / 2.
pauli_matrix_map = {
    'I': identity_csc,
    'X': pauli_x_csc,
    'Y': pauli_y_csc,
    'Z': pauli_z_csc
}


def qubop_from_file(fname,skiplines=1,str_style="pauli_k_type1"):
    """Reads in a QubitOperator from Ham file.
    NOTE THESE FUNCTIONS EXIST:
    * openfermion.utils.save_operator
    * openfermion.utils.load_operator
    If inpOp is a string, returns matrix repr. If inpOp is an np.array,
        it returns the input.
    Args:
        inpOp (string, numpy.array): d-by-d operator
    """

    assert str_style in ["pauli_k_type1",]

    op = QubitOperator()

    with open(fname,'r') as f:

        for ctr in range(skiplines):
            f.readline()

        line = f.readline()

        while line:
            spl = line.split(':')
            term = spl[0]
            if term.strip()=="I":
                term = " "
            k = float(spl[1].split(';')[0])
            op  += QubitOperator(term,k)
            line = f.readline()
    return op


def multi_kron(arr):
    # kronecker product for an arbitrary number of operators as list
    return reduce(np.kron, arr)

# this function evaluate the trottor decomposition to a rotation for evaluation
def rotation(pauli_string, th, nqbit):
    qbOperator = QubitOperator(pauli_string)
    qbOperator_sparse = openfermion.linalg.qubit_operator_sparse(qbOperator,
                                                                 n_qubits=nqbit+1)
    Imat = scipy.sparse.identity(2**(nqbit+1), format='csc', dtype=complex)
    return Imat*math.cos(0.5*th) - 1j*qbOperator_sparse*math.sin(0.5*th)

# this function evaluates <a|H|a> using the trottor decomposition 
def expecval_eho_trotter(hamiltonian, t_tot, m, a):
    # expHdot is exp(-i*Hdot*t) using trotter decomposition
    nQubit = openfermion.utils.count_qubits(hamiltonian)
    
    dt = t_tot/m

    # one qubit quantum basis state, needed for defining the new framework
    Zero = np.array([1, 0])

    # the state |Y+>, as positive eigenvector of pauli-Y
    Yplus = (1/np.sqrt(2))*np.array([1, 1j])

    # re-parametrizing the basis state |a>
    ap = np.kron(Zero, a)
    ab = np.kron(Yplus, a)

    Dt = t_tot/m  # grid size for extrapolated functions

    trot = []
    tau = []
    for i in range(m+1):
        recursive = ap
        for HML_terms in hamiltonian.terms:
            theta = hamiltonian.terms[HML_terms] * i * Dt

            HML_terms_addedQubit = ((0, 'X'),)
            for items in HML_terms:
                HML_terms_addedQubit = HML_terms_addedQubit + ((items[0]+1,
                                                                items[1]),)

            recursive = rotation(HML_terms_addedQubit, 2 * theta, nQubit).dot(recursive)
        term = np.conj(ab).dot(recursive)
        trot.append(np.conj(term)*term)
        
        if (i>0):
            tau.append(i*dt)
    t_matrix = np.array([[tau[i]**(j+1) for j in range(len(tau))] for i in range(len(tau))])
    g_trot = np.array([0.5-trot[i+1] for i in range(m)])
    sol_trot = np.linalg.solve(t_matrix, g_trot)
        
    return (sol_trot)







def expecval_eho_exact_expm(hamiltonian, t_tot, m, a):
    # calculating exact value of exp(-i*Hdot*tau)
    Dt = t_tot/m  # grid size for extrapolated functions
    Zero = np.array([1, 0])
    Yplus = (1/np.sqrt(2))*np.array([1, 1j])
    ap = np.kron(Zero, a)
    ab = np.kron(Yplus, a)
    HML_dot = 0
    for HML_terms in hamiltonian.terms:
        HML_terms_addedQubit = ((0, 'X'),)
        for items in HML_terms:
            HML_terms_addedQubit = HML_terms_addedQubit + ((items[0]+1,
                                                            items[1]),)
        HML_dot += hamiltonian.terms[HML_terms] * QubitOperator(
            HML_terms_addedQubit)

    HML_dot_sparse = openfermion.linalg.qubit_operator_sparse(HML_dot,
                                                              n_qubits=None)

    exp_HML_dot = []
    tau = []
    for i in range(m+1):
        #trm = np.conj(ab).dot(expm(-1j*Dt*i*HML_dot_sparse.toarray()).dot(ap))
        #trm = np.conj(ab).dot(scipy.sparse.linalg.expm(-1j*Dt*i*HML_dot_sparse).dot(ap))
        trm = np.conj(ab).dot(expm_multiply(-1j*Dt*i*HML_dot_sparse, ap, start=None, stop=None, num=None, endpoint=None))
        exp_HML_dot.append(trm*np.conj(trm))
        
        if (i>0):
            tau.append(i*Dt)
    t_matrix = np.array([[tau[i]**(j+1) for j in range(len(tau))] for i in range(len(tau))])
    g = np.array([-0.5+exp_HML_dot[i+1] for i in range(m)])
    sol_exc = np.linalg.solve(t_matrix, g)
    return sol_exc

def g_eho_trotter (HAM, tau, m, a, t):
    sol_trot = expecval_eho_trotter(HAM, tau, m, a)
    return sol_trot[0]*t+ sol_trot[1]*t**2 + sol_trot[2] * t**3


def f_eho_trotter (HAM, tau, m, a, t):
    return 0.5 - g_eho_trotter(HAM, tau, m, a, t)

def f_eho_analytic_1qubit (t):
    return np.sin(0.25*np.pi-t)**2 * np.cos(t)**2

def f_eho_analytic_2qubit (t):
    return 0.5*pow(np.cos(t),4) - (24/25)*np.sin(t)*pow(np.cos(t),3)+(288/390625)*np.sin(t)**2*(288*np.cos(2*t)+337)

def g_eho_expm (HAM, tau, m, a):
    sol_exc = expecval_eho_exact_expm(HAM, tau, m, a)
    t0 = tau /m
    return sol_exc[0]*t0+ sol_exc[1]*t0**2 + sol_exc[2] * t0**3
    
def f_eho_expm (HAM, tau, m, a):
    return 0.5 - g_eho_expm (HAM, tau, m, a)


def get_eho_expecval(HAM, tau, m, a):
    sol_trot = expecval_eho_trotter(HAM, tau, m, a)
    return sol_trot[0]

def get_exact_expecval(HAM,a):
    # calculating exact value of <a|H|a>
    HML_sparse = openfermion.linalg.qubit_operator_sparse(HAM,
                                                          n_qubits=None)
    return np.conj(a).dot(HML_sparse.dot(a))


def get_total_shot_counts_eho (HAM, tau, m, a, error):
    
    t0 = tau/m
    
    V00 = 3/t0
    V01 = -3/(2*t0)
    V02 = 1/(3*t0)
    
    # This is the function capital G based on f as a function of t0
    def capG (x):
        return np.sqrt(1-f_eho_trotter (HAM, tau, m, a, x)**2)
    
    # These are sqrt(alpha) using numerical evaluations
    sqa1 = V00 * capG (t0)
    sqa2 = V01 * capG (2*t0)
    sqa3 = V02 * capG (3*t0)
    
    # total number of measurements (numerical vs analytical)
    Nt = ((sqa1+sqa2+sqa3)/error)**2
    
    return Nt


def get_total_shot_counts_analytic (tau, m, error, N_qubit):
    
    t0 = tau/m
    
    V00 = 3/t0
    V01 = -3/(2*t0)
    V02 = 1/(3*t0)
    
    # This is the function capital G based on f as a function of t0
    def capG_a (x):
        if (N_qubit == 1):
            return np.sqrt(1-f_eho_analytic_1qubit(x)**2)
        elif(N_qubit == 2):
            return np.sqrt(1-f_eho_analytic_2qubit(x)**2)
        else:
            return 0
    
    # These are sqrt(alpha) using numerical evaluations
    sqa1 = V00 * capG_a (t0)
    sqa2 = V01 * capG_a (2*t0)
    sqa3 = V02 * capG_a (3*t0)
    
    # total number of measurements (numerical vs analytical)
    Nt = ((sqa1+sqa2+sqa3)/error)**2
    
    return Nt