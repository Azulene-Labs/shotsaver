# Tests for extrap-half-ortho algo
#
# Usage: pytest -s test_extrap-half-ortho.py
# USAGE: pytest -s test_qwc.py

import pytest

import math
from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce




def pauli_group():
    I  = np.array([[ 1, 0], [ 0, 1]])
    Sx = np.array([[ 0, 1], [ 1, 0]])
    Sy = np.array([[ 0,-1j],[1j, 0]])
    Sz = np.array([[ 1, 0], [ 0,-1]])
    return I, Sx, Sy, Sz

# kronecker product for an arbitrary number of operators sent to the functions as a list
def multi_kron (arr):
    return reduce(np.kron, arr)

# nD rotation matrix. The current parametrization needs an extra (ancilla) qubit. So, I's are one more than nQubit.
# 'qb_string' already holds the extra Sx as input
def rot(nQubit, qb_str, th):
    (I, Sx, Sy, Sz) = pauli_group()
    Imat = [I for i in range (nQubit+1)]
    return multi_kron(Imat)*math.cos(0.5*th) - 1j*multi_kron(qb_str)*math.sin(0.5*th)

# expectation value of 'o' operator respect to v state: <v|o|v>
def ExpVal (v , o):
    return reduce( np.dot, (np.conj(v), o, v) )

#transition amplitude of operator 'o' respect to states v1 and v2: <v1|o|v2>
def TransAmp (v1, o, v2):
    return reduce(np.dot, (np.conj(v1), o, v2))

# this function takes one of X,Y,Z,I characters and return the proper pauli's matrix
def pauli(op):
    (I, Sx, Sy, Sz) = pauli_group()
    return Sx if op =='X' else Sy if op == 'Y' else  Sz if op == 'Z' else I

# this function takes pauli string, i.e. XYZI and return the kronecker product in the matrix form
def pauli_eval (pauli_str):
    pauli_terms = []
    for s in pauli_str:
        pauli_terms.append(pauli(s))
    return multi_kron(pauli_terms)


# expHdot is exp(-i*Hdot*t) using trotter decomposition 
def trotter_eval (nqbit, cf, com_str, ab, ap, t_tot, m):
    
    Dt = t_tot/m  # grid size for extrapolated functions
    (I, Sx, Sy, Sz) = pauli_group()
    expHdot = []
    Imat = [I for i in range (nqbit+1)]
    expHdot.append(multi_kron(Imat))  # the first list item is exp(-i*Hdot*t) at t=0 which is IxIx... (as many as qubits)

    # evaluating the expression exp(-i*tau*Hml_dot) for different time steps
    for i in range(m):
    
        r = []
        for k in range (len(com_str)):
            q_str = []
            th = cf[k] * (i+1)* Dt  # global coefficient of each pauli string exists in the definition of rotation angle
            q_str = [pauli(s) for s in com_str[k]]   # This line converts XYZ to [Sx, Sy, Sz] as a list of matrices
            q_str.insert(0, Sx)  # adding a Sx operator in the new coordinate system
            r.append(rot(nqbit,q_str, 2*th)) # building a list of rotation matrices for each pauli string terms, i.e. XYZ --> [Rx,Ry,Rz]

        expHdot.append(reduce(np.matmul, r))  # matrix multiplication of all rotations corersponding to each term

    # trot and exac are trotter decomposition and exact evaluation of (<ap| exp(-i*Hdot*t)|ab>)^2
    trot = []
    for i in range(m+1):
        term = TransAmp(ab, expHdot[i], ap)
        trot.append((term*np.conj(term)))
    return trot


def exact_eval (H_dot, ab, ap, t_tot, m):
    Dt = t_tot/m
    exc = []
    for i in range(m+1):
        term = TransAmp(ab,expm(-1j*Dt*i*H_dot),ap)
        exc.append((term*np.conj(term)))
    return exc


def plot (tau, trotter, exac):
    
    plt.plot(tau, trotter, marker='o')
    plt.plot(tau, exac, marker='x')
    plt.xlabel('tau')
    plt.ylabel('|<ab|exp(-iHt)|ap>|^2')
    plt.legend(["trotter", "exact"])
    plt.show()


"""
Notes to Mo:
    * For each example, write a 'gold' case, from pen-and-paper
    * Compare gold case to code results
    * Exact equality is not desired: want "almost equal" within say 1e-9
      (https://stackoverflow.com/questions/8560131/pytest-assert-almost-equal)
"""


"""

def test_1q_examples():
    1-qubit examples.
    
    Use |a> = ...
    
    
    # **** Define |a>
    a = np.array( [3./5,4./5] ,dtype=complex)


    # **** H = X
    H = X # define X somewhere else (maybe have simple auxiliary lib for measureme)
    gold_expecval = ... # <a|H|a> (know from pen and paper)
    res_expectval = eho(H, a, {other params... tau_values, desired_precision, etc...})
    assert gold_expecval == pytest.approx(res_expectval,precision)
    # One little thought: should we be outputting the number of circuit repetitions too?
    # or should that be a totally separate function call?
    # If we do want that, we can simply have function do "return [energy, num_repetitions]"


    # **** H = Z + X


    # **** H = Z + 0.5*X



def test_2q_examples():
    2-qubit examples

    pass 
    
    # Define |a>
    a = 
    
    # H = Z Z
    


    # H = XX + YY + 0.5*ZZ + 0.25*ZX
    

"""

def main_func (nQubit, qb_cf, qb_com_str, trotter, exact, Expec_Val, total_tau, m, a):


    # pauli group
    (I, Sx, Sy, Sz) = pauli_group()
    
    # one qubit quantum basis state, needed for defining the new framework
    Zero =  np.array([1,0])
    
    # the state |Y+>, as positive eigenvector of pauli-Y
    Yplus = (1/np.sqrt(2))*np.array([1, 1j])
    
    
    # re-parametrizing the basis state |a>
    aprime = np.kron(Zero, a)
    abar = np.kron(Yplus, a)
    
    # Here we define the Hamiltonian in the matrix form 
    Hml = np.cdouble(np.zeros((2**nQubit,2**nQubit)))
    for i in range (len(qb_com_str)):
        Hml += qb_cf[i]* pauli_eval(qb_com_str[i])
    
    # Here we define the H_dot, as kronecker(X,H), in the matrix form 
    Hml_dot = np.cdouble(np.zeros((2**(nQubit+1),2**(nQubit+1))))
    for i in range (len(qb_com_str)):
        Hml_dot += qb_cf[i]* pauli_eval('X'+qb_com_str[i])
    
    
    tau_tot = total_tau  

    trotter = trotter_eval (nQubit, qb_cf, qb_com_str, abar, aprime, tau_tot, m)
    exact = exact_eval (Hml_dot, abar, aprime, tau_tot, m)
    Expec_Val = ExpVal(a, Hml)
    
    
    
    tau = []
    Dt  = tau_tot/m
    for i in range(m+1):
        tau.append(i*Dt)


    return trotter, Expec_Val

def test_1q_examples():
    total_tau = 0.01  #total evolution time
    m = 3   #number of extrapolation grid points
    a = np.array( [3./5,4./5] ,dtype=complex)
    nQubit = 1
    qb_cf = [1.0, 0.5]
    qb_str = ['Z', 'X']
    
    trotter = []
    exact = []
    Expec_Val = 0
    (trotter, Expec_Val) = main_func (nQubit, qb_cf, qb_str, trotter, exact, Expec_Val, total_tau, m, a)
    
    st = ''
    for i in range(len(qb_cf)):
        st += str(qb_cf[i])+'*'+qb_str[i]
        st += ' + '
    st = st[:-3]
        
    print ("\n_______One Qubit example: "+ st + "______________________________________")
    print( "<a|H|a> using Trotter decom: ",  (0.5-trotter[m])/total_tau)
    print ("<a|H|a> using exact eval: ", Expec_Val)
    print ("The % of error in evaluation of  <a|H|a> is: ", ((0.5-trotter[m])/total_tau-Expec_Val))


def test_2q_examples():
    total_tau = 0.01  #total evolution time
    m = 3   #number of extrapolation grid points
    a = np.array( [1./5,2./5,2./5,4./5] ,dtype=complex)
    nQubit = 2
    qb_cf = [1.0, 1.0, 0.5, 0.25]
    qb_str = ['XX', 'YY', 'ZZ', 'ZX']
    
    trotter = []
    exact = []
    Expec_Val = 0
    (trotter, Expec_Val) = main_func (nQubit, qb_cf, qb_str, trotter, exact, Expec_Val, total_tau, m, a)
    
    st = ''
    for i in range(len(qb_cf)):
        st += str(qb_cf[i])+'*'+qb_str[i]
        st += ' + '
    st = st[:-3]

    print ("\n________Two Qubit example: "+ st + "_______________________________________")
    print( "<a|H|a> using Trotter decom: ",  (0.5-trotter[m])/total_tau)
    print ("<a|H|a> using exact eval: ", Expec_Val)
    print ("The % of error in evaluation of  <a|H|a> is: ", ((0.5-trotter[m])/total_tau-Expec_Val))