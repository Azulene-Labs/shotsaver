''' Tests for extrap-half-ortho algo'''

# Usage: pytest -s test_extrap_half_ortho.py

import pytest
import time
import numpy as np
from openfermion import QubitOperator
import openfermion
import os
from scipy import linalg
from shotsaver import extrap_half_ortho


def f_eho_analytic_1qubit (t):
    return np.sin(0.25*np.pi-t)**2 * np.cos(t)**2

def f_eho_analytic_2qubit (t):
    return 0.5*pow(np.cos(t),4) - (24/25)*np.sin(t)*pow(np.cos(t),3)+(288/390625)*np.sin(t)**2*(288*np.cos(2*t)+337)



def test_get_eho_expecval():
    
    HAM = extrap_half_ortho.qubop_from_file(os.getcwd()+"/_1QB_hamiltonian.ham")
    tau = 0.01
    m=3
    a = np.array( [1.0,0.0], dtype=complex)
    
    assert extrap_half_ortho.get_eho_expecval(HAM, tau, m, a) == pytest.approx(extrap_half_ortho.get_exact_expecval(HAM, a)  , abs=0.01)



def test_get_total_shot_counts_eho_1qubit():
    
    HAM = extrap_half_ortho.qubop_from_file(os.getcwd()+"/_1QB_hamiltonian.ham")
    m = 3
    tau = 0.1
    a = np.array( [1.0,0.0], dtype=complex)
    error = 0.0005
    
    #assert extrap_half_ortho.get_total_shot_counts_eho (HAM, tau, m, a, error) == pytest.approx(extrap_half_ortho.get_total_shot_counts_analytic(tau, m, error, 1), abs = 0.01)
    print (extrap_half_ortho.get_total_shot_counts_eho (HAM, tau, m, a, error))
    print (extrap_half_ortho.get_total_shot_counts_analytic(tau, m, error, 1))
    
    

def test_get_total_shot_counts_eho_2qubit():
    
    HAM = extrap_half_ortho.qubop_from_file(os.getcwd()+"/_2QB_hamiltonian.ham")
    tau = 0.01
    m=3
    a = np.array( [9.0/25.0, 12.0/25.0, 12.0/25.0, 16.0/25.0] ,dtype=complex)
    error = 0.01
    
    assert extrap_half_ortho.get_total_shot_counts_eho (HAM, tau, m, a, error) == pytest.approx(extrap_half_ortho.get_total_shot_counts_analytic(tau, m, error, 2), abs = 0.01)

    
    
def test_f_1qubit ():
    HAM = extrap_half_ortho.qubop_from_file(os.getcwd()+"/_1QB_hamiltonian.ham")
    tau = 0.01
    m=3
    a = np.array( [1.0,0.0], dtype=complex)
    error = 0.01
    
    test_tau = tau/m
    
    assert extrap_half_ortho.f_eho_trotter (HAM, tau, m, a, test_tau) == pytest.approx(f_eho_analytic_1qubit (test_tau), abs = 0.01)
    
    
def test_f_2qubit ():
    HAM = extrap_half_ortho.qubop_from_file(os.getcwd()+"/_2QB_hamiltonian.ham")
    tau = 0.01
    m=3
    a = np.array( [9.0/25.0, 12.0/25.0, 12.0/25.0, 16.0/25.0] ,dtype=complex)
    error = 0.01
    
    test_tau = tau/m
    
    assert extrap_half_ortho.f_eho_trotter (HAM, tau, m, a, test_tau) == pytest.approx(f_eho_analytic_2qubit (test_tau), abs = 0.01)
    
    