# Tests for shotcounts.py
# usage: pytest -s ...

import numpy as np
from openfermion import QubitOperator, get_sparse_operator
from .util import s2p

from . import shotcounts

def test_shotcounts():

    eps = 0.01 # Error

    # ##########
    # 1-qubit 1-Pauli example
    qubob = QubitOperator("X0")
    psi = np.array([.6,.8],dtype=complex)
    parts = [[ ( (0,'X'), ) , ],]
    # parts = ( ( tuple(tuple((0,'X'))) , ) , )
    res_counts = shotcounts.get_shotcounts_nonoverlapping_parts(qubob, parts,psi,eps)
    gold_counts = (1-.96**2)/eps**2
    np.testing.assert_almost_equal(res_counts,gold_counts)

    # ##########
    # 1-qubit 2-Pauli example
    qubob = QubitOperator("[X0] + 0.5 [Z0]")
    parts = [ [((0,'X'),)],  [((0,'Z'),)] ]
    psi = np.array([.6,.8],dtype=complex)
    res_counts = shotcounts.get_shotcounts_nonoverlapping_parts(qubob, parts,psi,eps)
    var_X0 = 1-.96**2
    var_Z0 = 1-.28**2
    var_halfZ0 = .25*var_Z0
    gold_counts = (np.sqrt(var_X0) + np.sqrt(var_halfZ0))**2/eps**2
    np.testing.assert_almost_equal(res_counts,gold_counts)

    # print('2-qub ------------------')

    # ##########
    # 2-qubit example ( partitioning will be {X0, X0X1} & {Z0} )
    qubob = QubitOperator("[X0] + 0.5 [Z0] + 0.25 [X0 X1]")
    parts = [ [((0,'X'),), ((0,'X'),(1,'X'))],  [((0,'Z'),),]  ]
    # psi = (.6,.8) kron |+>
    psi = np.kron([.6,.8,], [1,1])/np.sqrt(2)
    res_counts = shotcounts.get_shotcounts_nonoverlapping_parts(qubob, parts,psi,eps)
    var_halfZ0 = .25*var_Z0
    var_X0_quarterX0X1 = (17/16 + .5) - (.96 + .25*.96)**2
    gold_counts = (np.sqrt(var_X0_quarterX0X1) + np.sqrt(var_halfZ0))**2/eps**2
    np.testing.assert_almost_equal(res_counts,gold_counts)

    # ##########
    # 2-qubit example that allows for *overalap* (this uses different function, get_shotcounts_from_opsum() )
    # (this example doesn't have overlaps, but we're just testing the function)
    hamsum = [ QubitOperator("[X0] + 0.25 [X0 X1]") , QubitOperator("0.5 [Z0]") ]
    # psi = (.6,.8) kron |+>
    psi = np.kron([.6,.8,], [1,1])/np.sqrt(2)
    res_counts = shotcounts.get_shotcounts_from_opsum(hamsum,psi,eps,nq=None)
    var_halfZ0 = .25*var_Z0
    var_X0_quarterX0X1 = (17/16 + .5) - (.96 + .25*.96)**2
    gold_counts = (np.sqrt(var_X0_quarterX0X1) + np.sqrt(var_halfZ0))**2/eps**2
    np.testing.assert_almost_equal(res_counts,gold_counts)
    # ##########
    # Now with matrix representations instead
    hamsum = [ get_sparse_operator(hamsum[0]), get_sparse_operator(hamsum[1],n_qubits=2) ]
    res_counts = shotcounts.get_shotcounts_from_opsum(hamsum,psi,eps,nq=None)
    np.testing.assert_almost_equal(res_counts,gold_counts)

    # ##########
    # Lower bound shot counts (counts if you diagonalize the entire operator)
    qubop = QubitOperator("[X0] + [Z0]")
    psi = np.kron([1/np.sqrt(2), 1/np.sqrt(2)], [1,0])
    gold_lb = 1/0.01**2  # Easy to work out arithmetically
    res_lb = shotcounts.get_shotcount_lowerbound(qubop,psi,eps)
    np.testing.assert_almost_equal(res_lb,gold_lb)



def test_Rhat():


    # ##########
    # Function for non-overlapping
    qubob = QubitOperator("[X0] - 0.5 [Z0] - 0.25 [X0 X1]")
    parts = [ [((0,'X'),), ((0,'X'),(1,'X'))],
                [((0,'Z'),),]  ]

    Rhat_gold = ( 
                (np.abs(1) + np.abs(-.5) + np.abs(-.25)) /
                (np.sqrt(1**2 + .25**2) + .5 ) 
                )**2
    Rhat_res  = shotcounts.get_Rhat_nonoverlapping_parts(qubob, parts)
    print(Rhat_res)

    assert Rhat_res==Rhat_gold


    # ##########
    # Function for overlapping
    fullqubob = QubitOperator("[X0] - 0.5 [Z0] - 0.25 [X0 X1]")
    op1 = QubitOperator("0.5 [X0] - 0.5 [Z0]")
    op2 = QubitOperator("0.5 [X0] - 0.25 [X0 X1]")
    op_sum = [op1,op2]
    Rhat_gold = ( 
                (np.abs(1) + np.abs(-.5) + np.abs(-.25)) /
                (np.sqrt(.5**2 + .5**2) + np.sqrt(.5**2 + .25**2) ) 
                )**2
    Rhat_res  = shotcounts.get_Rhat_from_opsum(fullqubob, op_sum)
    print(Rhat_res)

    assert Rhat_res==Rhat_gold





if __name__ == "__main__":
    test_shotcounts()
    print("test_shotcounts() passed.")











