# Testing util functions of shotcounts
# usage: pytest -s ...

# import util
from . import util
import pytest
from openfermion import QubitOperator, InteractionOperator
import numpy as np



def test_s2p():

    # ##########
    # Test 1
    s = "X0 Y1 Z2"
    res = util.s2p(s)
    gold = ( (0,'X'), (1,'Y'), (2,'Z') )
    assert gold==res, (gold,res)

    # ##########
    # Test 2
    s = "Y1 Z2"
    res = util.s2p(s)
    gold = ( (1,'Y'), (2,'Z') )
    assert gold==res, (gold,res)

    # ##########
    # Test 3
    s = "IIXZ"
    res = util.s2p(s, 'notnumbered')
    gold = ( (2,'X'), (3,'Z') )
    assert gold==res, (gold,res)

def test_suboperator():

    # ##########
    qubob = QubitOperator("[X0] + 0.5 [Z0] + 0.25 [X0 X1]")
    pstring_set = [ ((0, 'Z'),),    ((0, 'X'), (1, 'X')) ]
    res_sub_qubop = util.get_suboperator(qubob, pstring_set)

    gold_sub_qubop = QubitOperator("0.5 [Z0] + 0.25 [X0 X1]")

    assert gold_sub_qubop==res_sub_qubop, (gold_sub_qubop, res_sub_qubop)


def test_variance():

    # 1-qubit test
    qubob = QubitOperator("2 [X0]")
    psi = np.array([.6,.8],dtype=complex)
    res_var = util.get_variance(qubob, psi)

    gold_var = 4 * ( 1 - .96**2 )

    assert gold_var==res_var, (gold_var,res_var)


def test_paulistr2simplestr():
    
    pstr = [(2,'Z'), (3,'X')]
    gold = "IIZX"
    res =  util.paulistr2simplestr(pstr)
    assert gold==res, (gold,res)

    nq = 6
    gold = "IIZXII"
    res =  util.paulistr2simplestr(pstr, nq)
    assert gold==res, (gold,res)


def test_masks():
    # #####
    # Test rank-2 mask
    gold_rank2 = np.array([ [0, 0, 1, 1, 1],
                            [0, 0, 0, 1, 1],
                            [1, 0, 0, 0, 1],
                            [1, 1, 0, 0, 0],
                            [1, 1, 1, 0, 0]])
    res_rank2 = util.get_rank2_mask(5, 2)
    assert np.allclose(gold_rank2, res_rank2), (gold_rank2, res_rank2)

    # #####
    # Test rank-4 mask
    res_rank4 = util.get_rank4_mask(4, 2)
    # assert res_rank4.shape==(5,5,5,5), res_rank4.shape
    assert res_rank4[0,0,0,0]==0
    assert res_rank4[0,0,0,1]==0
    assert res_rank4[0,0,0,2]==1
    assert res_rank4[0,0,0,3]==1
    #
    assert res_rank4[1,1,1,2]==0
    assert res_rank4[1,1,1,3]==1

    # x = util.get_rank4_mask(3, 2)
    # print("^^^^^")
    # print(np.sum(x))
    # # answer: 38
    # print("^^^^^")

    # #####
    # Test rank-2 distance mask
    res_rank2_diff = util.get_rank2_distance_mask( 5 )
    gold = np.array([[0, 1, 2, 3, 4],
                     [1, 0, 1, 2, 3],
                     [2, 1, 0, 1, 2],
                     [3, 2, 1, 0, 1],
                     [4, 3, 2, 1, 0]])
    assert np.array_equal(gold, res_rank2_diff), (gold, res_rank2_diff)

    # #####
    # Test rank-4 distance mask
    res_r4d = util.get_rank4_distance_mask( 3 )
    # gold = np.array([[0, 1, 2, 3, 4],
    #                  [1, 0, 1, 2, 3],
    #                  [2, 1, 0, 1, 2],
    #                  [3, 2, 1, 0, 1],
    #                  [4, 3, 2, 1, 0]])
    # assert np.array_equal(gold, res_rank2_diff), (gold, res_rank2_diff)

    assert res_r4d[2,2,2,2]==0
    assert res_r4d[2,2,2,1]==1
    assert res_r4d[2,2,2,0]==2
    assert res_r4d[2,2,1,0]==2
    assert res_r4d[0,1,2,2]==2
    assert res_r4d[0,0,2,2]==2




def test_reindexing():


    # #####
    # Test cost functions
    k = 2
    n = 3
    tens2 = 2*np.ones((n,n))
    tens4 = 2*np.ones((n,n,n,n))
    iop = InteractionOperator(1., tens2, tens4)

    cost = util.cost_of_fermion_ordering(iop,k,"1bod_1norm")
    gold = 2*2
    assert gold==cost, (gold,cost)
    cost = util.cost_of_fermion_ordering(iop,k,"2bod_1norm")
    gold = 50*2
    assert gold==cost, (gold,cost)
    cost = util.cost_of_fermion_ordering(iop,k,"1bod2bod_1norm")
    gold = 2*2 + 50*2
    assert gold==cost, (gold,cost)

    cost = util.cost_of_fermion_ordering(iop,k,"1bod_frob")
    gold = 2*2**2
    assert gold==cost, (gold,cost)
    cost = util.cost_of_fermion_ordering(iop,k,"2bod_frob")
    gold = 50*2**2
    assert gold==cost, (gold,cost)
    cost = util.cost_of_fermion_ordering(iop,k,"1bod2bod_frob")
    gold = 2*2**2 + 50*2**2
    assert gold==cost, (gold,cost)


    # #####
    # Test permutation
    t2 = np.array( [[3, 1, 2], 
                    [2, 4, 1], 
                    [1, 2, 5]])
    t4 = np.zeros((3,3,3,3))
    iop = InteractionOperator(1., t2, t4)
    perm = [2, 0, 1]
    res_iop = util.permute_iop_tensors(iop, perm)
    gold_t2 = np.array([[5, 1, 2],
                        [2, 3, 1],
                        [1, 2, 4]])
    assert np.array_equal(gold_t2, res_iop.one_body_tensor), (gold_t2, res_iop.one_body_tensor)


    # #####
    # Test index optimization
    # 3x3 array
    k = 2
    t2 = np.array([[1,  1, -2], 
                   [1,  1,  1], 
                   [-2, 1,  1]])
    t4 = np.ones((3,3,3,3))
    t4[0,0,0,2] = -2
    iop = InteractionOperator(1., t2, t4)

    new_iop,p = util.optimize_indexing_interaction_op(iop,k)

    # Ensuring that the '-2' are not on the boundaries ensures that the optimization is working
    assert new_iop.one_body_tensor[0,2]==new_iop.one_body_tensor[0,2]==1

    # Find the index of the (absolute) maximum value in the flattened tensor
    flat_index = np.argmax( np.abs( new_iop.two_body_tensor ) )

    # Convert the flat index back to 4-dimensional indices
    arg4 = np.unravel_index( flat_index, new_iop.two_body_tensor.shape )

    # Ensure that the maximum (absolute) value is not on the boundaries
    assert max(arg4) - min(arg4) + 1 <= k, (arg4, k)








