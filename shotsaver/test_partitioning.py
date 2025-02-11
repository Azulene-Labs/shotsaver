# Tests for partitioning.py
# usage: pytest -s test_partitioning.py

from openfermion import QubitOperator
from . import partitioning
from . import util

import pprint


def test_term_ordering():
    """Test if term ordering is correct"""

    op = QubitOperator("5 [X1] + 2 [X2] + 8 [X3] + 1 [X4] + -9 [X5]")
    gold = [ ((5, 'X'),), ((3, 'X'),), ((1, 'X'),), ((2, 'X'),), ((4, 'X'),) ]
    res = partitioning.get_terms_ordered_by_abscoeff(op)
    
    assert gold==res, (gold,res)


def test_si_groupings():
    """Test if groupings are correct"""

    # X0 + Y1 - only one group
    op = QubitOperator("[X0] + [Y1]")
    assert partitioning.get_si_sets(op) == [ [ ((0, 'X'),), ((1, 'Y'),),] ]

    # X0 + Y0 - two groups
    op = QubitOperator("[X0] + [Y0]")
    assert partitioning.get_si_sets(op) == [[((0, 'X'),)], [((0, 'Y'),)] ]

    # One group
    op = QubitOperator("[X0 Y1 Z2] + [X1 X2 X3]") # these do fully commute
    gold = [ [ ((0, 'X'), (1, 'Y'), (2, 'Z')), ((1, 'X'), (2, 'X'), (3, 'X')), ] ]
    res = partitioning.get_si_sets(op)
    assert gold==res

    # Two groups
    op = QubitOperator("[X0 Y1 X2] + [X1 X2 X3]") # these do not commute
    gold = [ [ ((0, 'X'), (1, 'Y'), (2, 'X')), ],
             [ ((1, 'X'), (2, 'X'), (3, 'X')), ] ]
    res = partitioning.get_si_sets(op)
    assert gold==res


def test_graphbased_groupings():
    """Test of graph-based groupings are reasonable"""
    pass


def test_blocked_noclid_residual():

    qop = QubitOperator

    block_size = 3
    # 3 windows (be sure to test *periodicity*)
    #               ***---*
    a = util.s2p("XXXIIII",'notnumbered') # first fragment
    b = util.s2p("IYIIYII",'notnumbered') # residual
    c = util.s2p("IIIIYYI",'notnumbered') # first fragment
    d = util.s2p("IIIZZZI",'notnumbered') # first fragment
    e = util.s2p("IXXXIII",'notnumbered') # second fragment
    f = util.s2p("IIZZZII",'notnumbered') # third fragment
    g = util.s2p("ZIIIIIZ",'notnumbered') # residual (not doing periodic)
    h = util.s2p("IIZZZZZ",'notnumbered') # residual
    op = qop(a) + qop(b) + qop(c) + qop(d) + qop(e) + qop(f) + qop(g) + 0.5*qop(h)
    gold_parts = [[a,c,d], [e,], [f,],      [b,g,], [h,]]
    res_parts = partitioning.get_sets_blocked_noclid_residual(op, block_size)
    # print()
    # pprint.pprint(gold_parts)
    # pprint.pprint(res_parts)
    assert gold_parts==res_parts, (gold_parts,res_parts)


# def test_blocked_fald_with_qubit_ids():

#     qop = QubitOperator

#     block_size = 3
#     # 3 windows (be sure to test *periodicity*)
#     #               ***---*
#     a = util.s2p("XXXIIII",'notnumbered') # first fragment
#     b = util.s2p("IYIIYII",'notnumbered') # residual
#     c = util.s2p("IIIIYYI",'notnumbered') # first fragment
#     d = util.s2p("IIIZZZI",'notnumbered') # first fragment
#     e = util.s2p("IXXXIII",'notnumbered') # second fragment
#     f = util.s2p("IIZZZII",'notnumbered') # third fragment
#     g = util.s2p("ZIIIIIZ",'notnumbered') # residual (not doing periodic)
#     h = util.s2p("IIZZZZZ",'notnumbered') # residual
#     op = qop(a) + qop(b) + qop(c) + qop(d) + qop(e) + qop(f) + qop(g) + 0.5*qop(h)
#     gold_parts = [[a,c,d], [e,], [f,],      [b,g,], [h,]]
#     res_parts = partitioning.get_sets_blocked_ttld_residual(op, block_size)
#     # print()
#     # pprint.pprint(gold_parts)
#     # pprint.pprint(res_parts)
#     assert gold_parts==res_parts, (gold_parts,res_parts)



def test_tensor_trains():

    # Test adding to an existing ttrain
    k=2 # but there is only 1 free qubit in this example
    ttrain1 = partitioning.tensor_train(k)
    p = util.s2p("XZZZ",'notnumbered')
    res = ttrain1.attempt_add_term(p)
    assert set()==res
    p = util.s2p("YZZZ",'notnumbered') # mismatched 0
    res = ttrain1.attempt_add_term(p)
    assert {0,}==res
    p = util.s2p("IIIZ",'notnumbered') # mismatched 0,1,2
    res = ttrain1.attempt_add_term(p)
    assert False==res

    # Test adding to an existing ttrain
    k=2
    ttrain1 = partitioning.tensor_train(k)
    p = util.s2p("XZZZ",'notnumbered')
    res = ttrain1.attempt_add_term(p)
    assert set()==res
    p = util.s2p("YYZZ",'notnumbered') # mismatched 0,1
    res = ttrain1.attempt_add_term(p)
    assert {0,1}==res
    p = util.s2p("IIIZ",'notnumbered') # mismatched 0,1,2
    res = ttrain1.attempt_add_term(p)
    assert False==res

    # # Test the qwc check
    # # Test adding to an existing ttrain
    # k=2
    # ttrain1 = partitioning.tensor_train(k)
    # p = util.s2p("XZZZ",'notnumbered')
    # ttrain1.attempt_add_term(p)
    # p = util.s2p("YYZZ",'notnumbered') 
    # ttrain1.attempt_add_term(p)
    # p = util.s2p("IIIZ",'notnumbered') 
    # res = ttrain1.check_qwc_with_all_terms(p)
    # assert True==res
    # p = util.s2p("IZIZ",'notnumbered') 
    # res = ttrain1.check_qwc_with_all_terms(p)
    # assert False==res


def test_partition_of_ttrains():

    # Test adding to an existing partition_of_ttrains
    k=2
    ptt = partitioning.partition_of_ttrains(k)
    p = util.s2p("XZZZ",'notnumbered')
    res = ptt.attempt_add_term(p)
    assert set()==res
    p = util.s2p("YYZZ",'notnumbered') 
    res = ptt.attempt_add_term(p)
    assert {0,1}==res
    p = util.s2p("IIIZ",'notnumbered') 
    res = ptt.attempt_add_term(p)
    assert False==res
    p = util.s2p("IZIZ",'notnumbered') 
    res = ptt.attempt_add_term(p)
    assert False==res


    # Test adding to an existing partition_of_ttrains
    k=2
    ptt = partitioning.partition_of_ttrains(k)
    p = util.s2p("YZII",'notnumbered')
    res = ptt.attempt_add_term(p)
    assert set()==res # First TT
    p = util.s2p("XXII",'notnumbered') 
    res = ptt.attempt_add_term(p)
    assert {0,1}==res # First TT
    p = util.s2p("IIYY",'notnumbered') 
    res = ptt.attempt_add_term(p)
    assert set()==res # Second TT (because it QWCs with previous TT)
    p = util.s2p("IXII",'notnumbered') 
    res = ptt.attempt_add_term(p)
    assert {0,1}==res # First TT
    p = util.s2p("IXYI",'notnumbered') 
    res = ptt.attempt_add_term(p)
    assert False==res # Rejected from both existing TTs, and cannot form new one
    # These are the two tensor trains in this partition:
    # {YZII, XXII, IXII} and {IIYY}
    assert [((0,'Y'),(1,'Z')), ((0,'X'),(1,'X')), ((1,'X'),)] == ptt.tensor_trains[0].tensor_train_qubops
    assert [((2,'Y'),(3,'Y'))] == ptt.tensor_trains[1].tensor_train_qubops
    
    # *** Test get_pstrings()
    # Test getting pauli terms. Needed for using shotcounts.get_shotcounts() and shotcounts.get_Rhat().
    pstrings = ptt.get_pstrings()
    assert [ ((0,'Y'),(1,'Z')), ((0,'X'),(1,'X')), ((1,'X'),), ((2,'Y'),(3,'Y')) ] == pstrings
    

def test_get_fald_partitions():

    k=2

    # Easier to see relations between strings when I build operator this way
    # Coefficients there to make sure FALD-SI orders them, so I know result
    op =  QubitOperator()
    op += QubitOperator( util.s2p("YZII",'notnumbered') , 7 ) # First TT
    op += QubitOperator( util.s2p("XXII",'notnumbered') , 6 ) # First TT
    op += QubitOperator( util.s2p("IIYY",'notnumbered') , 5 ) # Second TT (because it QWCs with previous TT)
    op += QubitOperator( util.s2p("IXII",'notnumbered') , 4 ) # First TT
    op += QubitOperator( util.s2p("IXYI",'notnumbered') , 3 ) # Second partition

    ptt_set = partitioning.set_of_ttrain_partitions(op,k)

    # print("printing ptt_set props:")
    # print(len(ptt_set.partitions))
    # print(ptt_set.partitions)

    # [Partition of tensor trains. 2 tensor trains with k=2.
    # [((0, 'Y'), (1, 'Z')), ((0, 'X'), (1, 'X')), ((1, 'X'),)]
    # [((2, 'Y'), (3, 'Y'))], Partition of tensor trains. 1 tensor trains with k=2.
    # [((1, 'X'), (2, 'Y'))]]
    
    assert [((0,'Y'),(1,'Z')), ((0,'X'),(1,'X')), ((1,'X'),)] == ptt_set.partitions[0].tensor_trains[0].tensor_train_qubops # 1st partition 1st TT
    assert [((2,'Y'),(3,'Y'))] == ptt_set.partitions[0].tensor_trains[1].tensor_train_qubops # 1st partition 2nd TT
    assert [((1,'X'),(2,'Y'))] == ptt_set.partitions[1].tensor_trains[0].tensor_train_qubops # 2nd partition 1st TT



    # # These are the two partitions:
    # # ({YZII, XXII, IXII} & {IIYY}) and IXYI
    # assert [((0, 'Y'), (1, 'Z')), ((0, 'X'), (1, 'X')), ((1, 'X'),)] == ptt.tensor_trains[0].tensor_train_qubops
    # assert [((2, 'Y'), (3, 'Y'))] == ptt.tensor_trains[1].tensor_train_qubops
















