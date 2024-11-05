# Tests for paulirelat.py

# import pytest

from . import paulirelat
from .util import s2p

from openfermion import QubitOperator
import networkx as nx
from pprint import pprint



def test_qwc_pstringpair():
    """Test QWC on pairs of Pauli strings

    Note it is e.g. ((0, 'X'), (1, 'Y'), (2, 'Z'))
    """

    A = QubitOperator.identity()
    B = QubitOperator.identity()
    Araw = next(iter(A.terms))
    Braw = next(iter(B.terms))
    assert paulirelat.pstrings_qubitwise_commute(A,B) == True

    A = QubitOperator("X0")
    B = QubitOperator.identity()
    Araw = next(iter(A.terms))
    Braw = next(iter(B.terms))
    assert paulirelat.pstrings_qubitwise_commute(A,B) == True

    A = QubitOperator("X0")
    B = QubitOperator("Y0")
    Araw = next(iter(A.terms))
    Braw = next(iter(B.terms))
    assert paulirelat.pstrings_qubitwise_commute(A,B) == False

    A = QubitOperator("X0")
    B = QubitOperator("Y1")
    Araw = next(iter(A.terms))
    Braw = next(iter(B.terms))
    assert paulirelat.pstrings_qubitwise_commute(A,B) == True

    A = QubitOperator("X0 Y1 Z2")
    B = QubitOperator("   Y1")
    Araw = next(iter(A.terms))
    Braw = next(iter(B.terms))
    assert paulirelat.pstrings_qubitwise_commute(A,B) == True

    A = QubitOperator("X0 Z1 Z2")
    B = QubitOperator("   Y1")
    Araw = next(iter(A.terms))
    Braw = next(iter(B.terms))
    assert paulirelat.pstrings_qubitwise_commute(A,B) == False



def test_full_commute_pstringpair():
    """Test full commutation on pairs of Pauli strings"""

    A = QubitOperator.identity()
    B = QubitOperator.identity()
    Araw = next(iter(A.terms))
    Braw = next(iter(B.terms))
    assert paulirelat.pstrings_commute(A,B) == True

    A = QubitOperator("X0")
    B = QubitOperator.identity()
    Araw = next(iter(A.terms))
    Braw = next(iter(B.terms))
    assert paulirelat.pstrings_commute(A,B) == True

    A = QubitOperator("X0")
    B = QubitOperator("Y0")
    Araw = next(iter(A.terms))
    Braw = next(iter(B.terms))
    assert paulirelat.pstrings_commute(A,B) == False

    A = QubitOperator("X0")
    B = QubitOperator("Y1")
    Araw = next(iter(A.terms))
    Braw = next(iter(B.terms))
    assert paulirelat.pstrings_commute(A,B) == True

    A = QubitOperator("X0 Y1 Z2")
    B = QubitOperator("   Y1")
    Araw = next(iter(A.terms))
    Braw = next(iter(B.terms))
    assert paulirelat.pstrings_commute(A,B) == True

    A = QubitOperator("X0 Z1 Z2")
    B = QubitOperator("   Y1 X2") # even num mismatches
    Araw = next(iter(A.terms))
    Braw = next(iter(B.terms))
    assert paulirelat.pstrings_commute(A,B) == True 

    A = QubitOperator("X0 Z1 Z2")
    B = QubitOperator("   Z1 X2") # odd num mismatches
    Araw = next(iter(A.terms))
    Braw = next(iter(B.terms))
    assert paulirelat.pstrings_commute(A,B) == False 

    A = QubitOperator("X0 Z1 Z2 X3 Z4 Z5 Z6")
    B = QubitOperator("   Z1 X2 Y3 Z4 Z5 Z6") # even num mismatches
    Araw = next(iter(A.terms))
    Braw = next(iter(B.terms))
    assert paulirelat.pstrings_commute(A,B) == True 

    A = QubitOperator("X0 Z1 Z2 X3 Z4 Z5 Z6")
    B = QubitOperator("   Z1 X2 Y3 Y4 Z5 Z6") # odd num mismatches
    Araw = next(iter(A.terms))
    Braw = next(iter(B.terms))
    assert paulirelat.pstrings_commute(A,B) == False 

def test_k_fc():
    '''Test k-qubits full commuting'''

    pass


def test_qwc_graph():
    '''Test creation of qubitwise commuting graph'''

    # ***
    # *** Simple 2-qubit example ***
    op = QubitOperator("[X0] + [Y0] + [Z0] + [X0 Y1]")
    # . 0 0 1
    #   . 0 0 
    #     . 0
    #       .

    # Only one edge, because only two terms are non-QWC
    gold = nx.Graph()
    gold.add_nodes_from( op.terms )
    gold.add_edge( ((0,'X'),) , ((0,'X'),(1,'Y'),) )

    # Code's graph result
    G = paulirelat.get_relationship_graph(op, "qubitwise")
    assert gold.edges() == G.edges()

    # ***
    # *** Example from Izmaylov QWC [arXiv:1907.03358] (*not* "alphabetical") ***
    op = QubitOperator("[Z1 Z2 Z3 Z4] + [Z1 Z2 Z3] + [Z1 Z2] + [Z1]")
    op += QubitOperator("[X3 X4] + [Y1 Y2 X3 X4] + [Y1 X3 X4]")

    # Gold graph (as in, ground-truth)
    gold = nx.Graph()
    gold.add_nodes_from( op.terms )
    gold.add_edge( s2p("Z1 Z2 Z3 Z4") , s2p("Z1 Z2 Z3") )
    gold.add_edge( s2p("Z1 Z2 Z3 Z4") , s2p("Z1") )
    gold.add_edge( s2p("Z1 Z2 Z3 Z4") , s2p("Z1 Z2") )
    gold.add_edge( s2p("Z1 Z2 Z3") , s2p("Z1") )
    gold.add_edge( s2p("Z1 Z2 Z3") , s2p("Z1 Z2") )
    gold.add_edge( s2p("Z1") , s2p("Z1 Z2") )
    gold.add_edge( s2p("Z1") , s2p("X3 X4") )
    gold.add_edge( s2p("Z1 Z2") , s2p("X3 X4") )
    gold.add_edge( s2p("X3 X4") , s2p("Y1 Y2 X3 X4") )
    gold.add_edge( s2p("X3 X4") , s2p("Y1 X3 X4") )
    gold.add_edge( s2p("Y1 X3 X4") , s2p("Y1 Y2 X3 X4") )
    # pprint(list(gold.edges))

    # Code's result for graph
    G = paulirelat.get_relationship_graph(op, "qubitwise")   
    # print()
    # pprint(list(G.edges))
    # print()
    # pprint(list( nx.complement(G).edges ) )
    # #print(list( G.complement().edges ) )
    # Assert graphs are the same
    assert gold.edges() == G.edges()
    
    # # Coloring
    # colored_sets = qwc.get_qwc_groups(op)
    # print("Coloring of 5-qubit Izmaylov example:")
    # pprint(colored_sets)    



def test_full_commute_graph():
    '''Test creation of full commuting graph'''

    # ***
    # *** Simple 2-qubit example ***
    op = QubitOperator("[X0 X1] + [Y0 Y1] + [X0]")
    # openfermion should reorder this to X0, X0X1, Y0Y1
    # {X0, X0X1} - True
    # {X0, Y0Y1} - False
    # {X0X1, Y0Y1} - True
    # . 1 0 
    #   . 1  

    # Only one edge, because only two terms are non-QWC
    gold = nx.Graph()
    gold.add_nodes_from( op.terms )
    gold.add_edge( ((0,'X'),) , ((0,'X'),(1,'X'),) )
    gold.add_edge( ((0,'X'),(1,'X'),) , ((0,'Y'),(1,'Y'),) )

    # Code's graph result
    G = paulirelat.get_relationship_graph(op, "fullcommute")
    assert gold.edges() == G.edges()

def test_mismatched_qubits():
    """Test get_mismatched_qubits()."""

    pi = ((0, 'X'), (1, 'Y'), (2, 'Z'),)
    pj = (          (1, 'X'), (2, 'Z'),(3, 'Z'),)
    gold = {0,1,3}
    res = paulirelat.get_mismatched_qubits(pi,pj)
    assert gold==res, (gold,res)

def test_disjoint():
    """Test pstrings_are_disjoint()"""

    pi = ((0, 'X'), (1, 'Y'), (2, 'Z'),)
    pj = (          (1, 'X'), (2, 'Z'),(3, 'Z'),)
    assert paulirelat.pstrings_are_disjoint(pi,pj) == False

    pi = ((0, 'X'), (1, 'Y'), (2, 'Z'),)
    pj = (                              (3, 'X'), (4, 'Z'),(5, 'Z'),)
    assert paulirelat.pstrings_are_disjoint(pi,pj) == True

    pi = ((0, 'X'), (1, 'Y'), (2, 'Z'),)
    pj = (          (1, 'X'), (2, 'Z'),(3, 'Z'),)
    assert paulirelat.pstrings_are_disjoint(pi,pj) == False

    pi = ((0, 'X'), (1, 'Y'), (2, 'Z'),)
    pj = ((0, 'X'), (1, 'Y'), (2, 'Z'),)
    assert paulirelat.pstrings_are_disjoint(pi,pj) == False

    pi = QubitOperator([(0, 'X'), (1, 'Y'), (2, 'Z'),])
    pj = QubitOperator([(0, 'X'), (1, 'Y'),             (3, 'Z'),])
    assert paulirelat.pstrings_are_disjoint(pi,pj) == False

















