# Relations between Pauli strings, commutation graphs, etc.
# 

from openfermion import QubitOperator
import networkx as nx




def s2p(inpstr, input_type="numbered"):
    '''Convert string to pauli term of tuples.
    
    Examples:
    ('numbered'):    "Z1 Z2 Z3 Z4" -> [(1,'Z'),(2,'Z'),(3,'Z'),(4,'Z')]
    ('notnumbered'): "ZZZZ"        -> [(1,'Z'),(2,'Z'),(3,'Z'),(4,'Z')]
    '''

    assert input_type in ["numbered","notnumbered"]

    if input_type=="numbered":
        spl = inpstr.split(' ')
        return tuple([(int(pq[1:]),pq[:1]) for pq in spl])
    elif input_type=="notnumbered":
        return tuple([(i,pq) for i,pq in enumerate(inpstr)])
    
class singlePauliTerm:
    '''Class for single Pauli term'''
    def __init__(self, pstring, coeff=1., input_type="numbered"):
        if isinstance(pstring, QubitOperator):
            assert len(pstring.terms) == 1
            self.pstring = next(iter(pstring.terms))
            self.coeff = pstring.terms[self.pstring]
        if isinstance(pstring, str):
            self.pstring = s2p(pstring, input_type)
        if isinstance(pstring, tuple):
            self.pstring = pstring

        self.coeff *= coeff



def pstrings_qubitwise_commute(Pi, Pj):
    """Return True if qubit-wise commute, False otherwise"""

    # Convert to raw tuples if necessary
    if isinstance(Pi, QubitOperator):
        assert len(Pi.terms) <= 1
        Pi = next(iter(Pi.terms))
    if isinstance(Pj, QubitOperator):
        assert len(Pj.terms) <= 1
        Pj = next(iter(Pj.terms))

    # Get qubits in common
    qubs_i = set([p[0] for p in Pi])
    qubs_j = set([p[0] for p in Pj])
    qubs_in_common = qubs_i.intersection(qubs_j)
    dictPi = dict(Pi)
    dictPj = dict(Pj)

    # Find out if any local Paulis do not commute
    qubit_wise_commute = True
    for qub in qubs_in_common:

        if dictPi[qub] != dictPj[qub]:
            return False

    return True


def pstrings_commute(Pi, Pj):
    """Return True if ("fully") commute, False otherwise"""

    # Convert to raw tuples if necessary
    if isinstance(Pi, QubitOperator):
        assert len(Pi.terms) <= 1
        Pi = next(iter(Pi.terms))
    if isinstance(Pj, QubitOperator):
        assert len(Pj.terms) <= 1
        Pj = next(iter(Pj.terms))

    # Get qubits in common
    qubs_i = set([p[0] for p in Pi])
    qubs_j = set([p[0] for p in Pj])
    qubs_in_common = qubs_i.intersection(qubs_j)
    dictPi = dict(Pi)
    dictPj = dict(Pj)

    # Find out if any local Paulis do not commute
    qubit_wise_commute = True
    for qub in qubs_in_common:

        if dictPi[qub] != dictPj[qub]:
            # Every time local Paulis don't commute,
            # it negates the full commuation.
            # An even number denotes full commutation.
            qubit_wise_commute = not qubit_wise_commute

    return qubit_wise_commute


def get_relationship_graph(qubop,commute_func,**kwargs):
    '''Return graph with an edge if function returns True (e.g. if commute)
    
    (Note: Do we need a *default ordering*?)

    Args:
        qubop - QubitOperator 
        commute_func - \in ("qubitwise","fullcommute"), or function taking Pi,Pj & returning boolean
        kwargs - required for e.g. k-qubitwise commuting

    Returns:
        networkx.Graph where each is a different QubitOperators

    '''
    
    if isinstance(commute_func,str):
        if commute_func=="qubitwise":
            commute_func = pstrings_qubitwise_commute
        elif commute_func=="fullcommute":
            commute_func = pstrings_commute


    G = nx.Graph()
    G.add_nodes_from(qubop.terms)
    for Pi in G:
        for Pj in G:
            
            # To make it a triangular adjacency matrix
            if Pi <= Pj:
                continue

            relation_bool = commute_func(Pi,Pj)

            if relation_bool:
                G.add_edge(Pi,Pj)


            # qubs_i = set([p[0] for p in Pi])
            # qubs_j = set([p[0] for p in Pj])
            # qubs_in_common = qubs_i.intersection(qubs_j)
            # dictPi = dict(Pi)
            # dictPj = dict(Pj)
            
            # qubit_wise_commute = True
            # for qub in qubs_in_common:
                
            #     if dictPi[qub]!=dictPj[qub]:
            #         qubit_wise_commute = False
            #         break
            
            # if qubit_wise_commute:
            #     G.add_edge(Pi,Pj)
            
    return G


def get_mismatched_qubits(Pi, Pj):
    """Return set of qubits where Pauli strings differ.
    The convention (required for FALD) is that e.g. 'I' and 'X' are mismatched.
    
    Args:
        Pi,Pj: either QubitOperator or tuple of tuples.
        
    Returns:
        set of qubits where Pauli strings differ"""

    # Convert to raw tuples if necessary
    if isinstance(Pi, QubitOperator):
        assert len(Pi.terms) <= 1
        Pi = next(iter(Pi.terms))
    if isinstance(Pj, QubitOperator):
        assert len(Pj.terms) <= 1
        Pj = next(iter(Pj.terms))

    # Get qubits in common
    qubs_i = set([p[0] for p in Pi])
    qubs_j = set([p[0] for p in Pj])
    qubs_in_common = qubs_i.intersection(qubs_j)
    dictPi = dict(Pi)
    dictPj = dict(Pj)
    # Qubits not in common
    qubs_not_matching = qubs_i.union(qubs_j) - qubs_in_common

    # Find out if any local Paulis do not commute
    mismatched_qubits = set()
    for qub in qubs_in_common:

        if dictPi[qub] != dictPj[qub]:
            mismatched_qubits.add(qub)

    mismatched_qubits = mismatched_qubits.union(qubs_not_matching)

    return mismatched_qubits


def pstrings_are_disjoint(Pi,Pj):
    """Return True if Pauli strings are disjoint, False otherwise"""

    # Convert to raw tuples if necessary
    if isinstance(Pi, QubitOperator):
        assert len(Pi.terms) <= 1
        Pi = next(iter(Pi.terms))
    if isinstance(Pj, QubitOperator):
        assert len(Pj.terms) <= 1
        Pj = next(iter(Pj.terms))

    # Get qubits in common
    qubs_i = set([p[0] for p in Pi])
    qubs_j = set([p[0] for p in Pj])
    qubs_in_common = qubs_i.intersection(qubs_j)

    return len(qubs_in_common) == 0








