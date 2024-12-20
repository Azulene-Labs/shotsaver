# shotcounts.py

import numpy as np
from openfermion import QubitOperator,count_qubits
from scipy.sparse import issparse
import warnings

from .util import get_variance,get_suboperator



def get_shotcounts_from_opsum(ops_to_sum,psi,epsilon,nq=None):
    """

    This allows for "overlapping" partitions, where the same Pauli string
    belongs to multiple partitions.

    Args:
        ops_to_sum (list of {QubitOperators or numpy matrix or sparse matrix}): the
            Hamiltonian = the sum of the list's members
        psi (numpy vector): state
        epsilon (float): desired error

    Returns:
        shot counts (float)
    """

    if nq is None:
        nq = int( np.log2(len(psi)) )

    temp = 0
    for op in ops_to_sum:
        # op can be QubitOperator or numpy matrix or sparse matrix
        var = get_variance(op,psi,nq)

        temp += np.sqrt(var)

    temp = temp**2
    shotcounts = temp.real / epsilon**2

    return shotcounts.real


def get_shotcounts_nonoverlapping_parts(op,partitions,psi,epsilon,nq=None):
    """Return shot counts, given desired error. Operator must be QubitOperator.
    See get_shotcounts_from_opsum() for more flexible inputs.
    
    Args:
        op (QubitOperator): Operator
        partitions (list): list of lists of Pauli strings
        psi (numpy vector): state
        epsilon (float): desired error
        nq (int): number of qubits
        
    Returns:
        shot counts (float)"""

    assert isinstance(op,QubitOperator)

    if nq is None:
        nq = int(np.log2(len(psi)))
    
    shotcounts = 0

    # Need to create a sub-operator for each partition
    for pstringset in partitions:
        subop = get_suboperator(op,pstringset)
        var = get_variance(subop,psi,nq)
        shotcounts += np.sqrt(var)
    
    shotcounts = shotcounts**2

    if np.abs(shotcounts.imag)/np.abs(shotcounts.real) > 1e-10:
        warnings.warn(f"Shot counts appears complex. shotcounts*eps**2 = ({shotcounts.real},{shotcounts.imag})", UserWarning)

    shotcounts = shotcounts.real / epsilon**2

    return shotcounts.real


def get_shotcount_lowerbound(qubop,psi,eps):
    """Return show count lower bound. Corresponds to case of diagonalizing entire Hamiltonian.
    
    Args:
        qubob (QubitOperator): Operator
        psi (numpy vector): state
        eps (float): desired error
        
    Returns:
        shot count lower bound (float)"""
    
    nq = round( np.log2(len(psi)) )
    assert len(psi)==2**nq, (len(psi),nq)
    variance = get_variance(qubop,psi,nq)
    shots_lower_bound = variance/eps**2

    return shots_lower_bound
    

def get_error_given_shotcounts():    
    """Return error, given total shot counts"""

    raise NotImplementedError()


def get_Rhat_from_opsum(fullop, ops_to_sum, remove_identity=True, validity_check=True):
    """Return Rhat quantity (eq20 arxiv:1908.06942) given operator sum.
    
    Args:
        fullop (QubitOperator): Full operator
        ops_to_sum (list): list of QubitOperators to sum. This sum should equal fullop.
        remove_identity (bool): remove identity terms from Rhat calculation
        validity_check (bool): check if ops_to_sum sum to fullop

    Returns:
        Rhat (float)
    """

    if validity_check:
        if not fullop == sum(ops_to_sum):
            raise ValueError("ops_to_sum must sum to fullop")

    numerator = 0
    for pstr,coeff in fullop.terms.items():
        if pstr==() and remove_identity:
            continue
        numerator += np.abs(coeff)

    denominator = 0
    for op in ops_to_sum:
        if not isinstance(op,QubitOperator):
            raise ValueError("Input must be QubitOperator")
        
        sum_coeff_sq_i = 0
        for pstr,coeff in op.terms.items():
            if pstr==() and remove_identity:
                continue
            a = op.terms[pstr]
            sum_coeff_sq_i += np.abs(a)**2
        
        denominator += np.sqrt(sum_coeff_sq_i)
    
    Rhat = (numerator/denominator)**2
    return Rhat
    


def get_Rhat_nonoverlapping_parts(qubop,partitions,remove_identity=True):
    """Rhat quantity (eq20 arxiv:1908.06942)
    
    Args:
        qubop (QubitOperator): QubitOperator
        p_grouping (list): list of lists of Pauli strings
        remove_identity (bool): remove identity terms from Rhat calculation

    Returns:
        Rhat (float)
    """

    total_terms = len(qubop.terms)
    terms_processed = 0

    numerator = 0
    denominator = 0
    
    for set_j in partitions:
        
        sum_coeff_sq_i = 0
        
        for i,Pi in enumerate(set_j):

            if Pi==() and remove_identity:
                total_terms -= 1
                continue
            
            a = qubop.terms[Pi]
            numerator += np.abs( a )
            sum_coeff_sq_i += np.abs( a )**2
            terms_processed += 1
        
        denominator += np.sqrt(sum_coeff_sq_i)
    
    if terms_processed<(total_terms):
        # Above is '-1' because we don't count the identity term
        warnings.warn(f"WARNING: Terms processed ({terms_processed}) is less than total terms ({total_terms})", UserWarning)

    Rhat = (numerator/denominator)**2
    return Rhat
    

def get_Mghat_from_opsum(fullop, ops_to_sum, remove_identity=True, validity_check=True):
    """Return Mghat quantity (denominator of eq20 arxiv:1908.06942) given operator sum.
    
    Args:
        fullop (QubitOperator): Full operator
        ops_to_sum (list): list of QubitOperators to sum. This sum should equal fullop.
        remove_identity (bool): remove identity terms from Mghat calculation
        validity_check (bool): check if ops_to_sum sum to fullop

    Returns:
        Mghat (float)
    """

    if validity_check:
        if not fullop == sum(ops_to_sum):
            raise ValueError("ops_to_sum must sum to fullop")

    # numerator = 0
    # for pstr,coeff in fullop.terms.items():
    #     if pstr==() and remove_identity:
    #         continue
    #     numerator += np.abs(coeff)

    denominator = 0
    for op in ops_to_sum:
        if not isinstance(op,QubitOperator):
            raise ValueError("Input must be QubitOperator")
        
        sum_coeff_sq_i = 0
        for pstr,coeff in op.terms.items():
            if pstr==() and remove_identity:
                continue
            a = op.terms[pstr]
            sum_coeff_sq_i += np.abs(a)**2
        
        denominator += np.sqrt(sum_coeff_sq_i)
    
    # Rhat = (numerator/denominator)**2
    # return Rhat

    return denominator**2
    


def get_Mghat_nonoverlapping_parts(qubop,partitions,remove_identity=True):
    """Return Mghat quantity (denominator of eq20 arxiv:1908.06942)
    
    Args:
        qubop (QubitOperator): QubitOperator
        p_grouping (list): list of lists of Pauli strings
        remove_identity (bool): remove identity terms from Mghat calculation

    Returns:
        Mghat (float)
    """

    total_terms = len(qubop.terms)
    terms_processed = 0

    # numerator = 0
    denominator = 0
    
    for set_j in partitions:
        
        sum_coeff_sq_i = 0
        
        for i,Pi in enumerate(set_j):

            if Pi==() and remove_identity:
                total_terms -= 1
                continue
            
            a = qubop.terms[Pi]
            # numerator += np.abs( a )
            sum_coeff_sq_i += np.abs( a )**2
            terms_processed += 1
        
        denominator += np.sqrt(sum_coeff_sq_i)
    
    if terms_processed<(total_terms):
        # Above is '-1' because we don't count the identity term
        warnings.warn(f"WARNING: Terms processed ({terms_processed}) is less than total terms ({total_terms})", UserWarning)

    # Rhat = (numerator/denominator)**2
    # return Rhat

    return denominator**2































