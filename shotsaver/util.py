# Utility functions for quantum measurement package
# usage: pytest -s ...

import numpy as np
from scipy.sparse import issparse
from openfermion import QubitOperator, InteractionOperator
from openfermion import get_sparse_operator, count_qubits

paulis = {}
paulis['I'] = np.eye(2, dtype=complex)
paulis['X'] = np.array([[0, 1], [1, 0]], dtype=complex)
paulis['Y'] = np.array([[0, -1.j], [1.j, 0]], dtype=complex)
paulis['Z'] = np.array([[1. + 0j, 0], [0, -1]], dtype=complex)



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
        pterms = []
        for i,p in enumerate(inpstr):
            if p in ['X','Y','Z']:
                pterms.append((i,p))
            elif p=='I':
                pass
            else:
                raise ValueError(f"Invalid Pauli char ({p})")
            
        return tuple(pterms)
    

def get_suboperator(op,pstrings):
    '''Gets suboperator from set of pstrings'''

    sub_op = QubitOperator()

    for p in pstrings:
        if p in op.terms:
            sub_op += QubitOperator(p,op.terms[p])
        else:
            raise ValueError(f"pstring ({p}) not in op.terms")

    return sub_op


def get_variance(qubop,psi,nq=None):
    '''Returns variance <O^2> - <O>^2 w.r.t. given state'''

    # Ensure is instance
    assert isinstance(qubop, (QubitOperator,np.ndarray) ) or issparse(qubop)

    if isinstance(qubop,QubitOperator):
        if nq is None:
            nq = count_qubits(qubop)

        op = get_sparse_operator(qubop, nq)
        opsq = get_sparse_operator(qubop * qubop, nq)

    elif isinstance(qubop,(np.ndarray)) or issparse(qubop):
        opsq = qubop @ qubop
        op   = qubop

    mean        = psi.conj().T @ (op @ psi)
    mean_sq     = mean**2
    opsq_expect = psi.conj().T @ (opsq @ psi)

    return opsq_expect - mean_sq


def coloring_to_grouping(coloring_dict):
    """Takes {P_i:color_id} to actual groupings of terms"""
    
    # Largest value, then create list
    numcolors = 1 + max( coloring_dict.values() )
    the_sets = []
    for i in range(numcolors):
        the_sets.append([])
    
    for key,val in coloring_dict.items():
        the_sets[val].append(key)
    
    return the_sets


def paulistr2simplestr(paulistr,nq=None):
    """Convert pauli string to simple string
    
    Example: [(2,Z), (3,Z)] --> 'IIZZ' """
    pass
    
    # Assert op is QubitOperator or list of tuples
    assert isinstance(paulistr,QubitOperator) or isinstance(paulistr,list)
    
    if nq is None:
        # Get max integer in the list of tuples
        nq = 1 + max([p[0] for p in paulistr])
    else:
        assert nq >= 1 + max([p[0] for p in paulistr])

    # Turn list of tuples e.g. [(2,Z), (3,Z)] into dict
    terms_dict = {key: value for key, value in paulistr}
    
    simplestr = ""

    for i in range(nq):
        if i not in terms_dict:
            simplestr += "I"
        else:
            simplestr += terms_dict[i]
    
    return simplestr



def get_rank2_mask(n, k):
    """
    Create an n x n mask with 0s on the (k-1)-th diagonals and 1s elsewhere.
    
    Parameters:
    n (int): Size of the matrix (n x n).
    k (int): The number of steps away from the main diagonal to set to 0.
    
    Returns:
    numpy.ndarray: An n x n mask matrix.
    """

    mask = np.ones((n, n))
    K = k-1
    
    for i in range(-K, K + 1):
        np.fill_diagonal(mask[max(0, i):, max(0, -i):], 0)
        
    return mask



def get_rank4_mask(n, k):
    """
    Create an n x n x n x n mask with 0s on the (k-1)-th diagonals and 1s elsewhere.
    
    Parameters:
    n (int): Size of each dimension of the tensor (n x n x n x n).
    k (int): The number of steps away from the main diagonal to set to 0.
    
    Returns:
    numpy.ndarray: An n x n x n x n mask tensor.
    """

    K = k-1

    mask = np.ones((n, n, n, n))
    
    # for i in range(-K, K + 1):
    for i in range(n):
        for j in range(-K, K + 1):
            for l in range(-K, K + 1):
                for m in range(-K, K + 1):
                    # Create a diagonal with the given offsets
                    # idx = np.arange(max(0, i), n + min(0, i))

                    # print('*********')
                    # print(i,j,l,m)
                    # print(idx)
                    # print('*********',flush=True)

                    # mask[idx, idx - i, idx - j, idx - l] = 0
                    J = i + j
                    L = i + l
                    M = i + m

                    # We want the "spread" to be at most 'k'
                    if J>=0 and J<n and L>=0 and L<n and M>=0 and M<n:
                        if (max(i,J,L,M) - min(i,J,L,M) + 1) <= k:
                            mask[i,J,L,M] = 0

                    # if J>=0 and J<n and L>=0 and L<n and M>=0 and M<n:
                    #     mask[i,J,L,M] = 0
    
    return mask

def get_rank2_distance_mask(n):
    """Return matrix with 0's on diag, and Manhattan distance-from-diag elsewhere"""

    # Create an n x n matrix of zeros
    matrix = np.zeros((n, n), dtype=int)
    
    # Fill the off-diagonals
    for k in range(1, n):
        np.fill_diagonal(matrix[:, k:], k)
        np.fill_diagonal(matrix[k:, :], k)
    
    return matrix

def get_rank4_distance_mask(n):
    """Return tensor with 0's on diag, and max(i,j,k,l)-min(i,j,k,l) elsewhere"""

    # Create an n x n x n x n tensor of zeros
    tensor = np.zeros((n, n, n, n))
    
    # Fill the off-diagonals
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    tensor[i, j, k, l] = max(i, j, k, l) - min(i, j, k, l)
    
    return tensor


def permute_iop_tensors(iop, permutation):
    """
    Permute the one- and two-body tensors of an InteractionOperator.

    Parameters:
    iop (InteractionOperator): The InteractionOperator.
    permutation (list): The permutation to apply to the tensors.

    Returns:
    InteractionOperator: The InteractionOperator with permuted tensors.
    """

    # Apply the permutation
    permuted_t2 = iop.one_body_tensor[np.ix_(permutation, permutation)]
    permuted_t4 = iop.two_body_tensor[np.ix_(permutation, permutation, permutation, permutation)]
    permuted_iop = InteractionOperator(iop.constant, permuted_t2, permuted_t4)

    return permuted_iop


def cost_of_fermion_ordering(iop, k, cost_method="1bod2bod_frob",mask_1b=None,mask_2b=None):
    """Cost of fermion ordering. Used to help determine good reorderings.
    The cost is lower (i.e. better) when larger values are closer to diag.
    
    Parameters:
    iop (InteractionOperator): The interaction operator.
    k (int): The locality beyond which want to penalize.
    cost_method (str): The method to use for calculating the cost.
            "_frob" is Frobenius norm, "_1norm" is 1-norm.
            If "dist_" then 'k' is ignored, and penalty is prop to distance from diagonal.
    mask_1b (np.ndarray): Mask for one-body tensor. If not specified, will be created. If 
            specified, 'k' will be ignored.
    mask_1b (np.ndarray): Mask for one-body tensor. If not specified, will be created. If 
            specified, 'k' will be ignored.

    Returns:
    float: The cost of the fermion ordering.
    """
    
    assert isinstance(iop, InteractionOperator)
    assert cost_method in ( "1bod_1norm", "2bod_1norm", "1bod2bod_1norm",
                            "1bod_frob", "2bod_frob", "1bod2bod_frob",
                            "dist_1norm", "dist_frob"  ), cost_method

    n = count_qubits(iop)

    # Apply masks to both tensors
    # iop.one_body_tensor()
    # iop.two_body_tensor()

    cost = 0.

    # If "nomask_" then 'k' is ignored, and penalty is prop to distance from diagonal.
    if "dist_" in cost_method:
        
        if mask_1b is None:
            mask_1b = get_rank2_distance_mask( n )
        if mask_2b is None:
            mask_2b = get_rank4_distance_mask( n )

        if "1norm" in cost_method:
            cost += np.sum( np.abs(mask_1b * iop.one_body_tensor) )
            cost += np.sum( np.abs(mask_2b * iop.two_body_tensor) )
        elif "frob" in cost_method:
            cost += np.sum( np.square(mask_1b * iop.one_body_tensor) )
            cost += np.sum( np.square(mask_2b * iop.two_body_tensor) )

        return cost
        
    # Sum appropriately depending on norm
    if "1bod" in cost_method:
        # Element-wise multiplcation applies the mask
        if mask_1b is None:
            mask_1b = get_rank2_mask( n , k )
        masked_1body = np.multiply( mask_1b , iop.one_body_tensor )

        if "1norm" in cost_method:
            cost += np.sum( np.abs(masked_1body) )
        elif "frob" in cost_method:
            cost += np.sum( np.square(masked_1body) )

    if "2bod" in cost_method:
        # Element-wise multiplcation applies the mask
        if mask_2b is None:
            mask_2b = get_rank4_mask( n , k )
        masked_2body = np.multiply( mask_2b , iop.two_body_tensor )

        if "1norm" in cost_method:
            cost += np.sum( np.abs(masked_2body) )
        elif "frob" in cost_method:
            cost += np.sum( np.square(masked_2body) )

    return cost
    




def optimize_indexing_interaction_op(iop,k,cost_method="1bod2bod_frob",niter=10000,nconv=100):
    """Optimize indexing interaction operator
    
    Parameters:
        iop (InteractionOperator): The interaction operator.
        k (int): The locality beyond which want to penalize.
        cost_method (str): The method to use for calculating the cost.
        niter (int): Number of iterations to run.
        nconv (int): Number of iterations to run without improvement before stopping.

    Returns:
        InteractionOperator: The optimized InteractionOperator.
        overall_permutation (list): The overall permutation applied.
    """
    
    assert isinstance(iop, InteractionOperator)
    
    n = count_qubits(iop)
    
    # Get masks
    if "dist_" in cost_method:
        mask_1b = get_rank2_distance_mask( n )
        mask_2b = get_rank4_distance_mask( n )
    else:
        mask_1b = get_rank2_mask( n , k )
        mask_2b = get_rank4_mask( n , k )

    prev_cost = cost_of_fermion_ordering(iop, k, cost_method, mask_1b, mask_2b)

    overall_permutation = list(range(n))
    niter_without_improvement = 0

    for i in range(niter):
        
        # Apply masks to both tensors
        masked_1body = np.multiply( mask_1b , iop.one_body_tensor )
        masked_2body = np.multiply( mask_2b , iop.two_body_tensor )

        # Choose two random indices
        a,b = np.random.choice( n , 2 )
        perm = list(range(n))
        perm[a], perm[b] = perm[b], perm[a]
        # New iop with permuted indices
        permuted_iop = permute_iop_tensors(iop, perm)

        # Calculate new cost
        new_cost = cost_of_fermion_ordering(permuted_iop, k, cost_method, mask_1b, mask_2b)

        # If new cost is better, update
        if new_cost < prev_cost:
            iop = permuted_iop
            prev_cost = new_cost
            overall_permutation[a], overall_permutation[b] = overall_permutation[b], overall_permutation[a]
            niter_without_improvement = 0
        else:
            niter_without_improvement += 1

        # If no improvement for 'nconv' iterations, stop
        if niter_without_improvement >= nconv:
            break
        
    return iop, overall_permutation






def read_qubitop_from_file(fname,frmt="intel-1",skiplines=0):
    """Read QubitOperator from file"""

    # assert frmt in ["openfermion","intel-1"]
    assert frmt in ["intel-1",]

    op = QubitOperator()

    if frmt=="intel-1":
        with open(fname,'r') as f:

            # print(skiplines)

            # Skip 
            for ctr in range(skiplines):
                f.readline()

            line = f.readline()

            while line:

                # print(line)

                spl = line.split(':')

                term = spl[0]
                if term.strip()=="I":
                    term = " "
                k    = float(spl[1].split(';')[0])
                op  += QubitOperator(term,k)

                line = f.readline()


    return op




