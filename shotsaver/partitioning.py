# Algos for partitioning Pauli sets

from . import paulirelat
from . import util

from openfermion import QubitOperator, count_qubits
import networkx as nx
import math

from copy import deepcopy


def get_op_partitioning(op,comm_method,opt_method,**kwargs):
    '''Returns sets of Paulis based on method chosen (e.g. original QWC arXiv:1907.03358)
    
    Args: 
        op (QubitOperator): 
        comm_method (bool): "Type" of commutation b/n pstrings
        opt_method (bool): Optimization (e.g. greedy_color, sorted insertion)
        kwargs: can be used in e.g. 'greedy' which is implemented via networkx

    Returns:
        list of list of terms, for example:
            [ [ (0,'X'), (1,'Y'), (2,'Z') ],
                [ (3,'Z',),],
                ...]

    '''

    # FUNCTION INCOMPLETE. RAISE ERROR.
    raise NotImplementedError()

    assert comm_method in ("qubitwise", "fullcommute")
    assert opt_method in ("sortedinsertion","greedy_color")

    if opt_method=="sortedinsertion":
        return get_si_sets(op,comm_method,**kwargs)
    elif opt_method=="greedy_color":
        relgraph = paulirelat.get_relationship_graph(op,comm_method)
        # Take complement
        relgraphC = nx.complement(relgraph)
        return nx.greedy_color(relgraphC, **kwargs) # ex. kwargs: strategy="largest_first"


def get_terms_ordered_by_abscoeff(op):
    '''Returns terms of QubitOperator, ordered by abs(coeff)

    Args:
        op (QubitOperator)

    Returns:
        list of tuples
    '''

    # Ensure is instance
    assert isinstance(op, QubitOperator)
    
    # Order the terms by absolute val of coefficient
    terms = sorted(op.terms.items(), key=lambda x: abs(x[1]), reverse=True)
    terms = [t[0] for t in terms]

    # Return terms
    return terms


def get_si_sets(op, comm_method="fullcommute", **options):
    '''Returns grouping from sorted insertion algo.

    Args:
        op (QubitOperator): 
        comm_method (bool): "Type" of commutation b/n pstrings
    
    Returns:
        list of list of terms, for example:
            [ [ (0,'X'), (1,'Y'), (2,'Z') ],
                [ (3,'Z',),],
                ...]

    '''
    # pytest run spe

    # Basic assertions
    assert isinstance(op, QubitOperator)
    if not callable(comm_method): # If it's not a function
        assert comm_method in ("qubitwise", "fullcommute")
        if comm_method=="qubitwise":
            comm_func = paulirelat.pstrings_qubitwise_commute
        elif comm_method=="fullcommute":
            comm_func = paulirelat.pstrings_commute


    # Commuting sets (as list datatype)
    commuting_sets = []

    # Order the terms by absolute val of coefficient
    terms_ord = get_terms_ordered_by_abscoeff(op)

    # Loop over terms
    for pstring in terms_ord:

        found_commuting_set = False

        # Loop over existing commuting sets
        for commset in commuting_sets:
            comm_checks = [comm_func(pstring,pstring2) for pstring2 in commset]

            # All must be true
            if all(comm_checks):
                # Add to set
                commset.append(pstring)
                found_commuting_set = True
                break
            
        if not found_commuting_set:
            # Create new commuting set
            commuting_sets.append([pstring])

    return commuting_sets


def get_sets_blocked_noclid_residual(op,blocksize,resid_comm_method="fullcommute",max_noclid_parts=None):
    '''Returns grouping for blocked-TTLD-with-residual.

    Blocks are defined contigously, hence blocks are determined entirely by qubit ordering.
    After a few large fragments are produced, the remaining terms ('residual') are partitioned
    using the specified commutation method with SortedInsertion.

    max_ttld_parts specifies how many times to do "windowing." If None, it's equal to blocksize.

    Args:
        op (QubitOperator): 
        blocksize (int): Number of qubits per block
        resid_comm_method (bool): "Type" of commutation b/n pstrings
        max_ttld_parts (int): Maximum number of TTLD parts (=blocksize if None)
    
    Returns:
        list of list of terms, for example:
            [ [ (0,'X'), (1,'Y'), (2,'Z') ],
                [ (3,'Z',),],
                ...]
    '''

    # # User-defined max_ttld_parts not yet implemented
    # if max_ttld_parts is not None:
    #     raise NotImplementedError("max_ttld_parts param not implemented yet")

    # Basic assertions
    assert isinstance(op, QubitOperator)
    if not callable(resid_comm_method): # If it's not a function
        assert resid_comm_method in ("qubitwise", "fullcommute")
        if resid_comm_method=="qubitwise":
            comm_func = paulirelat.pstrings_qubitwise_commute
        elif resid_comm_method=="fullcommute":
            comm_func = paulirelat.pstrings_commute
    if (max_noclid_parts is None) or max_ttld_parts>blocksize:
        max_noclid_parts = blocksize


    nblocks = math.ceil(count_qubits(op)/blocksize)

    # Each window has a set of sets of qubit ids
    # Window 1: [(0,1,2), (3,4,5), ...]
    # Window 2: [(1,2,3), (4,5,6), ...]
    # 
    sets_of_sets_of_qubit_sets = []
    for window in range(0,max_noclid_parts):
        set_of_qid_sets = []
        for block in range( nblocks ):
            start = window+blocksize*block
            set_of_qid_sets.append( set(range(start,start+blocksize)) )
        
        sets_of_sets_of_qubit_sets.append(set_of_qid_sets)

    # Create dict of all qubit sets
    qubit_sets_by_term = {}
    for term in op.terms:
        qubit_sets_by_term[term] = set([q for q,op in term])

    # Will remove terms from resid_op. Will keep op the same.
    resid_op = deepcopy(op)

    # Loop through op and place terms
    parts = []
    for window_set in sets_of_sets_of_qubit_sets:
        parts_this_window = []
        for set_of_qids in window_set:
            curr_terms = list(resid_op.terms.keys())
            for term in curr_terms:
                if qubit_sets_by_term[term].issubset(set_of_qids):
                    parts_this_window.append(term)
                    del resid_op.terms[term]
                    del qubit_sets_by_term[term]
        parts.append(parts_this_window)

    # Now partition the residual with SortedInsertion nd specified commutation method
    resid_parts = get_si_sets(resid_op,resid_comm_method)

    # Combine
    parts += resid_parts

    # Return
    return parts




class tensor_train:
    """Tensor train class.
    
    Two very different storage methods.
    "pstrings" is a list of pstrings. Their sum gives the tensor train.
    "numpy_list" is a list of numpy arrays. Their kronecker product gives the tensor train.
    """

    def __init__(self,k,repr_method="pstrings"):
        """Initializes tensor train"""

        assert repr_method in ("pstrings", "numpy_list"), (repr_method, "repr_method")

        self.k = k
        self.repr_method = repr_method
        self.free_qubits = set()

        # In "qubitoperator" repr, this is a list of QubitOperators
        if self.repr_method=="pstrings":
            self.tensor_train_qubops = []
        elif self.repr_method=="numpy_list":
            self.tensor_train_numpy_list = []

    
    def __repr__(self):
        if self.repr_method=="pstrings":
            return repr(self.tensor_train_qubops)
        elif self.repr_method=="numpy_list":
            return repr(self.tensor_train_numpy_list)


    def _add_term(self,new_term,free_qubits):
        """Adds term to tensor train.
        
        There is not a full check for whether the term fits the criteria;
        this method should be used only internally by the class."""

        self.tensor_train_qubops.append(new_term)
        new_free_qubits = self.free_qubits.union(free_qubits)
        # Partial check--at least make sure free qubits are fewer than 'k'
        if len(new_free_qubits)>self.k:
            raise Exception(f"Too many free qubits ({len(new_free_qubits)})")


    def get_pstrings(self):
        """Returns all pstrings in tensor train"""
        return self.tensor_train_qubops


    def attempt_add_term(self, new_term):
        """Attempts to add new term to tensor train
        
        Returns False if not possible,
            else returns the 'free qubits' e.g. (0,2)
            
        They are called 'free' because they can be anything. We also use the term 'mismatched'."""
        
        # Loop through all terms
        # Union of already-free qubits & new term's mismatched qubits w/ all terms
        temp_free_qubits = self.free_qubits
        for term in self.tensor_train_qubops:
            mismatched_qubits = paulirelat.get_mismatched_qubits(new_term,term)
            temp_free_qubits = temp_free_qubits.union(mismatched_qubits)
            if len(temp_free_qubits)>self.k:
                return False
        
        # If we get here, we can add the term, because the free qubits are fewer than k
        self._add_term(new_term,temp_free_qubits)
        # Return the free qubits
        return temp_free_qubits
    
    # def check_qwc_with_all_terms(self, inp_term):
    #     """Checks qubitwise commutation with all terms.

    #     Returns true if inp_term qubitwise-commutes with all terms.
        
    #     For some partitioning algos, this is necessary for determining whether
    #     new tensor train can be created in the same partition."""
        
    #     for term in self.tensor_train_qubops:

    #         if not paulirelat.pstrings_qubitwise_commute(inp_term,term):
    #             return False
    #     return True

    
    def check_disjoint_with_all_terms(self, inp_term):
        """Checks whether input term is disjoint with all existing terms.

        Returns true if inp_term is disjoint (over qubits) with all terms.
        
        For some partitioning algos, this is necessary for determining whether
        new tensor train can be created in the same partition."""
        
        for term in self.tensor_train_qubops:

            if not paulirelat.pstrings_are_disjoint(inp_term,term):
                return False
            
        return True


    def get_tensor_train(self):
        """Returns tensor train"""
        if self.repr_method=="pstrings":
            return self.tensor_train_qubop
        elif self.repr_method=="kron_numpy":
            return self.tensor_train_numpy
        

class partition_of_ttrains:
    """Partition of tensor trains.
    
    All tensor trains in the partition are supposed to have certain properites,
    for example all tensor trains ought to commute with each other (where
    free qubits *always* commute)."""

    def __init__(self, k, repr_method="pstrings"):
        """Initializes partition of tensor trains"""

        assert repr_method in ("pstrings", "numpy_list"), (repr_method, "repr_method")

        self.k = k
        self.repr_method = repr_method

        self.tensor_trains = [] # List of tensor trains

    # def get_fald_pauli_sets(self):
    #     """Returns the FALD sets"""
    #     return self.fald_sets

    # def get_fald_tensor_trains(self):
    #     """Returns the FALD tensor trains"""
    #     return self.tensor_trains
    
    def __repr__(self):
        strout = f"Partition of tensor trains. {len(self.tensor_trains)} tensor trains with k={self.k}."
        for tt in self.tensor_trains:
            strout += "\n" + repr(tt)
    
        return strout
    
    def __str__(self):
        return f"Partition of tensor trains. {len(self.tensor_trains)} tensor trains with k={self.k}."

    def _create_new_tensor_train(self,pstring):
        """Creates new tensor train"""
        
        self.tensor_trains.append( tensor_train(self.k,self.repr_method) )
        self.tensor_trains[-1].attempt_add_term(pstring)

    def get_pstrings(self):
        """Returns all pstrings in the partition."""
        pstrings = []
        for tt in self.tensor_trains:
            pstrings += tt.get_pstrings()

        return pstrings
    

    def attempt_add_term(self, new_term):
        """Attempts to add new term to each tensor train in partition
        
        Returns False if not possible,
            else returns the 'free qubits' e.g. (0,2)"""
        
        # # First find out which TTs the term QWCs with
        # # If it's not for more than one, then it can't be added to any TT
        # qwc_false_ttid = None # TT id where qwc is false
        # for ttid,tt in enumerate(self.tensor_trains):
        #     bool_qwc = tt.check_qwc_with_all_terms(new_term)
        #     if not bool_qwc:
        #         if qwc_false_ttid is None:
        #             qwc_false_ttid = ttid
        #         else:
        #             # More than one false qwc means can't fit into partition
        #             return False

        # # Exactly one false qwc check. See if it fits in that one; otherwise returns False.
        # if qwc_false_ttid is not None:
        #     tt = self.tensor_trains[qwc_false_ttid]
        #     # Attempt to add term to the one TT for which qwc is false
        #     free_qubits = tt.attempt_add_term(new_term)
        #     return free_qubits # This just returns False if adding failed
        

        # First find out which TTs the term is disjoint with
        # If it's not for more than one, then it can't be added to any TT
        disjoint_false_ttid = None # TT id where disjoint test fails
        for ttid,tt in enumerate(self.tensor_trains):
            bool_qwc = tt.check_disjoint_with_all_terms(new_term)
            if not bool_qwc:
                if disjoint_false_ttid is None:
                    disjoint_false_ttid = ttid
                else:
                    # More than one false qwc means can't fit into partition
                    return False

        # Exactly one false-disjoint check. See if it fits in that one; otherwise returns False.
        if disjoint_false_ttid is not None:
            tt = self.tensor_trains[disjoint_false_ttid]
            # Attempt to add term to the one TT for which qwc is false
            free_qubits = tt.attempt_add_term(new_term)
            return free_qubits # This just returns False if adding failed


        # If you're here it means the term is QWC with all TTs
        # Loop through all tensor trains in partition
        for tt in self.tensor_trains:
            free_qubits = tt.attempt_add_term(new_term)
            if free_qubits:
                # This means term was atted to the tt
                return free_qubits

        # If you're here, it means the term didn't fit in any tensor train,
        # but it did qubitwise commute with all terms in all ttrains.
        # This means we can begin a new ttrain inside this partition.
        self._create_new_tensor_train(new_term)
        # Return empty set of free qubits, as this is only term in new tensor train
        return set()
    

    def expectval(self, psi):
        """Returns expectation value"""
        raise NotImplementedError()


class set_of_ttrain_partitions:
    """Set of ttrain partitions.
    """

    def __init__(self, op, k, repr_method="pstrings"):
        """Initializes set of tensor train partitions"""
        self.k = k
        self.repr_method = repr_method
        self.partitions = []

        # Sort the terms by abs(coeff)
        terms_sorted = get_terms_ordered_by_abscoeff(op)

        # First partition
        self.partitions.append( partition_of_ttrains(self.k, self.repr_method) )

        # Loop over terms
        for term in terms_sorted:

            flag_placed = False
            for ptt in self.partitions:
                free_qubits = ptt.attempt_add_term(term)
                if free_qubits is not False:
                    flag_placed = True
                    break
            
            if not flag_placed:
                # Create new partition
                self.partitions.append( partition_of_ttrains(self.k, self.repr_method) )
                self.partitions[-1]._create_new_tensor_train(term)


    def get_partitions_of_pstrings(self):
        """Returns list of list of pstrings."""
        list_list_pstrings = []
        for part in self.partitions:
            list_list_pstrings.append( part.get_pstrings() )

        return list_list_pstrings





# def get_fald_pauli_sets(op, k, factor_method="si_simple"):
#     '''Returns grouping from FALD algo.
    
#     Args:
#         op (QubitOperator):
#         k (int): max locality of the diagonalization
#         factor_method (bool): "simple", "disjoint"
        
#     Returns:
#     dict:
#     {'partitions':
#         list of list of terms, for example:
#             [ [ (0,'X'), (1,'Y'), (2,'Z') ],
#                 [ (3,'Z',),],
#                 ...],
#     'free_qubits':
#         which qubits in each partition are free, for example:
#         [ (0,1,4), (0,2), ... ]
#     NO. CANNOT QUITE DO IT LIKE THIS. DIFFERENT TENSOR TRAINS HAVE DIFFERENT FREE QUBITS.

#     '''

#     assert factor_method in ("si_simple",)

#     if factor_method=="si_simple":
        
#         # Sort the terms by abs(coeff)
#         terms_sorted = get_terms_ordered_by_abscoeff(op)

#         # Convert all the terms to strings ('IIIZZ' etc)
#         terms_charstrings = dict([{pstring: util.paulistr2simplestr(pstring)} for pstring in terms_sorted])

#         # The partitions
#         #(In 'simple' method, each tensor train is in its own partition,
#         # but it should still be a list of list of lists.)
#         partitions  = []
#         free_qubits = []

#         # # Loop over terms
#         # for term in terms_sorted:
#         #     for partition 























