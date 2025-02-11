# Data structure for partitioning of a quantum operator

"""
These datastructures provide more organization and information 
for partitioning schemes of quantum operators. While the partitioning.py
module provides the functions to actually partition the operators, it
does not provide e.g. qubit IDs and other functionality.

The definition of the data structure follows definitions in 
Sawaya et al. arxiv:2408.11898. Specifically,

- An operator H is partitioned into sets {M_i} which are *not* 
necessarily mutually commuting.

- Each M_q is a sum of {W_qr}.

- Each W_qr is a tensor product A_qr1 (x) A_qr2 (x) ...


"""



def reduce_qub_ids(qubitop, qids=None):
    """Reduce the qubit IDs in a QubitOperator

    Args:
        qubitop: QubitOperator
        qids: list of qubit IDs

    Returns:
        QubitOperator with qubit IDs reduced
    """
    
    pass


def get_reduced_unitary(qids, qubitop):
        
    pass




# class Mop():


class Wop():

    def __init__(self, qubitop, qubits):
        """Initialize Wop object

        Args:
            

        Returns:

        """
        
        pass




    def get_unitaries_as_dict():


class parts_organizer:

    def __init__(self, qubitop, partitioning_scheme, **kwargs):
        """Initialize parts_organizer object

        Args:
            qubitop: QubitOperator
            partitioning_scheme: str, in {"SI FC","SI QWC","SI k-QWC","noclid","bosonic"}

        Returns:
            None

        """
        
        # M ops - each is a sum of W ops
        self.Ms = []
        













