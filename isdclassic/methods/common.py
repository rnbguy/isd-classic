import itertools
import logging
import numpy as np
from abc import ABC, abstractmethod
from isdclassic.utils import lu

logger = logging.getLogger(__name__)


class ISDWithoutLists():
    ALG_BRUTEFORCE = 'bruteforce'
    ALG_PRANGE = 'prange'
    ALG_LEE_BRICKELL = 'lee_brickell'
    ALG_LEON = 'leon'
    ALG_ALL = [ALG_BRUTEFORCE, ALG_PRANGE, ALG_LEE_BRICKELL, ALG_LEON]

    def __init__(self, h, s, t, alg_name, rref_mode="standard"):
        if alg_name not in self.ALG_ALL:
            raise Exception("Algorithm name undefined")
        self.h = h
        self.s = s
        self.t = t
        self.alg_name = alg_name
        self.r = h.shape[0]
        self.n = h.shape[1]
        self.k = self.n - self.r
        self.rref_mode = rref_mode
        self.result = {}
        self.result['completed'] = False
        logger.debug("*******")
        logger.debug("s={0}, t={1}, H=\n{2}".format(self.s, self.t, self.h))

    @abstractmethod
    def run(self):
        pass

    def get_result(self):
        if self.result['completed']:
            return self.result

    def get_matrix_rref(self, **kwargs):
        logger.debug("*******")
        logger.debug("s={0}, t={1}, H=\n{2}".format(self.s, self.t, self.h))
        logger.debug("r={0}, n={1}, k={2}".format(self.r, self.n, self.k))

        hp, hr, u, perm = self.get_rref(self.h, self.rref_mode, **kwargs)

        # Apply U to s to obtain s_signed and applies mod2 to only obtain bits
        s_sig = np.mod(np.dot(u, self.s), 2)
        logger.debug("s signed is {0}".format(s_sig))
        return hp, hr, u, perm, s_sig

    @classmethod
    def get_rref(cls, h, mode, **kwargs):
        if mode == "standard":
            return cls.get_rref_std(h, **kwargs)
        elif mode == "preselect":
            return cls.get_rref_pre(h, **kwargs)

    @classmethod
    def get_rref_pre(cls, h, **kwargs):
        # Trying to permute and then obtain the RREF
        exit_condition_rref = False
        # From now on exit_condition_rref is used to continue the algorithm until we found
        # the right rref, i.e. having the identity matrix on the right
        for cols in kwargs['comb']:
            logger.debug(f"cols {cols}")
            missing = tuple(set(range(h.shape[1])) - set(cols))
            hp = np.hstack((h[:, missing], h[:, cols]))
            iden = np.eye(h.shape[1])
            perm = np.hstack((iden[:, missing], iden[:, cols]))
            # hr stands for the matrix put in RREF, u for the transformation matrix
            # applied to obtain the RREF from the original matrix
            # We are trying to get the RREF of hp with the identity matrix r x r
            # placed to the right of hp
            hr, u = cls._rref(hp)
            # If rref returns None, it means that reduction was not possible,
            # i.e. the rightmost r x r matrix is not full-rank (different from
            # the id matrix in our case.
            exit_condition_rref = not (all(item is None for item in (hr, u)))
            # input("c")
            logger.debug("perm is \n{0}".format(perm))
            logger.debug("hr, that is u.h.p is \n{0}".format(hr))
            logger.debug("u is \n{0}".format(u))
            if exit_condition_rref:
                logger.debug(
                    "EXIT CONDITION RREF IS TRUE, GOING TO CHECK WEIGHT")
                return hp, hr, u, perm
            else:
                logger.debug("exit condition rref is false, retrying")

    @classmethod
    def get_rref_std(cls, h, **kwargs):
        # Trying to permute and then obtain the RREF
        exit_condition_rref = False
        hp, hr, u, perm = None, None, None, None
        # From now on exit_condition_rref is used to continue the algorithm until we found
        # the right rref, i.e. having the identity matrix on the right
        while (not exit_condition_rref):
            # p stands for permutation matrix, hp is the permuted version of h
            # We are trying to permute the columns of H in such a way that the
            # columns of the information set I, with |I| = k, are packed to the
            # left of h.
            hp, perm = cls._permute(h)
            # hr stands for the matrix put in RREF, u for the transformation matrix
            # applied to obtain the RREF from the original matrix
            # We are trying to get the RREF of hp with the identity matrix r x r
            # placed to the right of hp
            hr, u = cls._rref(hp)
            # If rref returns None, it means that reduction was not possible,
            # i.e. the rightmost r x r matrix is not full-rank (different from
            # the id matrix in our case.
            exit_condition_rref = not (all(item is None for item in (hr, u)))
            if exit_condition_rref:
                logger.debug(
                    "EXIT CONDITION RREF IS TRUE, GOING TO CHECK WEIGHT")
            else:
                logger.debug("exit condition rref is false, retrying")

            logger.debug("perm is \n{0}".format(perm))
            logger.debug("hr, that is u.h.p is \n{0}".format(hr))
            logger.debug("u is \n{0}".format(u))

        return hp, hr, u, perm

    # Random permutation of columns (by default) or rows
    # Return the permutated matrix and the permutation matrix utilized
    @staticmethod
    def _permute(m, cols=True):
        """
        Random permutation of columns (by default) or rows.
        
        :param m: the original matrix as a numpy.array
        :param cols: True (default) to permute the columns; False to permute the rows.
        :returns: The permutated matrix mp and the permutation matrix used p
        :rtype: tuple(numpy.array, numpy.array)
        
        """
        # length of columns or rows
        axis = m.shape[1] if cols else m.shape[0]
        # Create an identity matrix I and permute its columns to obtain
        # a permutation matrix
        i = np.eye(axis)
        p = np.random.permutation(i)

        # Post-multiply to permute columns, pre-multiply to permute rows
        # No need to do modulo operations since p contains only ones.
        mp = np.dot(m, p) if cols else np.dot(p, m)
        return (mp, p)

    @staticmethod
    def _rref(m):
        """Uses LU decomposition w/ Doolittle algorithm, i.e. PA = LU.
        
        :param m: the original matrix as numpy.array
        :returns: mr, the matrix M in RREF (and also unitary); U the matrix used to obtain the RREF
        :rtype: tuple
        
        """
        # WARNING: The (ptot, ltot, u) returned from get_rref() are different from ours.
        # Basically, the returned u is the original matrix put in RREF (so u corresponds to our mr);
        # ltot is the matrix of transformations applied to the original matrix to
        # obtain the RREF (so ltot corresponds to our u)
        # Note that the 1st parameter returned by the get_rref function is not used
        _, u, mr = lu.get_rref(m, startAtEnd=True, mod=2)
        #logger.debug("u is\n {0}".format(u))
        return (mr, u)
