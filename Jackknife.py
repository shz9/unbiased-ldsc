import numpy as np


class Jackknife(object):
    '''

    (Modified version of LDSC Jackknife base class to allow more flexibility)

    Base class for jackknife objects. Input involves x,y, so this base class is tailored
    for statistics computed from independent and dependent variables (e.g., regressions).
    The __delete_vals_to_pseudovalues__ and __jknife__ methods will still be useful for other
    sorts of statistics, but the __init__ method will need to be overriden.

    Parameters
    ----------
    x : np.matrix with shape (n, p)
        Independent variable.
    y : np.matrix with shape (n, 1)
        Dependent variable.
    n_blocks : int
        Number of jackknife blocks
    *args, **kwargs :
        Arguments for inheriting jackknives.

    Attributes
    ----------
    n_blocks : int
        Number of jackknife blocks
    p : int
        Dimensionality of the independent varianble
    N : int
        Number of datapoints (equal to x.shape[0])

    Methods
    -------
    jknife(pseudovalues):
        Computes jackknife estimate and variance from the jackknife pseudovalues.
    delete_vals_to_pseudovalues(delete_vals, est):
        Converts delete values and the whole-data estimate to pseudovalues.
    get_separators():
        Returns (approximately) evenly-spaced jackknife block boundaries.
    '''

    def __init__(self, func_inputs, n_blocks=None, separators=None):

        N = np.unique([fi.shape[0] for fi in func_inputs])

        if len(N) > 1:
            raise ValueError('Function inputs must of the same size.')
        else:
            self.N = N[0]

        self.func_inputs = func_inputs

        if separators is not None:
            if max(separators) != self.N:
                raise ValueError(
                    'Max(separators) must be equal to number of data points.')
            if min(separators) != 0:
                raise ValueError('Max(separators) must be equal to 0.')
            self.separators = sorted(separators)
            self.n_blocks = len(separators) - 1
        elif n_blocks is not None:
            self.n_blocks = n_blocks
            self.separators = self.get_separators(self.N, self.n_blocks)
        else:
            raise ValueError('Must specify either n_blocks are separators.')

        if self.n_blocks > self.N:
            raise ValueError('More blocks than data points.')

    @classmethod
    def jknife(cls, pseudovalues):
        '''
        Converts pseudovalues to jackknife estimate and variance.

        Parameters
        ----------
        pseudovalues : np.matrix pf floats with shape (n_blocks, p)

        Returns
        -------
        jknife_est : np.matrix with shape (1, p)
            Jackknifed estimate.
        jknife_var : np.matrix with shape (1, p)
            Variance of jackknifed estimate.
        jknife_se : np.matrix with shape (1, p)
            Standard error of jackknifed estimate, equal to sqrt(jknife_var).
        jknife_cov : np.matrix with shape (p, p)
            Covariance matrix of jackknifed estimate.

        '''

        n_blocks = pseudovalues.shape[0]

        jknife_cov = np.atleast_2d(np.cov(pseudovalues.T, ddof=1) / n_blocks)
        jknife_var = np.atleast_2d(np.diag(jknife_cov))
        jknife_se = np.atleast_2d(np.sqrt(jknife_var))
        jknife_est = np.atleast_2d(np.mean(pseudovalues, axis=0))

        return {
            'Estimate': jknife_est,
            'Variance': jknife_var,
            'SE': jknife_se,
            'Cov': jknife_cov
        }

    @classmethod
    def delete_values_to_pseudovalues(cls, delete_values, est):
        '''
        Converts whole-data estimate and delete values to pseudovalues.

        Parameters
        ----------
        delete_values : np.matrix with shape (n_blocks, p)
            Delete values.
        est : np.matrix with shape (1, p):
            Whole-data estimate.

        Returns
        -------
        pseudovalues : np.matrix with shape (n_blocks, p)
            Psuedovalues.

        Raises
        ------
        ValueError :
            If est.shape != (1, delete_values.shape[1])

        '''
        n_blocks, p = delete_values.shape
        if est.shape != (1, p):
            raise ValueError(
                'Different number of parameters in delete_values than in est.')

        return n_blocks * est - (n_blocks - 1) * delete_values

    @classmethod
    def get_separators(cls, N, n_blocks):
        '''Define evenly-spaced block boundaries.'''
        return np.floor(np.linspace(0, N, n_blocks + 1)).astype(int)

    def delete_values(cls, func, s):
        '''
        Compute delete values by deleting one block at a time.

        Parameters
        ----------
        x : np.matrix with shape (n, p)
            Independent variable.
        y : np.matrix with shape (n, 1)
            Dependent variable.
        func : function (n, p) , (n, 1) --> (1, p)
            Function of x and y to be jackknived.
        s : list of ints
            Block separators.

        Returns
        -------
        delete_values : np.matrix with shape (n_blocks, p)
            Delete block values (with n_blocks blocks defined by parameter s).

        Raises
        ------
        ValueError :
            If x.shape[0] does not equal y.shape[0] or x and y are not 2D.

        '''
        d = [func(
            *(np.vstack([x[0:s[i], ...], x[s[i + 1]:, ...]]) for x in cls.func_inputs))
            for i in range(len(s) - 1)]

        return np.concatenate(d, axis=0)

    def resample(cls, func):
        del_vals = cls.delete_values(func, cls.separators)
        return cls.jknife(cls.delete_values_to_pseudovalues(del_vals, func(*cls.func_inputs)))
