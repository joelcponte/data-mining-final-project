"""Von Mises Mixture Model."""

import numpy as np

from scipy import linalg

from sklearn.mixture.base import BaseMixture, _check_shape
from sklearn.externals.six.moves import zip
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import row_norms


def _estimate_von_mises_parameters(X, resp):
    """Estimate the von mises distribution parameters.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data array.
    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.
    Returns
    -------
    weights : array-like, shape (n_components,)
        weight of components.
    means : array-like, shape (n_components, n_features)
        The centers of the current components.
	kappas :
    """
    n_samples,_ = X.shape
    nk = resp.sum(axis=0) + 1e-4
    weights = nk/n_samples                                                            #(n_components,)
    means = np.arctan(np.dot(resp.T, np.sin(X)) / np.dot(resp.T, np.cos(X)))          #(n_components, 1)
    extended_means = -1 * np.tile(means, resp.shape[0]).T                             #(n_samples, n_components)
    A_kappas = np.sum(np.multiply(resp, np.cos(extended_means + X)), axis = 0) / nk   #(n_components,)
    kappas = (2 * A_kappas - A_kappas**3) / (1 - A_kappas**2)                         #(n_components,)

    return weights, means, kappas[:, np.newaxis]

def _estimate_log_von_Mises_prob(X, means, kappas):
    """Estimate the log von Mises probability.

    Parameters
    ----------
    X : array-like, shape (n_samples, 1)

    means : array-like, shape (n_components, 1)

    kappas : array-like, shape (n_components, 1)

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, _ = X.shape
    n_components, _ = means.shape
    
    extended_means = -1 * np.tile(means, n_samples).T                        #(n_samples, n_components)
    extended_kappas = np.tile(kappas, n_samples).T                           #(n_samples, n_components)
    
    log_kappas = np.log( 2* np.pi * np.i0(kappas))                           #(n_components,)
    #print(log_kappas)
    extended_log_kappas = np.tile(log_kappas[:, np.newaxis], n_samples).T    #(n_samples, n_components)
    
    return np.multiply(extended_kappas, np.cos(extended_means + X)) - extended_log_kappas


class vonMisesMixture(BaseMixture):
    """ Von Mises Mixture.
    Representation of a von mises mixture model probability distribution.
    This class allows to estimate the parameters of a von mises mixture
    distribution.
    
    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components.
    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.
    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.
    max_iter : int, defaults to 100.
        The number of EM iterations to perform.
    n_init : int, defaults to 1.
        The number of initializations to perform. The best results are kept.
    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::
            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several time on similar problems.
    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.
    verbose_interval : int, default to 10.
        Number of iteration done before the next print.
    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weights of each mixture components.
    means_ : array-like, shape (n_components, n_features)
        The mean of each mixture component.
    kapps_ : array-like
        **********************
        
    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.
    lower_bound_ : float
        Log-likelihood of the best fit of EM.
    See Also
    --------
    BayesianGaussianMixture : Gaussian mixture model fit with a variational
        inference.
    """
    
    def __init__(self, n_components=1, tol=1e-3,reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 random_state=None, warm_start=False, verbose=0, verbose_interval=10):
        super(vonMisesMixture, self).__init__(n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params, random_state=random_state, 
            warm_start=warm_start, verbose=verbose, verbose_interval=verbose_interval)

    def _check_parameters(self, X):
        """Check the von mises mixture parameters are well defined."""
        pass


    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        self.weights_, self.means_, self.kappas_ = _estimate_von_mises_parameters(X, resp)
    
    
    def _m_step(self, X, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        self.weights_, self.means_, self.kappas_ = _estimate_von_mises_parameters(X, np.exp(log_resp))
        

    def _estimate_log_prob(self, X):
        """Estimate the log-probabilities log P(X | Z).

        Compute the log-probabilities per each component for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
        """
        return _estimate_log_von_Mises_prob(X, self.means_, self.kappas_)

    def _estimate_log_weights(self):
        """Estimate log-weights in EM algorithm, E[ log pi ] in VB algorithm.

        Returns
        -------
        log_weight : array, shape (n_components, )
        """
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _check_is_fitted(self):
        check_is_fitted(self, ['weights_', 'means_', 'kappas_'])
    
    def _get_parameters(self):
        """Get parameters
        """
        return (self.weights_, self.means_, self.kappas_)

    def _set_parameters(self, params):
        """Set parameters
        """
        (self.weights_, self.means_, self.kappas) = params
    
    def _n_parameters(self):
        _, n_features = self.means_.shape
        return n_features * self.n_components * 2 + self.n_components - 1
    
    def bic(self, X):
        return (-2 * self.score(X) * X.shape[0] + self._n_parameters() * np.log(X.shape[0]))
    
    def aic(self, X):
        """Akaike information criterion for the current model on the input X.
        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()