"""
Univariate loess function, like in R.

References
----------
Hastie, Tibshirani, Friedman. (2009) The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition: Chapter 6.

Cleveland, W.S. (1979) "Robust Locally Weighted Regression and Smoothing Scatterplots". Journal of the American Statistical Association 74 (368): 829-836.
"""

import pdb

import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
from scipy.linalg import lstsq


def loess(endog, exog, q, d, w=None):
    """
    LOESS (Locally Weighted Scatterplot Smoothing)

    Parameters
    ----------
    endog: 1-D numpy array
        The y-values of the observed points
    exog: 1-D numpy array
        The x-values of the observed points
    q: int
        Number of points to take into account for local regression
    d: int
        Degree of polynomial used for local regressio
    w: 1-D numpy array
        Weights. Optional.

    Returns
    -------
    out: numpy array
        A numpy array with two columns. The first column
        is the sorted x values and the second column the
        associated estimated y-values.

    """

    x = exog
    y = endog


    if exog.ndim != 1:
        raise ValueError('exog must be a vector')
    if endog.ndim != 1:
        raise ValueError('endog must be a vector')
    if endog.shape[0] != x.shape[0] :
        raise ValueError('exog and endog must have same length')
    if w is None or isinstance(w, float):
        w = np.ones(exog.shape)
    if w.ndim != 1.:
        raise ValueError('w must be a vector')
    if w.shape[0] != x.shape[0] :
        raise ValueError('w and endog must have same length')
    

    n = exog.shape[0]
    fitted = np.zeros(n)

    ##k = int(frac * n)

    index_array = np.argsort(exog)
    x_copy = np.array(exog[index_array])
    y_copy = endog[index_array]

    fitted, weights = _loess_fit(x_copy, y_copy, q, d)

    ##for i in xrange(it):
    ##    _loess_robustify_fit(x_copy, y_copy, fitted,
    ##                                        weights, k, n)

    out = np.array([x_copy, fitted]).T
    out.shape = (n,2)

    return out


def _loess_fit(x_copy, y_copy, q, d):
    """
    The initial weighted local linear regression for loess.

    Parameters
    ----------
    x_copy : 1-d ndarray
        The x-values/exogenous part of the data being smoothed
    y_copy : 1-d ndarray
        The y-values/ endogenous part of the data being smoothed
    q : int
        The number of data points which affect the polynomial fit for
        each estimated point
    d : int
        The degree of the locally fitted polynomial

    Returns
    -------
    fitted : 1-d ndarray
        The fitted y-values
    weights : 2-d ndarray
        An n by k array. The contribution to the weights in the
        local linear fit coming from the distances between the
        x-values

    """
    n = x_copy.shape[0]
    weights = np.zeros((n,q), dtype = x_copy.dtype)
    nn_indices = [0,q]

    fitted = np.zeros(n)

    for i in xrange(n):
        #note: all _loess functions are inplace, no return
        left_width = x_copy[i] - x_copy[nn_indices[0]]
        right_width = x_copy[nn_indices[1]-1] - x_copy[i]
        width = max(left_width, right_width)
        _loess_wt_standardize(weights[i,:],
                                x_copy[nn_indices[0]:nn_indices[1]],
                            x_copy[i], width)
        _loess_tricube(weights[i,:])
        ###weights[i,:] = np.sqrt(weights[i,:])

        x_i = x_copy[nn_indices[0]:nn_indices[1]]
        y_i = y_copy[nn_indices[0]:nn_indices[1]]
#        pdb.set_trace()
        beta = polyfit(x_i, y_i, deg=d, w=weights[i])
        fitted[i] = polyval(x_copy[i], beta)
#        beta = lstsq(weights[i,:].reshape(k,1) * X, y_i)[0]
#        fitted[i] = beta[0] + beta[1]*x_copy[i]

        _loess_update_nn(x_copy, nn_indices, i+1)


    return fitted, weights


def _loess_wt_standardize(weights, new_entries, x_copy_i, width):
    """
    The initial phase of creating the weights.
    Subtract the current x_i and divide by the width.

    Parameters
    ----------
    weights : ndarray
        The memory where (new_entries - x_copy_i)/width will be placed
    new_entries : ndarray
        The x-values of the k closest points to x[i]
    x_copy_i : float
        x[i], the i'th point in the (sorted) x values
    width : float
        The maximum distance between x[i] and any point in new_entries

    Returns
    -------
    Nothing. The modifications are made to weight in place.

    """
    weights[:] = new_entries
    weights -= x_copy_i
    weights[:] = np.absolute(weights)
    weights /= width



#def _loess_robustify_fit(x_copy, y_copy, fitted, weights, k, n):
#    """
#    Additional weighted local linear regressions, performed if
#    iter>0. They take into account the sizes of the residuals,
#    to eliminate the effect of extreme outliers.
#
#    Parameters
#    ----------
#    x_copy : 1-d ndarray
#        The x-values/exogenous part of the data being smoothed
#    y_copy : 1-d ndarray
#        The y-values/ endogenous part of the data being smoothed
#    fitted : 1-d ndarray
#        The fitted y-values from the previous iteration
#    weights : 2-d ndarray
#        An n by k array. The contribution to the weights in the
#        local linear fit coming from the distances between the
#        x-values
#    k : int
#        The number of data points which affect the linear fit for
#        each estimated point
#    n : int
#        The total number of points
#
#   Returns
#    -------
#    Nothing. The fitted values are modified in place.
#
#
#    """
#    nn_indices = [0,k]
#    X = np.ones((k,2))
#
#    residual_weights = np.copy(y_copy)
#    residual_weights.shape = (n,)
#    residual_weights -= fitted
#    residual_weights = np.absolute(residual_weights)#, out=residual_weights)
#    s = np.median(residual_weights)
#    residual_weights /= (6*s)
#    too_big = residual_weights>=1
#    _loess_bisquare(residual_weights)
#    residual_weights[too_big] = 0
#
#
#    for i in xrange(n):
#
#        total_weights = weights[i,:] * residual_weights[nn_indices[0]:
#                                                        nn_indices[1]]
#
#        X[:,1] = x_copy[nn_indices[0]:nn_indices[1]]
#        y_i = total_weights * y_copy[nn_indices[0]:nn_indices[1]]
#        total_weights.shape = (k,1)
#
#        beta = lstsq(total_weights * X, y_i)[0]
#
#        fitted[i] = beta[0] + beta[1] * x_copy[i]
#
#        _loess_update_nn(x_copy, nn_indices, i+1)





def _loess_update_nn(x, cur_nn,i):
    """
    Update the endpoints of the nearest neighbors to
    the ith point.

    Parameters
    ----------
    x : iterable
        The sorted points of x-values
    cur_nn : list of length 2
        The two current indices between which are the
        k closest points to x[i]. (The actual value of
        k is irrelevant for the algorithm.
    i : int
        The index of the current value in x for which
        the k closest points are desired.

    Returns
    -------
    Nothing. It modifies cur_nn in place.

    """
    while True:
        if cur_nn[1]<x.size:
            left_dist = x[i] - x[cur_nn[0]]
            new_right_dist = x[cur_nn[1]] - x[i]
            if new_right_dist < left_dist:
                cur_nn[0] = cur_nn[0] + 1
                cur_nn[1] = cur_nn[1] + 1
            else:
                break
        else:
            break

def _loess_tricube(t):
    """
    The _tricube function applied to a numpy array.
    The tricube function is (1-abs(t)**3)**3.

    Parameters
    ----------
    t : ndarray
        Array the tricube function is applied to elementwise and
        in-place.

    Returns
    -------
    Nothing
    """
    #t = (1-np.abs(t)**3)**3
    t[:] = np.absolute(t) #, out=t) #numpy version?
    _loess_mycube(t)
    t[:] = np.negative(t) #, out = t)
    t += 1
    _loess_mycube(t)

def _loess_mycube(t):
    """
    Fast matrix cube

    Parameters
    ----------
    t : ndarray
        Array that is cubed, elementwise and in-place

    Returns
    -------
    Nothing

    """
    #t **= 3
    t2 = t*t
    t *= t2


