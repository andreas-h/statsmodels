import numpy as np

from statsmodels.nonparametric.api import lowess

def stl(endog, exog, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni, no):
    """
    Seasonal-Trend decomposition procedure based on LOESS

    endog : ndarray
        time series data to be decomposed
    
    exog : ndarray

    n : int
        n = y.shape[0]
    
    np : int
        Period of the seasonal component.
        For example, if  the  time series is monthly with a yearly cycle, then
        np=12.

    ns : int
        Length of the seasonal smoother.
        The value of  ns should be an odd integer greater than or equal to 3.
        A value ns>6 is recommended. As ns  increases  the  values  of  the
        seasonal component at a given point in the seasonal cycle (e.g., January
        values of a monthly series with  a  yearly cycle) become smoother.

    nt : int
        Length of the trend smoother.
        The  value  of  nt should be an odd integer greater than or equal to 3.
        A value of nt between 1.5*np and 2*np is  recommended. As nt increases,
        the values of the trend component become  smoother.
        If nt is None, it is estimated as the smallest odd integer greater
        or equal to (1.5*np)/[1-(1.5/ns)]

    nl : int
        Length of the low-pass filter.
        The value of nl should  be an odd integer greater than or equal to 3.
        The smallest odd integer greater than or equal to np is used by default.

    isdeg : int
        Degree of locally-fitted polynomial in seasonal smoothing.
        The value is 0 or 1.

    itdeg : int
        Degree of locally-fitted polynomial in trend smoothing.
        The value is 0 or 1.
        
    ildeg : int
        Degree of locally-fitted polynomial in low-pass smoothing.
        The value is 0 or 1.
        
    nsjump : int
        Skipping value for seasonal smoothing.
        The seasonal smoother skips ahead nsjump points and then linearly
        interpolates in between.  The value  of nsjump should be a positive
        integer; if nsjump=1, a seasonal smooth is calculated at all n points.
        To make the procedure run faster, a reasonable choice for nsjump is
        10%-20% of ns. By default, nsjump= 0.1*ns.
        
    ntjump : int
        Skipping value for trend smoothing. If None, ntjump= 0.1*nt

    nljump : int
        Skipping value for low-pass smoothing. If None, nljump= 0.1*nl

    ni :int
        Number of loops for updating the seasonal and trend  components.
        The value of ni should be a positive integer.
        See the next argument for advice on the  choice of ni.
        If ni is None, ni is set to 1 for robust fitting, to 5 otherwise.

    no : int
        Number of iterations of robust fitting. The value of no should
        be a nonnegative integer. If the data are well behaved without
        outliers, then robustness iterations are not needed. In this case
        set no=0, and set ni=2 to 5 depending on how much security
        you want that  the seasonal-trend looping converges.
        If outliers are present then no=3 is a very secure value unless
        the outliers are radical, in which case no=5 or even 10 might
        be better.  If no>0 then set ni to 1 or 2.
        If None, then no is set to 15 for robust fitting, to 0 otherwise.

    returns
    
    rw : ndarray
        rw.shape = (n, )
    
    season : ndarray
        season.shape = (n, )
    
    trend : ndarray
        trend.shape = (n, )
    
    work : ndarray
        work.shape = (n + 2 * np, 5)
        
    """
    pass
