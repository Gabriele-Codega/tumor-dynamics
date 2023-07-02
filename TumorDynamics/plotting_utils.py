import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import matplotlib.lines as ln
import numpy as np
from scipy.special import erfinv


def confidenceBand(v:np.ndarray, eps:float, m:np.ndarray,p:np.ndarray,P:float = 0.99)->tuple[ln.Line2D,ln.Line2D]:
    """Build the edges of the confidence band around a cycle.

    Parameters
    ----------
    v : np.ndarray
        points in phase space of the evolution along the cycle.
    eps : float
        noise intensity
    m : np.ndarray
        values of the stochastic sensitivity function along the cycle. Should be the values of `m` at points `v`.
    p : np.ndarray
        vectors orthogonal to the trajectory at points `v`.
    P : float, optional
        fiducial probability, by default 0.99.

    Returns
    -------
    tuple[matplotlib.lines.Line2D,matplotlib.lines.Line2D]
        borders of the confidence band.
    """
    
    n = len(m)
    bp = np.array([v[i] + erfinv(P) * eps * np.sqrt(2*m[i]) * p[i] for i in range(n)])
    bm = np.array([v[i] - erfinv(P) * eps * np.sqrt(2*m[i]) * p[i] for i in range(n)])

    pLine = ln.Line2D(bp[:,0],bp[:,1], linestyle='--', color='b')
    mLine = ln.Line2D(bm[:,0],bm[:,1], linestyle='--', color='b')

    return pLine, mLine


def confidenceEllipse(v:np.ndarray, eps:float, W:np.ndarray, P:float = 0.99)->ptc.Ellipse:
    """Build a confidence ellipse around an attractor.

    Parameters
    ----------
    v : np.ndarray
        coordinates of the attractor
    eps : float
        noise intensity
    W : np.ndarray
        stochastic sensitivity matrix evaluated at `v`.
    P : float, optional
        fiducial probability, by default 0.99

    Returns
    -------
    matplotlib.patches.Ellipse
        confidence ellipse.
    """
    eigval, eigvec = np.linalg.eig(W)

    a = np.sqrt(-eigval[0]*eps**2*2*np.log(1-P))
    b = np.sqrt(-eigval[1]*eps**2*2*np.log(1-P))
    confEllipse = ptc.Ellipse([v[0],v[1]],2*a,2*b,angle=np.degrees(np.arctan(eigvec[1,0]/eigvec[0,0])),fill=False,edgecolor='b',ls='--')
    return confEllipse