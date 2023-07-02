import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.optimize import newton
from scipy.linalg import solve

from .immune_system import DeterministicImmuneSystem


class StochasticImmuneSystem(DeterministicImmuneSystem):
    def __init__(self,b,eps) -> None:
        """System of equations representing an immune system. Default parameters are hard-coded according to values by Kuznetsov et al.

        Parameters
        ----------
        b : float
            intensity of chemotherapy
        eps : float
            intensity of the noise
        """
        self._eps = eps
        super().__init__(b)
        
    @property
    def eps(self):
        return self._eps
    @eps.setter
    def eps(self,eps):
        self._eps = eps

    def sIntegrate(self,v0:np.ndarray,ts:np.ndarray,m:int,method='EM')-> None:
        """Integrate the stochastic system and save the trajectory.

        Parameters
        ----------
        v0 : np.ndarray
            initial point of the simulation.
        ts : np.ndarray
            array with times at which the solution shall be evaluated.
        m : int
            number of instances to simulate.
        method : str, optional
            specify which method should be used to integrate. Possible values are
            - `EM`: Euler-Maruyama
            - `RK`: 2-step Runge-Kutta

            Note that the convergence of RK is stronger, but it requires more calls to the right hand side of the system.
            RK is faster for large time intervals, since it allows to take less steps.

            By default 'EM'.

        Raises
        ------
        ValueError
            unrecognised value for `method`.
        """
        n = len(ts)

        v = np.zeros((m,n,2))
        h = ts[-1]/n

        v[:,0,:] = v0

        fvec = np.vectorize(self._f,signature='(2),() -> (2)')
        gvec = np.vectorize(self._g,signature='(2),() -> (2)')

        match method:
            case 'EM':
                for i in range(1,n):
                    G = np.random.normal(0,1,size = (m,))
                    v[:,i,:] = v[:,i-1,:] + fvec(v[:,i-1,:],0) * h + (gvec(v[:,i-1,:],0).T * np.sqrt(h) * G).T
            case 'RK':
                sqrth = np.sqrt(h)
                k1 = np.zeros((m,2))
                k2 = np.zeros((m,2))
                for i in range(1,n):
                    G = np.random.normal(0,1,size = (m,))
                    s = np.random.choice([-1,1])
                    k1 = h*fvec(v[:,i-1,:],0) + (sqrth * (G - s) * gvec(v[:,i-1,:],0).T).T
                    k2 = h*fvec(v[:,i-1,:] + k1,0) + (sqrth * (G + s) * gvec(v[:,i-1,:] + k1,0).T).T
                    v[:,i,:] = v[:,i-1,:] + 0.5*(k1+k2)
            case _:
                raise ValueError("'"+method+"' is an unrecognised method. Try 'EM' or 'RK'.")
        self._v = v

    def phase_portrait(self,every=1)-> tuple[plt.Figure ,plt.Axes]:
        """Draw the phase portrait for the current solution.

        Parameters
        ----------
        every : int, optional
            when plotting random trajectories, plot one every `every` instances. By default 1

        Returns
        -------
        tuple[plt.Figure ,plt.Axes]
            figure and axes of the plot
        """
        fig,ax = plt.subplots()
        if self.v.ndim == 2:
            ax.plot(self.v[:,0],self.v[:,1],color='b')
            ax.set_title('Deterministic trajectory')
        else:
            ax.plot(self.v[::every,:,0].T,self.v[::every,:,1].T,color='b')
            ax.set_title('Stochastic trajectory')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return fig,ax
    

    def S(self,v:np.ndarray,t)->np.ndarray:
        """Matrix of random disturbances

        Parameters
        ----------
        v : np.ndarray
            where to evaluate the matrix
        t : any
            time. Not used for this matrix

        Returns
        -------
        np.ndarray
            The 2x2 matrix of disturbances.
        """
        y = v[1]
        return np.diag([0,y**2/(16*(1+y**4)**2)])
    
    def computeSensitivityMatrix(self,v:np.ndarray)->np.ndarray:
        """Compute the stochastic sensitivity matrix as a solution of a system of equations.

        Parameters
        ----------
        v : np.ndarray
            where to evaluate the matrix

        Returns
        -------
        np.ndarray
            the stochastic sensitivity matrix.
        """
        J = self.J(v,0)
        S = self.S(v,0)

        A = np.array([[2*J[0,0],J[0,1],J[0,1],0],
                      [J[1,0],J[0,0]+J[1,1],0,J[0,1]],
                      [J[1,0],0,J[0,0]+J[1,1],J[0,1]],
                      [0,J[1,0],J[1,0],2*J[1,1]]])
        W = solve(A,-S.ravel())
        return W.reshape((2,2))
    

    def computeSensitivityFunction(self,v0:np.ndarray,T:float,N:int,method = 'numerical')->tuple[np.ndarray,np.ndarray]:
        """Compute the stochastic sensitivity function as a solution of a boundary value problem.

        Parameters
        ----------
        v0 : np.ndarray
            initial point
        T : float
            period of the cycle
        N : int
            number of steps
        method : str, optional
            Allowed values:
            - `numerical`: solve the problem with a forward differences scheme
            - `analytical`: solve the problem by (numerically) evaluating the analytical solution
            Numerical integration is faster.
            By default 'numerical'

        Returns
        -------
        m,p: tuple[np.ndarray,np.ndarray]
            - m : array of values of the function at the integration points
            - p : array of vectors orthogonal to the trajectory at each point

        Raises
        ------
        ValueError
            unsupported method.
        """
        ts = np.linspace(0,T,N)
        h = ts[1]-ts[0]

        v = odeint(self._f, v0, ts)

        a = np.zeros(N)
        s = np.zeros(N)
        pp = np.zeros((N,2))

        for i in range(N):
            p = np.array([self._f(v[i],0)[1],-self._f(v[i],0)[0]])
            p /= np.sqrt(p[0]**2+p[1]**2)
            J = self.J(v[i],0)
            S = self.S(v[i],0)

            a[i] = p.T @ ((J.T + J) @ p )
            s[i] = p.T @ (S @ p)
            pp[i] = np.copy(p)

        match method:
            case 'numerical':
                diag = [-(1+ h * a[i]) for i in range(N)]
                diag1 = np.ones(N-1)

                A = sp.sparse.diags([diag,diag1],[0,1],format = 'csr')
                A[N-1,1] = 1
                m = sp.sparse.linalg.spsolve(A,h*s)
            case 'analytical':
                g = np.array([np.exp(np.trapz(a[:i],ts[:i])) for i in range(N)])
                h = np.array([np.trapz(s[:i]/g[:i],ts[:i]) for i in range(N)])
                c = g[-1]*h[-1]/(1+g[-1])

                m = g*[c+h]
                m = m.reshape(-1)
            
            case _:
                raise ValueError("'"+method+"' is an unrecognised method. Try 'numerical' or 'analytical'.")
        return m, pp
    
    def _g(self,v,t):
        eps = self.eps
        y = v[1]

        return np.array([ 0 , 0.25 * eps*y/(1+y**4)])
