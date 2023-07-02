import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import newton


class DeterministicImmuneSystem():
    def __init__(self,b) -> None:
        """System of equations representing an immune system. Default parameters are hard-coded according to values by Kuznetsov et al.

        Parameters
        ----------
        b : float
            intensity of chemotherapy
        """
        self._b = b

        self._params = {'sigma' : 0.1181,
            'rho' : 1.131,
            'eta' : 20.19,
            'mu' : 0.001,
            'delta' : 0.3743,
            'alpha' : 1.636,
            'beta' : 0.002}
        
        self._findE1()
        self._E0 = np.array([self._params['sigma']/self._params['delta'],0])

    @property
    def b(self):
        return self._b
    @b.setter
    def b(self,b):
        self._b = b

    @property
    def E0(self):
        return self._E0

    @property
    def E1(self):
        return self._E1
    
    @property
    def v(self)->np.ndarray:
        return self._v
    
    def integrate(self,v0:np.ndarray,ts:np.ndarray)-> None:
        """Integrate the system. Uses `scipy.integrate.odeint`, which implements a Runge-Kutta method.

        Parameters
        ----------
        v0 : np.ndarray
            initial point
        ts : np.ndarray
            array of time points for the integration
        """
        self._v =  odeint(self._f, v0, ts)

    def phase_portrait(self)-> tuple[plt.Figure ,plt.Axes]:
        """Draw the phase portrait for the current solution.

        Returns
        -------
        tuple[plt.Figure ,plt.Axes]
            figure and axis of the plot.
        """
        fig,ax = plt.subplots()
        ax.plot(self.v[:,0],self.v[:,1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return fig,ax
    
    def J(self,v:np.ndarray,t)->np.ndarray:
        """Jacobian of the rhs of the system.

        Parameters
        ----------
        v : np.ndarray
            where to evaluate the Jacobian
        t : _type_
            time. Not used for this system.

        Returns
        -------
        np.ndarray
            Jacobian evauated at point v
        """
        rho = self._params['rho']
        eta = self._params['eta']
        mu = self._params['mu']
        delta = self._params['delta']
        alpha = self._params['alpha']
        beta = self._params['beta']
        b = self.b

        x = v[0]
        y = v[1]
        return np.array([[y**4 * ( rho/(eta + y**4) - mu ) - delta , 4 * x * y**3 * ( rho*eta/(eta + y**4)**2 - mu )],
                         [-0.25 * y , 0.25* (alpha * (1- 5 * beta * y**4) - x - b * (1-3*y**4)/(1 + y**4)**2)]])

    def limitCyclePeriod(self,v0:np.ndarray,ts:np.ndarray)->float:
        """Find the period of a limit cycle. 
        The method is quite naive and computes the period as the difference in time between passes of y - mean(y) through zero.
        The result depends on the number of time steps and on whether the initial condition belongs to the cycle.

        Parameters
        ----------
        v0 : np.ndarray
            starting point. Should be on the cycle.
        ts : np.ndarray
            array of times for the integration of the system.

        Returns
        -------
        float
            approximation of the period.
        """
        v =  odeint(self._f, v0, ts)

        data = v[:,1]-np.mean(v[:,1])
        times = []
        for i in range(len(data)-1):
            if data[i] > 0 and data[i+1] < 0:
                times.append((ts[i+1]+ts[i])/2)
        diff = np.diff(times)
        return np.mean(diff)

    def limitCyclePeriodFourier(self, v0:np.ndarray, tf:float, N:int)->float:
        """Compute the period of a cycle using its Fourier transform.
        The period is computed via the foundamental frequency.

        This method usually requires many time steps to get a good representation of the peaks in the transform.
        It is very sensitive to the choice of parameters and hard to fine tune.

        It is mainly implemented because it is interesting.

        Parameters
        ----------
        v0 : np.ndarray
            initial condition, should be on the cycle.
        tf : float
            final time of integration
        N : int
            number of integration steps

        Returns
        -------
        float
            period
        """
        ts = np.linspace(0,tf,N)
        v =  odeint(self._f, v0, ts)

        ft = np.fft.fft(v[:,1]-np.mean(v[:,1]),N)
        freq = np.fft.fftfreq(len(v[:,1]),d = tf/N)
        idx = np.argmax(np.abs(ft)**2)
        T = 1/np.abs(freq[idx])
        
        return T


    def _f(self,v,t):
        sigma = self._params['sigma']
        rho = self._params['rho']
        eta = self._params['eta']
        mu = self._params['mu']
        delta = self._params['delta']
        alpha = self._params['alpha']
        beta = self._params['beta']
        b = self.b

        x = v[0]
        y = v[1]
        return np.array([sigma + rho * (x * y**4)/(eta + y**4) - mu * x * y**4 - delta * x,0.25* (alpha * y * (1- beta * y**4) - x * y- b * y/(1 + y**4))])

    def _findE1(self):
        sigma = self._params['sigma']
        rho = self._params['rho']
        eta = self._params['eta']
        mu = self._params['mu']
        delta = self._params['delta']
        alpha = self._params['alpha']
        beta = self._params['beta']
        b = self.b

        def ff(y):
            return sigma + (rho*(y**4)/(eta+y**4)-mu*y**4-delta) * (alpha*(1-beta*y**4)-b/(1+y**4))
        def ffp(y):
            return 4*y**3 * ((eta*rho/(eta+y**4)**2 - mu)*(alpha*(1-beta*y**4)-b/(1+y**4))+(rho*(y**4)/(eta+y**4)-mu*y**4-delta)* (b/(1+y**4)**2 - beta))

        yE = newton(ff,1.65,ffp,maxiter=100)
        xE = alpha*(1-beta*yE**4) - b/(1+yE**4)
        self._E1 = np.array([xE,yE])
