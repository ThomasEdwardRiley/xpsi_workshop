from __future__ import print_function, division

import numpy as np
import math
from scipy.stats import truncnorm

import xpsi
from xpsi.global_imports import _G, _csq, _km, _M_s, _2pi, gravradius

class CustomPrior(xpsi.Prior):
    """ A custom (joint) prior distribution.
    
    Source: Fictitious
    Model variant: ST-U
        Two single-temperature, simply-connected circular hot regions with
        unshared parameters.
    
    Parameter vector:
    
        * p[0] = distance (kpc)
        * p[1] = (rotationally deformed) gravitational mass (solar masses)
        * p[2] = coordinate equatorial radius (km)
        * p[3] = inclination of Earth to rotational axis (radians)
        * p[4] = primary region centre colatitude (radians)
        * p[5] = primary region angular radius (radians)
        * p[6] = primary region log10(local comoving blackbody temperature [K])
        * p[7] = secondary cap centre colatitude (radians)
        * p[8] = secondary cap angular radius (radians)
        * p[9] = secondary cap log10(local comoving blackbody temperature [K])
        * p[10] = primary cap phase shift (cycles); (alias for initial azimuth, periodic)
        * p[11] = secondary cap phase shift (cycles)
    
    """
    def __init__(self, bounds, spacetime):
        """
        :param obj spacetime:
            Bit of a hack to access spacetime properties for defining
            the support of the prior.
        
        """
        # Execute abstract parent initialiser
        super(CustomPrior, self).__init__(bounds)

        assert isinstance(spacetime, xpsi.Spacetime),\
                'Invalid type for ambient spacetime object.'

        self._spacetime = spacetime

    def __call__(self, p):
        """ Evaluate distribution at :obj:`p`.
        
        :param list p: Model parameters values.
        
        :returns: Logarithm of the distribution evaluated at :obj:`p`.
        
        """
        for i, b in enumerate(self._bounds):
            if None not in b:
                if not b[0] <= p[i] <= b[1]:                                                                                                        
                    return -np.inf
        
        i = self._spacetime.num_params
        # update and access spacetime properties
        self._spacetime.update(*p[:i])

        # based on contemporary EOS theory
        if not self._spacetime.R <= 16.0*_km:
            return -np.inf

        # photon sphere
        if not 1.5 < self._spacetime.R_r_s:
            return -np.inf

        epsilon = self._spacetime.epsilon
        zeta = self._spacetime.zeta
        mu = math.sqrt(-1.0 / (3.0 * epsilon * (-0.788 + 1.030 * zeta)))

        # 2-surface cross-section have a single maximum in |z|
        # i.e., an elliptical surface; minor effect on support
        if mu < 1.0:
            return -np.inf

        R_p = 1.0 + epsilon * (-0.788 + 1.030 * zeta)
        
        # polar radius causality for ~static star (static ambient spacetime)
        # if R_p < 1.5 / self._spacetime.R_r_s:
        #     return -np.inf

        # limit polar radius to try to exclude deflections >= \pi radians
        if R_p < 1.76 / self._spacetime.R_r_s:
            return -np.inf
        
        # enforce order in hot region colatitude
        if p[4] > p[7]:
            return -np.inf

        theta_p = p[4]
        phi = (p[10] - 0.5 - p[11]) * _2pi
        rho_p = p[5]

        theta_s = p[7]
        rho_s = p[8]

        ang_sep = xpsi.HotRegion._psi(theta_s, phi, theta_p)

        # hot regions cannot overlap
        if ang_sep < rho_p + rho_s:
            return -np.inf

        return 0.0

    def inverse_sample(self, hypercube):
        """ Draw sample uniformly from the distribution via inverse sampling. """
        
        p = super(CustomPrior, self).inverse_sample(hypercube)

        # distance
        p[0] = truncnorm.ppf(hypercube[0], -2.0, 7.0, loc=0.3, scale=0.1)

        # phase of primary hot region
        if p[10] > 0.5:
            p[10] -= 1.0

        # phase of secondary hot region
        if p[11] > 0.5:
            p[11] -= 1.0

        return p

    def inverse_sample_and_transform(self, hypercube):

        p = self.transform(self.inverse_sample(hypercube))

        return p
    
    inverse_sample_and_transform.__doc__ = xpsi.Prior.inverse_sample_and_transform.__doc__

    @staticmethod
    def transform(p):
        """ A transformation for post-processing. """

        if not isinstance(p, list):
            p = list(p)
        
        # compactness ratio M/R_eq
        p += [gravradius(p[1]) / p[2]]
        
        # phase transforms
        if p[10] < 0.0:
            tempp = p[10] + 1.0
        else:
            tempp = p[10]
        
        temps = 0.5 + p[11]
        
        # phase separation
        if temps >= tempp:
            p += [temps - tempp]
        else:
            p += [1.0 - tempp + temps]
        
        # angle combinations
        p += [p[3] - p[4]]
        p += [p[3] + p[4]]
        p += [p[3] - p[7]]

        return p