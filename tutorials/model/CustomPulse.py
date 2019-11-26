from __future__ import print_function, division

import numpy as np
import math

import xpsi

from xpsi.likelihoods.default_background_marginalisation import eval_marginal_likelihood
from xpsi.likelihoods.default_background_marginalisation import precomputation

class CustomPulse(xpsi.Pulse):
    """ A custom calculation of the logarithm of the likelihood.
    We extend the :class:`xpsi.Pulse.Pulse` class to make it callable.
    We overwrite the body of the __call__ method. The docstring for the
    abstract method is copied.
    """

    def __init__(self, workspace_intervals = 1000, epsabs = 0, epsrel = 1.0e-8,
                 epsilon = 1.0e-3, sigmas = 10.0, **kwargs):
        """ Perform precomputation. """

        super(CustomPulse, self).__init__(**kwargs)

        try:
            self._precomp = precomputation(self._data.counts.astype(np.int32))
        except AttributeError:
            print('Warning: No data... can synthesise data but cannot evaluate a '
                  'likelihood function.')
        else:
            self._workspace_intervals = workspace_intervals
            self._epsabs = epsabs
            self._epsrel = epsrel
            self._epsilon = epsilon
            self._sigmas = sigmas

    def __call__(self, p, *args, **kwargs):
        """
        Parameter vector:
        * p[0] = phase shift primary (alias for initial azimuth/phase of photosphere)
        * p[1] = phase shift secondary
        """
        self.shift = np.array(p)

        self.loglikelihood, self.expected_counts, self.background_signal = \
                eval_marginal_likelihood(self._data.exposure_time,
                                          self._data.phases,
                                          self._data.counts,
                                          self._pulse,
                                          self._phases,
                                          self._shift,
                                          self._precomp,
                                          self._workspace_intervals,
                                          self._epsabs,
                                          self._epsrel,
                                          self._epsilon,
                                          self._sigmas,
                                          kwargs.get('llzero'))

    __call__.__doc__ = xpsi.Pulse.__call__.__doc__ + __call__.__doc__