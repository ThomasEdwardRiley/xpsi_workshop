from __future__ import print_function, division

import numpy as np
import math

import xpsi

class CustomInstrument(xpsi.Instrument):
    """ A model of the NICER telescope response.
    
    """
    def __init__(self, channels, channel_edges, *args):
        super(CustomInstrument, self).__init__(*args)
        self._channels = channels
        self._channel_edges = channel_edges

    @property
    def channels(self):
        return self._channels

    @property
    def channel_edges(self):
        """ Get the channel edges. """
        return self._channel_edges

    def construct_matrix(self, p):
        """ Implement response matrix parametrisation. """

        return self.matrix

    def __call__(self, p, signal, *args):
        """ Overwrite. """

        matrix = self.construct_matrix(p)

        self._folded_signal = np.dot(matrix, signal)

        return self._folded_signal

    @classmethod
    def from_response_files(cls, num_params, bounds,
                            ARF, RMF, max_input, min_input=0,
                            channel_edges=None):
        """ Constructor which converts response files into :class:`numpy.ndarray`s.
        :param str ARF: Path to ARF which is compatible with
                                :func:`numpy.loadtxt`.
        :param str RMF: Path to RMF which is compatible with
                                :func:`numpy.loadtxt`.
        :param str channel_edges: Optional path to edges which is compatible with
                                  :func:`numpy.loadtxt`.
        """

        if min_input != 0:
            min_input = int(min_input)

        max_input = int(max_input)

        try:
            ARF = np.loadtxt(ARF, dtype=np.double, skiprows=3)
            RMF = np.loadtxt(RMF, dtype=np.double)
            if channel_edges:
                channel_edges = np.loadtxt(channel_edges, dtype=np.double, skiprows=3)[:,1:]
        except:
            print('A file could not be loaded.')
            raise
            
        matrix = np.ascontiguousarray(RMF[min_input:max_input,20:201].T, dtype=np.double)

        edges = np.zeros(ARF[min_input:max_input,3].shape[0]+1, dtype=np.double)

        edges[0] = ARF[min_input,1]; edges[1:] = ARF[min_input:max_input,2]

        for i in range(matrix.shape[0]):
            matrix[i,:] *= ARF[min_input:max_input,3]

        channels = np.arange(20, 201)

        return cls(channels, channel_edges[20:202,-2],
                   num_params, bounds, matrix, edges)