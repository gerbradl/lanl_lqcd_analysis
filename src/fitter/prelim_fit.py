import collections
import logging
import itertools
import numpy as np
import lsqfit
import gvar as gv
import os
import h5py
import sys
import pytest
import argparse
import importlib
import corrfitter

import fitter.corr_functions as cf 
import fitter.load_data as ld
# import fitter.plotting as plot
LOGGER = logging.getLogger(__name__)

class PrelimFit(object):
    """
    Quick and dirty fit of a 2pt correlator. Taken from Lepage:
    Derived form Lepage's corrfitter module:
    https://github.com/gplepage/corrfitter/blob/master/src/corrfitter/_corrfitter.py
    
    Gab(t) = sn * sum_i an[i]*bn[i] * fn(En[i], t)
               + so * sum_i ao[i]*bo[i] * fo(Eo[i], t)
    where ``(sn, so)`` is typically ``(1, -1)`` and ::
        fn(E, t) =  exp(-E*t) + exp(-E*(tp-t)) # tp>0 -- periodic
               or   exp(-E*t) - exp(-E*(-tp-t))# tp<0 -- anti-periodic
               or   exp(-E*t)                  # if tp is None (nonperiodic)
        fo(E, t) = (-1)**t * fn(E, t)
    """
    
    
    def __init__(self, data, ampl='0(1)', dE='1(1)', E=None, s=(1, -1),
                 tp=None, tmin=6, svdcut=1e-6, osc=False, nterm=10):
        """
        Args:
            data: list / array of correlator data
            ampl: str, the prior for the amplitude. Default is '0(1)'
            dE: str, the prior for the energy splitting. Default is '1(1)'
            E0: str, the prior for the ground state energy. Default is None,
                which corresponds to taking the value for dE for the ground
                state as well
            s: tuple (s, so) with the signs for the towers of decay and
                oscillating states, repsectively. Default is (1, -1)
            tp: int, the periodicity of the data. Negative tp denotes
                antiperiodic "sinh-like" data. Default is None, corresponding
                to exponential decay
            tmin: int, the smallest tmin to include in the "averaging"
            svdcut: float, the desired svd cut for the "averaging"
            osc: bool, whether to estimate the lowest-lying oscillating state
                instead of the decay state
            nterms: the number of terms to include in the towers of decaying
                and oscillating states.
        Raises:
            RuntimeError: Can't estimate energy when cosh(E) < 1.
        """
        self.osc = osc
        self.nterm = nterm
        s = self._to_tuple(s)
        a, ao = self._build(ampl)
        dE, dEo = self._build(dE, E)
        s, so = s
        tmax = None

        if tp is None:
            # Data is not periodic
            # Model data with exponential decay
            def g(E, t):
                return gv.exp(-E * t)

        elif tp > 0:
            # Data is periodic
            # Model with a cosh
            def g(E, t):
                return gv.exp(-E * t) + gv.exp(-E * (tp - t))
            if tmin > 1:
                tmax = -tmin + 1
            else:
                tmax = None
        elif tp < 0:
            # Data is antiperiodic
            # Model with a sinh
            def g(E, t):
                return gv.exp(-E * t) - gv.exp(-E * (-tp - t))
            # Reflect and fold the data around the midpoint
            tmid = int((-tp + 1) // 2)
            data_rest = lsqfit.wavg(
                [data[1:tmid], -data[-1:-tmid:-1]], svdcut=svdcut)
            data = np.array([data[0]] + list(data_rest))

        else:
            raise ValueError('FastFit: bad tp')

        t = np.arange(len(data))[tmin:tmax]
        data = data[tmin:tmax]
        # print(data)

        if not t.size:
            raise ValueError(
                'FastFit: tmin too large? No t values left. '
                f'(tmin, tmax, tp)=({tmin}, {tmax}, {tp}).'
                )
        self.tmin = tmin
        self.tmax = tmax

        if osc:
            data *= (-1) ** t * so
            a, ao = ao, a
            dE, dEo = dEo, dE
            s, so = so, s

        d_data = 0.
        E = np.cumsum(dE)

        # Sum the tower of decaying states, excluding the ground state
        for aj, Ej in list(zip(a, E))[1:]:
            d_data += s * aj * g(Ej, t)

        # Sum the full tower of oscillating states
        if ao is not None and dEo is not None:
            Eo = np.cumsum(dEo)
            for aj, Ej in zip(ao, Eo):
                d_data += so * aj * g(Ej, t) * (-1) ** t
        # Marginalize over the exicted states
        data = data - d_data
        self.marginalized_data = data
        # Average over the remaining plateau
        meff = 0.5 * (data[2:] + data[:-2]) / data[1:-1]
        ratio = lsqfit.wavg(meff, prior=gv.cosh(E[0]), svdcut=svdcut)

        if ratio >= 1:
            self.E = type(ratio)(gv.arccosh(ratio), ratio.fit)
            self.ampl = lsqfit.wavg(data / g(self.E, t) / s,
                                    svdcut=svdcut, prior=a[0])
        else:
            LOGGER.warn(
                'Cannot estiamte energy in FastFit: cosh(E) = %s', 
                str(ratio)
            )
            self.E = None
            self.ampl = None

    def _to_tuple(self, val):
        """Convert val to tuple."""
        if isinstance(val, tuple):
            return val
        if self.osc:
            return (None, val)
        return (val, None)

    def _build(self, x, x0=(None, None)):
        x = self._to_tuple(x)
        x0 = self._to_tuple(x0)
        return (self._build_prior(x[0], x0[0]),
                self._build_prior(x[1], x0[1]))

    def _build_prior(self, x, x0):
        if x is None:
            return x
        x = gv.gvar(x)
        dx = 0 if abs(x.mean) > 0.1 * x.sdev else 0.2 * x.sdev
        xmean = x.mean
        xsdev = x.sdev

        if x0 is None:
            first_x = x
        else:
            first_x = gv.gvar(x0)

        return (
            [first_x + dx] +
            [gv.gvar(xmean + dx, xsdev) for i in range(self.nterm - 1)]
        )

    def __str__(self):
        return (
            "FastFit("
            "E: {} ampl: {} chi2/dof [dof]: {:.1f} {:.1f} [{}] "
            "Q: E:{:.1f} ampl:{:.1f} "
            "(tmin,tmax)=({},{}))"
        ).format(
            self.E, self.ampl, self.E.chi2 / self.E.dof,
            self.ampl.chi2 / self.ampl.dof, self.E.dof, self.E.Q, self.ampl.Q,
            self.tmin, self.tmax
        )
    # pylint: enable=invalid-name,protected-access 
    def to_dict(self):
        return {
            'energy': str(self.E),
            'ampl': str(self.ampl),
            'tmin': self.tmin,
            'tmax': self.tmax,
            'nterm': self.nterm,
            'osc': self.osc,
        }
if __name__ == '__main__':
    main()