import numpy as np
import gvar as gv
import re


def inflate(params, frac):
    """
    Inflates the width on the priors to frac*mean, unless the existing width is
    already wider.
    """
    for key, value in params.items():
        mean = gv.mean(value)
        sdev = np.maximum(frac*np.abs(mean), gv.sdev(value))
        params[key] = gv.gvar(mean, sdev)
    return params


class BasePrior(object):
    """
    Basic class for priors
    Args:
        mapping: dict, the prior's key-value pairs
        extend: bool, whether or not to treat handle energies
            'dE' as 'log' priors
    Returns:
        BasePrior object
    """

    def __init__(self, mapping, extend=True):
        _check_duplicate_keys(mapping.keys())
        self.extend = extend
        self.dict = dict(_sanitize_mapping(mapping))
        self._verify_keys()

    def __getitem__(self, key):
        """Get value corresponding to key, allowing for 'log' terms."""
        if self.extend and _is_log(key):
            return np.log(self.dict.__getitem__(key[4:-1]))

        return self.dict.__getitem__(key)

    def __setitem__(self, key, value):
        """Set value corresponding to key, allowing for 'log' terms."""
        if self.extend and _is_log(key):
            self.dict.__setitem__(key[4:-1], np.exp(value))
        else:
            self.dict.__setitem__(key, value)

    def __len__(self):
        return self.dict.__len__()

    def __iter__(self):
        for key in self.keys():
            yield key

    def __repr__(self):
        string = '{'
        str_tmp = []
        for key, val in self.items():
            str_tmp.append("{0} : {1},".format(key.__repr__(), val.__repr__()))
        string += '\n'.join(str_tmp)
        string += '}'
        return string

    def __str__(self):
        string = '{'
        str_tmp = []
        for key, val in self.items():
            str_tmp.append("{0} : {1},".format(key.__str__(), val.__str__()))
        string += '\n'.join(str_tmp)
        string += '}'
        return string

    def items(self):
        """Overrides items to handle 'logs' gracefully."""
        for key in self.keys():
            yield key, self.__getitem__(key)

    def _keys(self):
        for key in self.dict.keys():
            # nota bene: this enforces that these keys use log priors!
            # if ('dE' in key) or ('fluctuation' in key):
            if ('dE' in key) or (':a' in key) or ('fluctuation' in key):
                yield 'log({0})'.format(key)
            else:
                yield key

    def keys(self):
        """Override keys to handle 'logs' gracefully."""
        if self.extend:
            return self._keys()
        return self.dict.keys()

    def values(self):
        """Override values to handle 'logs' gracefully."""
        for key in self.keys():
            yield self.__getitem__(key)

    def update(self, update_with, width=None, fractional_width=False):
        """Update keys in prior with dict 'update_with'."""
        keys = self.keys()
        for key in update_with:
            if key in keys:
                value = update_with[key]
                if width:
                    if not hasattr(value, '__len__'):
                        value = [value]
                    if fractional_width:
                        value = [gv.gvar(gv.mean(val), gv.mean(val) * width) for val in value]
                    else:
                        value = [gv.gvar(gv.mean(val), width) for val in value]
                self.__setitem__(key, value)

    @property
    def p0(self):
        """Get central values for initial guesses"""
        return {key: gv.mean(val) for key, val in self.items()}

def pion_energy(n):
    """
    Get the energy of the nth excited pion in MeV.
    """
    if n == 0:
        return 0.
    if n == 1:
        return gv.gvar(135, 50)
    if n == 2:
        return (gv.gvar(1300, 400))
    return gv.gvar(1300 + 400*(n-2), 400)

def proton_energy(n):
    """
    Get the energy of the nth excited proton in MeV.
    """
    if n == 0:
        return 0.
    if n == 1:
        return gv.gvar(938, 30)
    if n == 2:
        return (gv.gvar(1300, 400))
    return gv.gvar(1300 + 400*(n-2), 400)


def decay_amplitudes(n, a0="0.50(20)", ai="0.1(0.3)"):
    """
    Get basic amplitude guesses in lattice units for n total decaying states.
    """
    def _amplitude(n):
        if n == 0:
            return gv.gvar(a0)
        else:
            return gv.gvar(ai)
    return np.array([_amplitude(ni) for ni in range(n)])

class PhysicalSplittings():
    """
    Class for handling splittings inspired by physical results in the PDG.
    Both lattice units and MeV are supported.
    """
    def __init__(self, state):
        state = str(state).lower()
        if state not in ['pion', 'pi', 'pion_osc', 'pi_osc',
                         'kaon', 'k', 'kaon_osc', 'k_osc',
                         'd', 'd_osc',
                         'ds', 'ds_osc',
                         'b', 'b_osc',
                         'bs', 'bs_osc']:
            raise ValueError(f"Unrecognized state. Found state={state}")
        self.state = state

    def energy(self, n):
        if self.state in ('pion', 'pi'):
            return pion_energy(n)
        elif self.state in ('pion_osc', 'pi_osc'):
            return pion_osc_energy(n)

    def __call__(self, n, a_fm=None, scale=1.0):
        """
        Get the energy splittings for n states in MeV (when a_fm is None) or
        lattice units (when a_fm is specified).
        """
        # Get energies in MeV
        energies = np.array([self.energy(ni) for ni in range(n+1)])
        # Apply scaling factor
        energies = energies * scale
        # Convert to energy splittings
        dE = energies[1:] - energies[:-1]
        if a_fm is None:
            return dE
        # Convert MeV to lattice units
        dE = dE * a_fm / 197
        return dE
class MesonPrior(BasePrior):
    """
    Prior for mesonic two-point correlation function.
    Args:
        n: int, the number of decaying states
        no: int, the number of oscillating states
        amps: list of strings specifying the source and
            sink amplitudes. Default is ['a','b','ao','bo'].
        tag: str giving a tag/name, 
        ffit: corrfitter.fastfit object for estimating masses and
            amplitudes. Default is None.
    """

    def __init__(self, n=1, no=0, amps=None, tag=None, ffit=None, **kwargs):
        print(n,no)
        if n < 1:
            raise ValueError("Must have n_decay >=1.")
        if no < 0:
            raise ValueError("Must have n_oscillate > 0.")
        if amps is None:
            amps = ['a', 'b', 'ao', 'bo']
        super(MesonPrior, self).\
            __init__(MesonPrior._build(n, no, amps, tag, ffit), **kwargs)

    @staticmethod
    def _build(n_decay, n_oscillate, amps, tag=None, ffit=None):
        """Build the prior dict."""
        prior = {}
        # Decaying energies and amplitudes
        n = range(n_decay)
        prior['dE'] = [gv.gvar('1.0(1.0)')] +\
            [gv.gvar('0.6(0.6)') for _ in range(1, n_decay)]
        if 'a' in amps:
            prior['a'] = [gv.gvar('0.1(1.0)') for _ in n]
        if 'b' in amps:
            prior['b'] = [gv.gvar('0.1(1.0)') for _ in n]

        # Oscillating eneriges and amplitudes
        if n_oscillate > 0:
            no = range(0, n_oscillate)
            prior['dEo'] = [gv.gvar('1.65(50)')] +\
                           [gv.gvar('0.6(0.6)') for _ in range(1, n_oscillate)]
            if 'ao' in amps:
                prior['ao'] = [gv.gvar('0.1(1.0)') for _ in no]
            if 'bo' in amps:
                prior['bo'] = [gv.gvar('0.1(1.0)') for _ in no]

        # Extract guesses for the ground-state energy and amplitude
        # if ffit is not None:
        #     dE_guess = gv.mean(ffit.E)
        #     amp_guess = gv.mean(ffit.ampl)
        #     prior['dE'][0] = gv.gvar(dE_guess, 0.5 * dE_guess)
        #     if 'a' in amps:
        #         prior['a'][0] = gv.gvar(amp_guess, 2.0 * amp_guess)
        #     elif 'b' in amps:
        #         prior['b'][0] = gv.gvar(amp_guess, 2.0 * amp_guess)
        #     else:
        #         msg = "Error: Unrecognized amplitude structure?"
        #         raise ValueError(msg)

        # Convert to arrays
        keys = list(prior.keys())
        if tag is None:
            # Just convert to arrays
            for key in keys:
                prior[key] = np.asarray(prior[key])
        else:
            # Prepend keys with 'tag:' and then convert
            for key in keys:
                new_key = "{0}:{1}".format(tag, key)
                prior[new_key] = np.asarray(prior.pop(key))

        return prior

class MesonPriorPDG(BasePrior):
    """
    Class for building priors for analysis of pion 2pt functions inspired by
    physical results in the PDG.
    Args:
        nstates: namedtuple
        tag: str, the name of the state
        a_fm: float, the lattice spacing in fm
        scale: float, amount by which to scale the spectrum with respect to the
            PDG value(s). T
    Returns:
        MesonPriorPDG, a dict-like object containing the prior
    """
    def __init__(self, nstates, tag, a_fm=None, scale=1.0, **kwargs):
        prior = {}
        # Decaying states
        prior[f"{tag}:dE"] = PhysicalSplittings(tag)(nstates.n, a_fm, scale)
        prior[f"{tag}:a"] = decay_amplitudes(nstates.n)
        # Oscillating states
        if nstates.no:
            prior[f"{tag}:dEo"] = PhysicalSplittings(f"{tag}_osc")(nstates.no, a_fm, scale)
            prior[f"{tag}:ao"] = osc_amplitudes(nstates.no)
        super(MesonPriorPDG, self).__init__(mapping=prior, **kwargs)


class JointFitPrior(BasePrior):
    """
    Prior for joint fits to extract form factors.
    """

    def __init__(self, nstates, c2=None, positive_ff=True, **kwargs):
        tags = c2.keys()
        if c2 is None:
            c2 = {}
        
        else:
            JointFitPrior._verify_tags(tags)
        self.positive_ff = positive_ff
        super(JointFitPrior, self).__init__(
                mapping=self._build(nstates, c2),
                **kwargs)

    @staticmethod
    def _verify_tags(tags):
        """Verify that the tags (from nstates) are supported."""
        tags = set(tags)
        print(tags)
        valid_tags = [
            ['SS'],
            ['PS'],

        ]
        for k in valid_tags:
            
            
            if tags == set(k):
                return
        # raise ValueError("Unrecognized tags in JointFitPrior")

    def _build(self, nstates, c2):
        """Build the prior dict."""
        prior = JointFitPrior._make_meson_prior(nstates, c2)
        tmp = self._make_vmatrix_prior(nstates, c2)
        for key in tmp:
            prior[key] = tmp[key]
        return prior

    @staticmethod
    def _make_meson_prior(nstates, c2):
        """Build prior associated with the meson two-point functions."""
        tags = c2.keys()
        c2_src = c2['SS']
        c2_snk = c2['PS']

        meson_priors = [
            MesonPrior(nstates.n, nstates.no,
                       tag='SS', ffit=c2_src.fastfit),
            MesonPrior(nstates.m, nstates.mo,
                       tag='PS', ffit=c2_snk.fastfit),
        ]
        prior = {}
        for meson_prior in meson_priors:
            for key, value in meson_prior.items():
                prior[key] = value
        return prior

    def _make_vmatrix_prior(self, nstates, c2):
        """Build prior for the 'mixing matrices' Vnn, Vno, Von, and Voo."""
        def _abs(val):
            return val * np.sign(val)
        c2_snk = c2['PS'] #THIS OBJ SHOULD JUST BE MEMBER OF C2
        mass = None
        n = nstates.n
        no = nstates.no
        m = nstates.m
        mo = nstates.mo
        mass = c2_snk.fastfit.E

        # General guesses
        tmp_prior = {}
        tmp_prior['Vnn'] = gv.gvar(n * [m * ['0.1(10.0)']])
        tmp_prior['Vno'] = gv.gvar(n * [mo * ['0.1(10.0)']])
        tmp_prior['Von'] = gv.gvar(no * [m * ['0.1(10.0)']])
        tmp_prior['Voo'] = gv.gvar(no * [mo * ['0.1(10.0)']])
        # v = gv.mean(ds.v_guess)
        # verr = 0.5 * _abs(v)
        # tmp_prior['Vnn'][0, 0] = gv.gvar(v, verr)


        return tmp_prior

def vmatrix(nstates):
    """
    Get the prior for matrices of matrix elements Vnn, Vno, Von, and Voo in
    lattice units.
    """
    n = nstates.n
    no = nstates.no
    m = nstates.m
    mo = nstates.mo
    # General guesses
    prior = {}
    prior['Vnn'] = gv.gvar(n * [m * ['0.1(10.0)']])
    prior['Vno'] = gv.gvar(n * [mo * ['0.1(10.0)']])
    prior['Von'] = gv.gvar(no * [m * ['0.1(10.0)']])
    prior['Voo'] = gv.gvar(no * [mo * ['0.1(10.0)']])
    return prior

