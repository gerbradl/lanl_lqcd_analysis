'''
Modules for:
2pt corr function
3pt corr function
eff mass function  
variants of exponential fit function
'''


import numpy as np
import sys
import lsqfit 
import importlib
import gvar as gv
import corrfitter as cf
import matplotlib.pyplot as plt
import matplotlib
import fitter.fastfit as fastfit #Lepage's prelim fitter -> p0 to speed up initial fits
# import figures as plt

def main():
    pass

def guess_tmax(ydata, sn):
    """Infer tmax based on signal to noise ratio."""
    if sn is None:
        return len(ydata) - 1
    guess = gv.sdev(ydata) / gv.mean(ydata) < sn
    if np.all(guess):
        tmax = len(ydata) - 1
    else:
        tmax = np.argmin(guess)
    return tmax


def effective_mass(data, ti=0, dt=1):
    """
    Computes effective mass from correlator data.
    Args:
        data: array or list, correlator data C(t)
        ti: int, the starting timeslice. Default is 0.
        dt: int, the step between timeslices. Default is 1.
    Returns:
        array, the effective mass
    """

    c_t = data[ti::dt][:-1]
    c_tpdt = data[ti::dt][1:]
    return np.array((1/dt)*np.log(c_t/c_tpdt))

def En(x, p, n):
        """
        g.s. energy is energy
        e.s. energies are given as dE_n = E_n - E_{n-1}
        """
        E = p['%s_E_0' % x['state']]
        for i in range(1, n+1):
            E += p['%s_dE_%d' % (x['state'], i)]
        return E


class TimeContainer(object):
    def __init__(self, tdata, tmin=0, tmax=None, nt=None, tp=None):
        self.tdata = np.asarray(tdata)
        if tmin < 0:
            raise ValueError('you are stuck in the past... ')
        self.tmin = tmin

        if tmax is None:
            self.tmax = len(tdata) - 1
        else:
            if tmax > len(tdata):
                raise ValueError('bad tmax')
            self.tmax = tmax

        if nt is None:
            self.nt = len(tdata)
        else:
            self.nt = nt

        if tp is None:
            self.tp = self.nt
        else:
            self.tp = tp

    @property
    def tfit(self):
        """Get fit times."""
        return self.tdata[self.tmin:self.tmax]

    @property
    def tdata_avg(self):
        """Fetch tdata safe for use with averaging functions."""
        # return np.arange(self.tmin, self.tmax - 2)
        return np.arange(len(self.tdata) - 2)

    def __str__(self):
        return "BaseTimes(tmin={0},tmax={1},nt={2},tp={3})".\
            format(self.tmin, self.tmax, self.nt, self.tp)

    def __repr__(self):
        return self.__str__()

class C_2pt(object):
    '''
    simple two-point correlation function
    x-data from input file should take the form:

    x['proton_SS'] = {'state':'proton'
              'corr':exp, 'n_state':3,
              't_range':np.arange(10,21,1), 'T':T
              'snk':'S', 'src':'S'
              'factorized_fit (z_snk_n z_src_n) or not (A_n)'
              }
    '''

    def __init__(self, tag, ydata,p, sn=None, prelim_fit=False, **time_kwargs):
        self.tag = tag
        self.ydata = ydata
        # self.x = x
        self.p = p
        self.sn = sn
        tdata = time_kwargs.pop('tdata', None)
        if tdata is None:
            tdata = np.arange(len(ydata))
        self.times = TimeContainer(tdata=tdata, **time_kwargs)
        self.times.tmax = 20 #guess_tmax(ydata, sn)

        # Estimate the ground-state energy and amplitude
        if self.prelim_fit:
            self.prelim_fit = prelim_fit.PrelimFit(
                data=self.ydata[:self.times.tmax],
                tp=self.times.tp,
                tmin=self.times.tmin)
        else:
            self.prelim_fit = None 
        self._mass = self.p['proton_E_0']
    
    def set_mass(self, mass):
        self._mass = self.p['proton_E_0']

    @property
    def mass(self):
        """Estimate the mass using prelim_fit."""
        if self._mass is not None:
            return self._mass
        if self.prelim_fit is not None:
            return self.prelim_fit.E
        msg = (
            "No PrelimFit result, "
            "you need to manually set the mass prior."
        )
        raise AttributeError(msg)


    def meff(self, avg=True):
        """Compute the effective mass of the correlator."""
        if avg:
            return effective_mass(self.avg())
        else:
            return effective_mass(self.ydata)

    def avg(self, mass=1.0):
        """
        Compute the time-slice-averaged two-point correlation function.
        """
        if self._mass is not None:
            mass = self._mass
        elif mass < 0:
            mass = self.prelim_fit.E
        c2 = self.ydata
        tmax = len(c2)
        c2_tp1s = np.roll(self.ydata, -1, axis=0)
        c2_tp2s = np.roll(self.ydata, -2, axis=0)
        c2bar = np.empty((tmax,), dtype=gv._gvarcore.GVar)
        for t in range(tmax):
            c2bar[t] = c2[t] / np.exp(-mass * t)
            c2bar[t] += 2 * c2_tp1s[t] / np.exp(-mass * (t + 1))
            c2bar[t] += c2_tp2s[t] / np.exp(-mass * (t + 2))
            c2bar[t] *= np.exp(-mass * t)
        return c2bar / 4.

    def __getitem__(self, key):
        return self.ydata[key]

    def __setitem__(self, key, value):
        self.ydata[key] = value

    def __len__(self):
        return len(self.ydata)

    def __str__(self):
        return "TwoPoint[tag='{}', tmin={}, tmax={}, nt={}, mass={}]".\
            format(self.tag, self.times.tmin, self.times.tmax, self.times.nt,
                   self.mass)
    def plot_corr(self, ax=None, avg=False, **kwargs):
        """Plot the correlator on a log scale."""
        if ax is None:
            _, ax = plt.subplots(1, figsize=(10, 5))
        if avg:
            y = self.avg()
            x = self.times.tfit[1:-1]
        else:
            y = self.ydata
            x = self.times.tdata
        ax = plt.mirror(ax=ax, x=x, y=y, **kwargs)
        ax.set_yscale('log')
        return ax
    
class C_3pt(object):
    """
    ThreePoint correlation function. Keys in ydata should be integer values,
    representing the Tau value/tsep. 
    Can handle:
    - tuple (tsrc,tsep)
    - single int tsep
    If neither, fallback to:
    1. parse input file search for set of tsep values
    2. request user supplied input via CLI 

    Returns C_3pt object 
    .. math::
        C_{ji}(t^{\\rm snk},t^{\\rm ins})
        = \sum_{mn} A^{\\rm snk}_{jn}
        e^{-E_{n} (t^{\\rm snk}-t^{\\rm ins})}
        V^{\\rm ins}_{nm}
        e^{-E_{m} t^{\\rm ins}}
        \\big[ A^{\\rm src}_{im} \\big]^{T}
    """
    def __init__(self, tag, ydata_3pt,t_ins,T, nt=None):
        #TODO prompt user override with t_ins and T from input file if keys not integers 
        self.tag = tag
        self.ydata_3pt = ydata_3pt
        self.t_ins = t_ins
        self.T = T
        # self._verify_ydict(nt)
        
        tmp = self.ydata_3pt 
        if nt is None:
            tdata = np.arange(len(tmp))
        else:
            tdata = np.arange(nt)
        self.times = TimeContainer(tdata=tdata, nt=nt)
        self.times.tmax =  25 #_infer_tmax(tmp, sn)

    def __str__(self):
        return "ThreePoint[tag='{}', tmin={}, tmax={}, nt={}, T={}]".\
            format(self.tag, self.times.tmin, self.times.tmax,
                   self.times.nt, sorted(list(self.T)))

    # def _confirm_T_keys(self):
    #     for tsnk in self.ydata_3pt:
    #         if not isinstance(tsnk, (int,float,np.int64,np.float32,tuple)):
    #             raise TypeError("T keys must be of real numerical type")

    #     try:
    #         np.unique([len(_ydata_3pt) for _ydata_3pt in self.ydata_3pt.values()]).item()
    #     except TypeError as _:
    #         raise RuntimeWarning("ydata_3pt values must be of equal length")


    # def _confirm_real(self, nt=None):
    #     """
    #      Confirms that a value is a real numeric type
    #      :returns: `True` if real, else False
    #     """
    #     for t in self.T:
    #         if not isinstance(t, (int,float,np.int64,np.float32)):
    #             raise TypeError("t_sink keys must be a real numerical type.")
    #     if nt is None:
    #         try:
    #             np.unique([len(arr) for arr in self.T.values()]).item()
    #         except ValueError as _:
    #             raise ValueError("Values in t_dict must be of equal length.")


    def avg(self, m_src, m_snk):
        """
        Computes a time-slice-averaged three-point correlation function.
        Combines the correlator BOTH at consecutive time slices and 
        consecutive src-snk seps 
        Args:
            m_src, m_snk: the masses of the ground states
                associated with the source at time t=0 and and the sink at
                time t=t_sink(Tau)
        Returns:
            c3bar: dict of arrays with the time-slice-averaged correlators
        """
        def _combine(ratio):
            """
            Combines according to (R(t) + 2*R(t+1) + R(t+2)) / 4
            """
            return 0.25 * (
                ratio
                + 2.0*np.roll(ratio, -1, axis=0)
                + np.roll(ratio, -2, axis=0)
            )

        c3bar = {}
        
        T = np.sort(np.array(self.T))
        print(T)
        t = np.arange(len(self.t_ins))
        print(t)
        # T_cut = T[1:] - T[:-1]
        # t_snks = np.sort(np.array(self.t_dict))
        # if self._confirm_real(self.t_ins):
        #     if self._confirm_real(self.T):
        for ts in T:
                c3 = self.ydata_3pt[ts] 
                print(c3) # C(t, T)
                ratio = c3 / np.exp(-m_src*t) / np.exp(-m_snk*(ts-t))
                tmp = _combine(ratio)
                c3bar[ts] = tmp * np.exp(-m_src*t) * np.exp(-m_snk*(ts-t))

        return c3bar

    @property
    # def T(self):
    #     """Returns the sink times T from the tuples (t,T)."""
    #     # T_empty = list()
    #     # for item in self.ydata_3pt.keys():
    #     #     if item[0] not in T_empty:
    #     #         T_empty.append(item[0]) 
    #     # return T_empt
    #     return list(self.keys())

    # def __getitem__(self, key):
    #     return self.ydata_3pt[key]

    # def __setitem__(self, key, value):
    #     self.ydata_3pt[key] = value

    def __len__(self):
        return len(self.ydata_3pt)

    # def __iter__(self):
    #     for key in self.keys():
    #         yield key

    def items(self):
        
        return self.ydata_3pt.items()

    def keys(self):
        
        return self.ydata_3pt.keys()

    def values(self):
        return self.ydata_3pt.values()

class CorrFunction:

    def En(self, x, p, n):
        '''
        g.s. energy is energy
        e.s. energies are given as dE_n = E_n - E_{n-1}
        '''
        E= 0
        if x['state'] in ['gA','gV']:
            E += p['%s_00' % x['state']]
            for i in range(1, n+1):
                E += p['%s_dE_%d' % ('proton', i)] #use wf overlap for proton
        elif x['state'] in 'proton':
            E += p['%s_E_0' % x['state']]
            for i in range(1, n+1):
                E += p['%s_dE_%d' % (x['state'], i)]
        return E

    def dEn(self, x, p, n):
        E = p['%s_dE_1' % x['state']]  
        for i in range(2, n+1):
            E += p['%s_dE_%d' % (x['state'], i)]
        return E

    def E_el_n(self, x, p, n):
        E = p['%s_E_el_1' % x['state']]
        for i in range(2, n+1):
            E += p['%s_E_el_%d' % (x['state'], i)]
        return E
    

    def exp(self, x, p):
        #print('DEBUG:',x)
        #sys.exit()
        r = 0
        t = x['t_range']
        for n in range(x['n_state']):
            E_n = self.En(x, p, n)
            if x['ztype'] == 'z_snk z_src':
                if x['state'] == 'proton':
                    z_src = p["%s_z%s_%d" % (x['state'], x['src'], n)]
                    z_snk = p["%s_z%s_%d" % (x['state'], x['snk'], n)]
                    r += z_snk * z_src * np.exp(-E_n*t)
                elif x['state'] in ['gA','gV']:
                    z_src = p["%s_z%s_%d" % ('proton', x['src'], n)]
                    z_snk = p["%s_z%s_%d" % ('proton', x['snk'], n)]
                    r += z_snk * z_src * np.exp(-E_n*t)

            elif x['ztype'] == 'A_snk,src':
                A = p['%s_z%s%s_%d' % (x['state'], x['snk'], x['src'], n)]
                r += A * np.exp(-E_n*t)
        return r


    def exp_open(self, x, p):
        ''' first add exp tower '''
        r = self.exp(x,p)
        ''' then add first open boundary condition term
            r += A_open * exp( -(E_2pi - m_pi)*(T-t) )

            we will model E_2pi = m_pi + dEOpen
            therefore, we can set E = dEOpen
        '''
        t = x['t_range']
        T = x['T']
        E = p['%s_dEOpen_1' % (x['state'])]
        if x['ztype'] == 'z_snk z_src':
            z_src = p["%s_z%sOpen_1" % (x['state'], x['src'])]
            z_snk = p["%s_z%sOpen_1" % (x['state'], x['snk'])]
            r += z_snk * z_src * np.exp(-E *(T-t))
        elif x['ztype'] == 'A_snk,src':
            A = p['%s_z%s%sOpen_1' % (x['state'], x['snk'], x['src'])]
            r += A * np.exp(-E *(T-t))
        return r

    def exp_gs(self,x,p):
        ''' model the correlator with a geometric series inspired model

            C(t) = A0 exp(-E0 t) / (1 - sum_n A_n exp( -dEn t))
        '''
        t  = x['t_range']
        E0 = self.En(x, p, 0)
        if x['ztype'] == 'z_snk z_src':
            z_src = p["%s_z%s_%d" % (x['state'], x['src'], 0)]
            z_snk = p["%s_z%s_%d" % (x['state'], x['snk'], 0)]
            r = z_snk * z_src * np.exp( -E0 * t)
        elif x['ztype'] == 'A_snk,src':
            A = p['%s_z%s%s_%d' % (x['state'], x['snk'], x['src'],0)]
            r = A * np.exp(-E0*t)
        ''' build denominator '''
        r_den = 1
        for n in range(1, x['n_state']):
            dEn = self.dEn(x, p, n)
            if x['ztype'] == 'z_snk z_src':
                z_src = p["%s_z%s_%d" % (x['state'], x['src'], n)]
                z_snk = p["%s_z%s_%d" % (x['state'], x['snk'], n)]
                r_den += -z_snk * z_src * np.exp(-dEn*t)
            elif x['ztype'] == 'A_snk,src':
                A = p['%s_z%s%s_%d' % (x['state'], x['snk'], x['src'],n)]
                r_den += -A * np.exp(-dEn*t)
        return r / r_den

    def cosh(self, x, p):
        r = 0
        t = x['t_range']
        # T = x['T']
        for n in range(x['n_state']):
            z_src = p["%s_z%s_%d" % (x['state'], x['src'], n)]
            z_snk = p["%s_z%s_%d" % (x['state'], x['snk'], n)]
            E_n = self.En(x, p, n)
            r += z_snk * z_src * (np.exp(-E_n*t) + np.exp(-E_n*(T-t)))
        return r

    def mres(self, x, p):
        ''' m_res = midpoint_pseudo / pseudo_pseudo
            we fit to a constant away from early/late time
            m_res = p[mres_l]
        '''
        return p[x['state']]*np.ones_like(x['t_range'])

    def two_h_ratio(self, x, p):
        ''' This model fits the g.s. as

            E_hh_0 = E1_0 + E2_0 + Delta_00

            and all other states independently of the single hadron energy levels.
        '''
        t = x['t_range']
        z_src = p["%s_z%s_%d" % (x['state'], x['src'], 0)]
        z_snk = p["%s_z%s_%d" % (x['state'], x['snk'], 0)]
        E0 = p[x['denom'][0]+'_E_0'] + p[x['denom']
                                         [1]+'_E_0'] + p['%s_dE_0_0' % x['state']]
        r = z_snk * z_src * np.exp(-E0*t)
        for n in range(1, x['n_state']):
            z_src = p["%s_z%s_%d" % (x['state'], x['src'], n)]
            z_snk = p["%s_z%s_%d" % (x['state'], x['snk'], n)]
            En = E0 + self.dEn(x, p, n)
            r += z_snk * z_src * np.exp(-En*t)
        return r

    def two_h_conspire(self, x, p):
        ''' This model assumes strong cancelation of e.s. in two and single hadron
            correlation functions.  Therefore, it uses many priors from the single
            hadron states as input.  For example, the correlator model looks like

            C(t) = A_0**2 * exp(-(2E_0+D_00)*t) + 2A0*A1*exp(-(E0+E1+D_01)*t) + A1**2*exp(-(2E1+D_11)*t)+...

            In addition, we have the option to add extra "elastic" excited states
        '''
        r = 0
        t = x['t_range']
        for n in range(x['n_state']):
            for m in range(x['n_state']):
                if n <= m:
                    En = self.En({'state': x['denom'][0]}, p, n)
                    Em = self.En({'state': x['denom'][1]}, p, m)
                    Dnm = p['%s_dE_%d_%d' % (x['state'], n, m)]
                    Anm = p['%s_A%s_%d_%d' %
                            (x['state'], x['snk']+x['src'], n, m,)]
                    #print(x['state'],n,m,En,Em,Dnm,Anm)
                    if n == m:
                        r += Anm * np.exp(-(En+Em+Dnm)*t)
                    elif n < m:
                        r += 2*Anm * np.exp(-(En+Em+Dnm)*t)
        if 'n_el' in x and x['n_el'] > 0:
            for n in range(1, x['n_el']+1):
                E_el_n = p['%s_E_0' % x['denom'][0]]
                E_el_n += p['%s_E_0' % x['denom'][0]]
                E_el_n += self.E_el_n(x, p, n)
                z_src = p["%s_z%s_el_%d" % (x['state'], x['src'], n)]
                z_snk = p["%s_z%s_el_%d" % (x['state'], x['snk'], n)]
                r += z_snk * z_src * np.exp(-E_el_n*t)
        return r

    def cosh_const(self, x, p):
        r = 0
        t = x['t_range']
        T = x['T']
        for n in range(x['n_state']):
            z_src = p["%s_z%s_%d" % (x['state'], x['src'], n)]
            z_snk = p["%s_z%s_%d" % (x['state'], x['snk'], n)]
            E_n = self.En(x, p, n)
            r += z_snk * z_src * np.exp(-E_n*t)
            r += z_snk * z_src * np.exp(-E_n*(T-t))
        # add the "const" term
        z_src = p["%s_z%s_c_0" % (x['state'], x['src'])]
        z_snk = p["%s_z%s_c_0" % (x['state'], x['snk'])]
        r += z_src * z_snk * np.exp(-p['D_E_0']*t) * np.exp(-p['pi_E_0']*(T-t))
        r += z_src * z_snk * np.exp(-p['pi_E_0']*t) * np.exp(-p['D_E_0']*(T-t))

        return r

    def eff_mass(self, x, p, ax, color='k', alpha=.4, t0=0, tau=1, denom_x=None):
        xp = dict(x)
        xm = dict(x)
        xp['t_range'] = x['t_range']+tau
        xm['t_range'] = x['t_range']-tau
        if x['type'] in ['cosh', 'cosh_const']:
            if x['type'] == 'cosh':
                corr = self.cosh(x, p)
                corr_p = self.cosh(xp, p)
                corr_m = self.cosh(xm, p)
            else:
                corr = self.cosh_const(x, p)
                corr_p = self.cosh_const(xp, p)
                corr_m = self.cosh_const(xm, p)
            meff = 1/tau * np.arccosh((corr_p + corr_m) / 2 / corr)
        elif 'exp' in x['type']:
            if denom_x:
                x0 = dict(denom_x[0])
                x1 = dict(denom_x[1])
                x0p = dict(x0)
                x1p = dict(x1)
                x0p['t_range'] = x0['t_range']+tau
                x1p['t_range'] = x1['t_range']+tau
                if x['type'] == 'exp_r':
                    corr = self.two_h_ratio(x, p) / self.exp(x0, p) / self.exp(x1, p)
                    corr_p = self.two_h_ratio(xp, p) / self.exp(x0p, p) / self.exp(x1p, p)
                elif x['type'] == 'exp_r_conspire':
                    corr = self.two_h_conspire(x, p) / self.exp(x0, p) / self.exp(x1, p)
                    corr_p = self.two_h_conspire(xp, p) / self.exp(x0p, p) / self.exp(x1p, p)
                elif x['type'] == 'exp_r_ind':
                    corr = self.exp(x, p)
                    corr_p = self.exp(xp, p)
            elif x['type'] == 'exp_r':
                corr = self.two_h_ratio(x, p)
                corr_p = self.two_h_ratio(xp, p)
            elif x['type'] == 'exp_r_conspire':
                corr = self.two_h_conspire(x, p)
                corr_p = self.two_h_conspire(xp, p)
            elif x['type'] in ['exp', 'exp_r_ind']:
                corr = self.exp(x, p)
                corr_p = self.exp(xp, p)
            elif x['type'] == 'exp_open':
                corr = self.exp_open(x, p)
                corr_p = self.exp_open(xp, p)
            elif x['type'] == 'exp_gs':
                corr = self.exp_gs(x, p)
                corr_p = self.exp_gs(xp, p)
            meff = 1/tau * np.log(corr / corr_p)
        elif x['type'] == 'mres':
            corr = self.mres(x,p)
            meff = corr
        else:
            sys.exit('unrecognized type, %s' % x['type'])

        m = np.array([k.mean for k in meff])
        dm = np.array([k.sdev for k in meff])
        ax.fill_between(x['t_range']+t0, m-dm, m+dm, color=color, alpha=alpha)


class FitCorr(object):
    def __init__(self):
        self.corr_functions = CorrFunction()

    def get_fit(self,priors,states,x,y):
        p0 = {k: v.mean for (k, v) in priors.items()}
        # only pass x for states in fit
        x_fit = dict()
        y_fit = dict()
        fit_lst = [k for k in x if k.split('_')[0] in states]
        print(fit_lst)
        for k in fit_lst:
            if 'mres' not in k:
                x_fit[k] = x[k]
                y_fit[k] = y[k]
            else:
                k_res = k.split('_')[0]
                if k_res not in x_fit:
                    x_fit[k_res] = x[k]
                    y_fit[k_res] = y[k_res]
        return fit_lst,p0,x_fit,y_fit

    def priors(self, prior):
        '''
        only keep priors that are used in fit
        truncate based on n_states
        '''
        p = dict()
        for q in prior:
            if 'log' in q:
                if int(q.split('_')[-1].split(')')[0]) < self.nstates:
                    p[q] = prior[q]
            else:
                if int(q.split('_')[-1]) < self.nstates:
                    p[q] = prior[q]
        return p

    def fit_function(self, x, p):
        r = dict()
        for k in x:
            if x[k]['type'] in dir(self.corr_functions):
                r[k] = getattr(self.corr_functions, x[k]['type'])(x[k],p)
            # NOTE: we should move exp_r and exp_r_conspire into corr_functions
            elif x[k]['type'] == 'exp_r':
                sp = k.split('_')[-1]
                r[k] = self.corr_functions.two_h_ratio(x[k], p)
                r[k] = r[k] / \
                    self.corr_functions.exp(x[x[k]['denom'][0]+'_'+sp], p)
                r[k] = r[k] / \
                    self.corr_functions.exp(x[x[k]['denom'][1]+'_'+sp], p)
            elif x[k]['type'] == 'exp_r_conspire':
                sp = k.split('_')[-1]
                r[k] = self.corr_functions.two_h_conspire(x[k], p)
                r[k] = r[k] / \
                    self.corr_functions.exp(x[x[k]['denom'][0]+'_'+sp], p)
                r[k] = r[k] / \
                    self.corr_functions.exp(x[x[k]['denom'][1]+'_'+sp], p)
            else:
                sys.exit('Unrecognized fit model, %s' %x[k]['type'])
        return r

    





