import matplotlib.pyplot as plt
import numpy as np
import os
import copy 
import gvar as gv
import sys
import lsqfit

import fitter.corr_functions as cf
fit_funcs = cf.FitCorr()

def get_nucleon_effective_mass(corr_gv=None, dt=None):
        if corr_gv is None:
            corr_gv = corr_gv

        # If still empty, return nothing
        if corr_gv is None:
            return None

        if dt is None:
            dt = 1
        return {key : 1/dt * np.log(corr_gv[key] / np.roll(corr_gv[key], -1))
                for key in corr_gv.keys()}
def get_nucleon_effective_wf(corr_gv=None, t=None, dt=None):
        if corr_gv is None:
            corr_gv = corr_gv

        # If still empty, return nothing
        if corr_gv is None:
            return None

        effective_mass = get_nucleon_effective_mass(corr_gv, dt)
        if t is None:
            t = {key : np.arange(len(corr_gv[key])) for key in corr_gv.keys()}
        else:
            t = {key : t for key in corr_gv.keys()}

        return {key : np.exp(effective_mass[key]*t[key]) * corr_gv[key]
                for key in corr_gv.keys()}

def plot_effective_wf(corr_gv=None, t_plot_min=None,
                           t_plot_max=None, show_plot=False, show_fit=True):
        if t_plot_min is None:
            t_plot_min = t_min
        if t_plot_max is None:
            t_plot_max = t_max

        if corr_gv is None:
            corr_gv = corr_gv

        # If fit_ensemble doesn't have a default a nucleon correlator,
        # it's impossible to make this plot
        if corr_gv is None:
            return None

        colors = np.array(['magenta', 'cyan', 'yellow'])
        t = {}
        A_eff = {}
        for j, key in enumerate(sorted(corr_gv.keys())):

            plt.subplot(int(str(21)+str(j+1)))

            t[key] = np.arange(t_plot_min, t_plot_max)
            A_eff[key] = get_nucleon_effective_wf(corr_gv)[key][t_plot_min:t_plot_max]

            pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
            lower_quantile = np.nanpercentile(gv.mean(A_eff[key]), 25)
            upper_quantile = np.nanpercentile(gv.mean(A_eff[key]), 75)
            delta_quantile = upper_quantile - lower_quantile
            plt.errorbar(x=t[key], y=gv.mean(A_eff[key]), xerr=0.0, yerr=gv.sdev(A_eff[key]),
                fmt='o', color=colors[j%len(colors)], capsize=5.0, capthick=2.0, alpha=0.6, elinewidth=5.0, label=key)


            plt.legend()
            plt.grid(True)
            plt.ylabel('$A^{eff}$', fontsize = 24)
            plt.xlim(t_plot_min-0.5, t_plot_max-.5)
            plt.ylim(lower_quantile - 0.5*delta_quantile,
                     upper_quantile + 0.5*delta_quantile)

        # if show_fit:
        #     t = np.linspace(t_plot_min-2, t_plot_max+2)
        #     dt = (t[-1] - t[0])/(len(t) - 1)
        #     fit_data_gv = self._generate_data_from_fit(model_type="corr", t=t)
        #     #t = t[1:-1]
        #     for j, key in enumerate(sorted(fit_data_gv.keys())):
        #         plt.subplot(int(str(21)+str(j+1)))
        #         if j == 0:
        #             plt.title("Best fit for $N_{states} = $%s" %(self.n_states['corr']), fontsize = 24)

        #         A_eff_fit = self.get_nucleon_effective_wf(fit_data_gv, t, dt)[key][1:-1]

        #         plt.plot(t[1:-1], pm(A_eff_fit, 0), '--', color=colors[j%len(colors)])
        #         plt.plot(t[1:-1], pm(A_eff_fit, 1), t[1:-1], pm(A_eff_fit, -1), color=colors[j%len(colors)])
        #         plt.fill_between(t[1:-1], pm(A_eff_fit, -1), pm(A_eff_fit, 1),
        #                          facecolor=colors[j%len(colors)], alpha = 0.10, rasterized=True)
        #         plt.xlim(t_plot_min-0.5, t_plot_max-.5)

        plt.xlabel('$t$', fontsize = 24)
        fig = plt.gcf()
        if show_plot == True: plt.show()
        else: plt.close()

        return fig

def get_naive_effective_g00(fh_num_gv, corr_gv):
    fh_ratio_gv = {key : fh_num_gv[key] / corr_gv[key] for key in fh_num_gv.keys()}
        #return fh_num_gv
        #fh_ratio_gv = {key : fh_num_gv[key] for key in fh_num_gv.keys()}
    print(fh_ratio_gv,"hi")
    return {key : (np.roll(fh_ratio_gv[key], -1) - fh_ratio_gv[key])/1 for key in fh_ratio_gv.keys()}

def plot_naive_effective_g00(fh_num_gv, corr_gv,
                    t_plot_min, t_plot_max, show_plot=True,show_fit=False,
                    observable=None):
    
    if observable is None:
        return None
    elif observable == 'gA':
        ylabel = r'$g^{eff}_A}$'
    elif observable == 'gV':
        ylabel = r'$g^{eff}_V}$'
    colors = np.array(['red', 'blue', 'yellow'])
    t = np.arange(t_plot_min, t_plot_max)
    effective_g00 = get_naive_effective_g00(fh_num_gv, corr_gv)

    for j, key in enumerate(effective_g00.keys()):
        y = gv.mean(effective_g00[key])[t]
        y_err = gv.sdev(effective_g00[key])[t]
        tp = t - 0.1 + j*0.2
        plt.errorbar(x=tp, y=y, xerr=0.0, yerr=y_err, fmt='o', capsize=5.0,
            color = colors[j], capthick=2.0, alpha=0.6, elinewidth=5.0, label=key)

    # if show_fit:
    #     t = np.linspace(t_plot_min-2, t_plot_max+2)
    #     dt = (t[-1] - t[0])/(len(t) - 1)

    #     fit__num_gv = self._generate_data_from_fit(model_type=model_type, t=t)
    #     fit_nucleon_gv = self._generate_data_from_fit(model_type="corr", t=t)


    #     for j, key in enumerate(fit_fh_num_gv.keys()):
    #         eff_g00_fit = self.get_effective_g00(fit_fh_num_gv, fit_nucleon_gv, dt)[key][1:-1]

    #         pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
    #         plt.plot(t[1:-1], pm(eff_g00_fit, 0), '--', color=colors[j%len(colors)])
    #         plt.plot(t[1:-1], pm(eff_g00_fit, 1), t[1:-1], pm(eff_g00_fit, -1), color=colors[j%len(colors)])

    #         plt.fill_between(t[1:-1], pm(eff_g00_fit, -1), pm(eff_g00_fit, 1),
    #                             facecolor=colors[j%len(colors)], alpha = 0.10, rasterized=True)
    #     plt.title("Best fit for $N_{states} = $%s" %(self.n_states[model_type]), fontsize = 24)

    # plt.ylim(np.max([lower_quantile - 0.5*delta_quantile, 0]),
    #             upper_quantile + 0.5*delta_quantile)
    # plt.xlim(t_plot_min-0.5, t_plot_max-.5)


    plt.legend()
    plt.grid(True)
    #plt.ylim(0.95, 1.35)
    #plt.ylim(1.1, 1.7)
    plt.xlabel('$t$', fontsize = 24)
    plt.ylabel(ylabel, fontsize = 24)
    fig = plt.gcf()
    if show_plot == True: plt.show()
    else: plt.close()

    return fig
def plot_effective_mass(correlators_gv,fit=None, t_plot_min = None, t_plot_max = 18,show_plot=True,show_fit=None):
    if t_plot_min == None: t_plot_min = 0
    if t_plot_max == None: t_plot_max = correlators_gv[correlators_gv.keys()[0]].shape[0] - 1

    tau = +2
    effective_mass = gv.BufferDict()
    for key in correlators_gv.keys():
        effective_mass[key] = (1.0/tau) * np.log(correlators_gv[key] / np.roll(correlators_gv[key], -tau))
    t = np.arange(t_plot_min, t_plot_max)
    for j, key in enumerate(sorted(correlators_gv.keys())):
        y = gv.mean(effective_mass[key])[t]
        y_err = gv.sdev(effective_mass[key])[t]
        
        tp = t + 0.1 - j*0.2
        plt.errorbar(tp, y, xerr = 0.0, yerr=y_err, fmt='o', capsize=5.0,capthick=2.0, alpha=0.6, elinewidth=5.0, label=key)
    if show_fit:
            t = np.linspace(t_plot_min-2, t_plot_max+2)
            dt = (t[-1] - t[0])/(len(t) - 1)
            fit_data_gv = fit #self._generate_data_from_fit(model_type="corr", t=t)

            for j, key in enumerate(fit_data_gv.keys()):
                eff_mass_fit = self.get_nucleon_effective_mass(fit_data_gv, dt)[key][1:-1]

                pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
                plt.plot(t[1:-1], pm(eff_mass_fit, 0), '--', color=colors[j%len(colors)])
                plt.plot(t[1:-1], pm(eff_mass_fit, 1), t[1:-1], pm(eff_mass_fit, -1), color=colors[j%len(colors)])
                plt.fill_between(t[1:-1], pm(eff_mass_fit, -1), pm(eff_mass_fit, 1),
                                 facecolor=colors[j%len(colors)], alpha = 0.10, rasterized=True)
            plt.title("Best fit for $N_{states} = $%s" %(self.n_states['corr']), fontsize = 24)


    # Label dirac/smeared data
    plt.xlim(t_plot_min-0.5, t_plot_max-.5)
    plt.legend()
    plt.grid(True)
    plt.ylim(0.0, 0.75)
    plt.xlabel('$t$', fontsize = 24)
    plt.ylabel('$m_{eff}$', fontsize = 24)
    fig = plt.gcf()
    if show_plot == True: plt.show()
    else: plt.close()
    return fig

def plot_correlators(correlators_gv, show_plot=True,t_plot_min = None, t_plot_max = None):
    if t_plot_min == None: t_plot_min = 0
    if t_plot_max == None: t_plot_max = correlators_gv[correlators_gv.keys()[0]].shape[0] - 1

    x = range(t_plot_min, t_plot_max)
    for j, key in enumerate(sorted(correlators_gv.keys())):
        y = gv.mean(correlators_gv[key])[x]
        y_err = gv.sdev(correlators_gv[key])[x]
        
        plt.errorbar(x, y, xerr = 0.0, yerr=y_err, fmt='o', capsize=5.0,capthick=2.0, alpha=0.6, elinewidth=5.0, label=key)


    # Label dirac/smeared data
    plt.legend()
    plt.grid(True)
    plt.xlabel('$t$', fontsize = 24)
    plt.ylabel('$R(t)$ (FH-ratio)', fontsize = 24)
    fit = plt.gcf()
    if show_plot == True: plt.show()
    else: plt.close()
    return fit

def subplots(*args,**kwargs):
    return plt.subplots(*args,**kwargs)

def plot_correlator_summary(correlators_gv,axarr=None, a_fm=None, avg=False, even_odd=False, tmax=np.inf, label=None):
    if axarr is None:
        _, axarr = plt.subplots(ncols=3, figsize=(15, 5))
    ax1, ax2, ax3 = axarr

    
    plot_meff = plot_effective_mass(correlators_gv)

    plot_corr = plot_correlators(correlators_gv)
    # plot_meff(ax=ax2, a_fm=a_fm, fmt='o', avg=avg, tmax=tmax, label=label)
    # self.plot_n2s(ax=ax3, label=label)

    ax1.set_title("Correlator C(t)")
    ax2.set_title("Effective mass")
    ax3.set_title("Noise-to-signal [%]")
    return axarr





class eff_plots():
    def __init__(self):
        self.ax_meff = {}
        self.ax_zeff = {}
        self.ax_r = {}

    def make_eff_plots(self,states,fp,x_fit,priors,gv_data,fit, scale,show_fit,save_figs,x):
        ''' Make dictionary of effective mass, and effective overlap factor plots

            Attributes:
            - states: list of states to make effective plots for
            - corr_lst: 
            - priors: 
        '''
        
        for k in ['proton','pion']:
            
            clrs = fp.corr_lst[k]['colors']
            if 't0' in fp.corr_lst[k]:
                t0 = fp.corr_lst[k]['t0']
            else:
                t0 = 0
        # m_eff
            fig = plt.figure('m_'+k, figsize=(7, 4))
            if scale:
                self.ax_meff[k] = plt.axes([0.15, 0.15, 0.74, 0.84])
            else:
                self.ax_meff[k] = plt.axes([0.15, 0.15, 0.84, 0.84])
            
            if fp.corr_lst[k]['type'] not in ['exp_r', 'exp_r_conspire']:
                if fp.corr_lst[k]['type'] == 'mres':
                    p = priors[k]
                else:
                    print(priors)
                    p = priors[k+'_E_0']
            else:
                d1, d2 = fp.corr_lst[k]['denom']
                p = priors[d1+'_E_0'] + priors[d2+'_E_0'] + priors[k+'_dE_0_0']
            self.ax_meff[k].axhspan(p.mean-p.sdev, p.mean
                                + p.sdev, color='k', alpha=.2)
            if 'mres' not in k:
                plot_eff(self.ax_meff[k], gv_data, k,
                                mtype=fp.corr_lst[k]['type'], colors=clrs, offset=t0)
            else:
                plot_mres(self.ax_meff[k], gv_data, k,
                                mtype=fp.corr_lst[k]['type'], colors=clrs, offset=t0)
            self.ax_meff[k].set_xlim(fp.corr_lst[k]['xlim'])
            self.ax_meff[k].set_ylim(fp.corr_lst[k]['ylim'])
            self.ax_meff[k].set_xlabel(r'$t/a$', fontsize=20)
            self.ax_meff[k].set_ylabel(
                r'$m_{\rm eff}^{\rm %s}(t)$' % k, fontsize=20)
            self.ax_meff[k].legend(fontsize=20)
            if 'mres' not in k:
                if 'denom' in fp.corr_lst[k]:
                    fig = plt.figure('r_'+k, figsize=(7, 4))
                    self.ax_r[k] = plt.axes([0.15, 0.15, 0.84, 0.84])
                    if fp.corr_lst[k]['type'] == 'exp_r_ind':
                        p = priors[k+'_E_0']
                    else:
                        p = priors[k+'_dE_0_0']
                    self.ax_r[k].axhspan(p.mean-p.sdev, p.mean
                                    + p.sdev, color='k', alpha=.2)
                    eff_plots.plot_eff(self.ax_r[k], gv_data, k, mtype=fp.corr_lst[k]['type'])
                    #self.ax_r[k].set_ylim(fp.corr_lst[k]['ylim'])
                    self.ax_r[k].set_ylim(-.02, .02)
                    self.ax_r[k].set_xlabel(r'$t/a$', fontsize=20)
                    self.ax_r[k].set_ylabel(
                        r'$m_{\rm eff}^{\rm %s}(t)$' % k, fontsize=20)
                    self.ax_r[k].legend(fontsize=20)

                # z_eff
                fig = plt.figure('z_'+k, figsize=(7, 4))
                self.ax_zeff[k] = plt.axes([0.15, 0.15, 0.84, 0.82]) # this raised deprecation warning; adding axes to same arg as prev axes
                snksrc = {'snks': fp.corr_lst[k]['snks'],
                            'srcs': fp.corr_lst[k]['srcs']}
                mtype = fp.corr_lst[k]['type']
                ztype = fp.corr_lst[k]['ztype']
                # this zeff prior is not generic yet
                #for z in fp.corr_lst[k]['snks']:
                #    p = priors[k+'_z'+z+'_0']
                #    self.ax_zeff[k].axhspan(p.mean-p.sdev, p.mean+p.sdev, color='k',alpha=.2)
                plot_zeff(self.ax_zeff[k], gv_data, k, ztype=ztype,
                                mtype=mtype, snksrc=snksrc, colors=clrs)
                self.ax_zeff[k].set_xlim(fp.corr_lst[k]['xlim'])
                self.ax_zeff[k].set_ylim(fp.corr_lst[k]['z_ylim'])
                self.ax_zeff[k].set_xlabel(r'$t/a$', fontsize=20)
                self.ax_zeff[k].set_ylabel(
                    r'$z_{\rm eff}^{\rm %s}(t)$' % k, fontsize=20)
                self.ax_zeff[k].legend(fontsize=20, loc=1)

        # Overlay fit on eff mass plot 
        if show_fit:
            x = fp.x
            x_fit = dict()
            fit_lst = [j for j in x if j.split('_')[0] in states]
            for j in fit_lst:
                x_fit[j] = x[j]

            x_plot = copy.deepcopy(x_fit)
            for k in x_plot:
                sp = k.split('_')[-1]
                ax = self.ax_meff[k.split('_')[0]]
                if 't0' in x_fit[k]:
                    t0 = x_fit[k]['t0']
                else:
                    t0 = 0
                x_plot[k]['t_range'] = np.arange(
                    x[k]['t_range'][0], x[k]['t_range'][-1]+.1, .1)
                fit_funcs.corr_functions.eff_mass(
                    x_plot[k], fit.p, ax, t0=t0, color=x_plot[k]['color'])
                x_plot[k]['t_range'] = np.arange(
                    x[k]['t_range'][-1]+.5, x[k]['t_range'][-1]+20.1, .1)
                fit_funcs.corr_functions.eff_mass(
                    x_plot[k], fit.p, ax, t0=t0, color='k', alpha=.1)
                if 'exp_r' in x_plot[k]['type']:
                    ax = self.ax_r[k.split('_')[0]]

                    if x_plot[k]['type'] in ['exp_r', 'exp_r_conspire']:
                        x_plot[k]['t_range'] = np.arange(
                            x[k]['t_range'][0], x[k]['t_range'][-1]+.1, .1)
                        x_plot[x_plot[k]['denom'][0]+'_'
                                + sp]['t_range'] = x_plot[k]['t_range']
                        x_plot[x_plot[k]['denom'][1]+'_'
                                + sp]['t_range'] = x_plot[k]['t_range']
                        d_x = [x_plot[x_plot[k]['denom'][0]+'_'+sp],
                                x_plot[x_plot[k]['denom'][1]+'_'+sp]]
                        fit_funcs.corr_functions.eff_mass(
                            x_plot[k], fit.p, ax, color=x_plot[k]['color'], denom_x=d_x)
                        x_plot[k]['t_range'] = np.arange(
                            x[k]['t_range'][-1]+.5, x[k]['t_range'][-1]+20.1, .1)
                        x_plot[x_plot[k]['denom'][0]+'_'
                                + sp]['t_range'] = x_plot[k]['t_range']
                        x_plot[x_plot[k]['denom'][1]+'_'
                                + sp]['t_range'] = x_plot[k]['t_range']
                        d_x = [x_plot[x_plot[k]['denom'][0]+'_'+sp],
                                x_plot[x_plot[k]['denom'][1]+'_'+sp]]
                        fit_funcs.corr_functions.eff_mass(
                            x_plot[k], fit.p, ax, color='k', alpha=.1, denom_x=d_x)
                    else:
                        x_plot[k]['t_range'] = np.arange(
                            x[k]['t_range'][0], x[k]['t_range'][-1]+.1, .1)
                        fit_funcs.corr_functions.eff_mass(
                            x_plot[k], fit.p, ax, color=x_plot[k]['color'])
                        x_plot[k]['t_range'] = np.arange(
                            x[k]['t_range'][-1]+.5, x[k]['t_range'][-1]+20.1, .1)
                        fit_funcs.corr_functions.eff_mass(
                            x_plot[k], fit.p, ax, color='k', alpha=.1)

        if scale:

            for k in self.ax_meff:
                s, units = float(scale[0]), scale[1]
                axr = self.ax_meff[k].twinx()
                print(k, self.ax_meff[k].get_ylim())
                print(self.ax_meff[k].get_yticks())
                axr.set_ylim(self.ax_meff[k].get_ylim()[0]*s,
                            self.ax_meff[k].get_ylim()[1]*s)
                axr.set_yticks([s*t for t in self.ax_meff[k].get_yticks()[:-1]])
                if units in ['GeV', 'gev']:
                    axr.set_yticklabels(["%.2f" % t for t in axr.get_yticks()])
                else:
                    axr.set_yticklabels(["%.0f" % t for t in axr.get_yticks()])
                axr.set_ylabel(r'$m_{\rm eff}(t) / {\rm %s}$' %
                            (units), fontsize=20)

        if save_figs:
            for k in states:
                n_s = str(fp.corr_lst[k]['n_state'])
                plt.figure('m_'+k)
                plt.savefig('figures/'+k+'_meff_ns'
                            + n_s+'.pdf', transparent=True)
                plt.figure('z_'+k)
                plt.savefig('figures/'+k+'_zeff_ns'
                            + n_s+'.pdf', transparent=True)
                if 'exp_r' in fp.corr_lst[k]['type']:
                    plt.figure('r_'+k)
                    plt.savefig('figures/'+k+'_ratio_meff_ns'
                                + n_s+'.pdf', transparent=True)

#for k in states:


def effective_mass(gvdata, mtype='exp', tau=1):
    ''' Create effective mass data from gvar of the correlation function
        versus time.
        gvdata = array of gvar data of correlation function
        mtype  = type of effective mass: exp, cosh, cosh_costant, ...
        tau    = shift variable for making effective mass
    '''
    meff = []
    if 'exp' in mtype:
        meff = 1./tau * np.log(gvdata / np.roll(gvdata, -tau))
    elif mtype == 'cosh':
        meff = 1./tau * \
            np.arccosh((np.roll(gvdata, -tau)+np.roll(gvdata, tau))/2/gvdata)
    # chop off last time slice for wrap-around effects
    return meff[:-1]


def plot_eff(ax, dsets, key, mtype='exp', tau=1, colors=None, offset=0, denom_key=None):
    lst = [k for k in dsets if key in k]
    for k in lst:
        data = dsets[k]
        if denom_key:
            for d in denom_key:
                data = data / dsets[k.replace(key, d)]
        lbl = k.split('_')[1]
        eff = effective_mass(data, mtype=mtype, tau=tau)
        x = np.arange(eff.shape[0]) + offset
        m = [k.mean for k in eff]
        dm = [k.sdev for k in eff]
        if colors is not None:
            ax.errorbar(x, m, yerr=dm, linestyle='None', marker='o',
                        color=colors[lbl], mfc='None', label=lbl)
        else:
            ax.errorbar(x, m, yerr=dm, linestyle='None', marker='o',
                        mfc='None', label=label)

    
def plot_zeff(ax, dsets, key, ztype='A_snk,src', snksrc=None, mtype='exp', tau=1, colors=None):
    lst = [k for k in dsets if key in k]
    if ztype == 'A_snk,src':
        for k in lst:
            lbl = k.split('_')[-1]
            eff = effective_mass(dsets[k], mtype=mtype, tau=tau)
            t = np.arange(eff.shape[0])
            if 'exp' in mtype:
                zeff = np.exp(eff * t) * dsets[k][:-1]
            elif mtype == 'cosh':
                zeff = dsets[k][:-1] / \
                    (np.exp(-eff * t) + np.exp(-eff * (len(t)-t)))
            z = [k.mean for k in zeff]
            dz = [k.sdev for k in zeff]
            if colors is not None:
                ax.errorbar(t, z, yerr=dz, linestyle='None', marker='o',
                            color=colors[lbl], mfc='None', label=lbl)
            else:
                ax.errorbar(t, z, yerr=dz, linestyle='None', marker='o',
                            mfc='None', label=lbl)
    elif ztype == 'z_snk z_src':
        for snk in snksrc['snks']:
            src = snksrc['srcs'][0]  # assume single source for now
            k = key+'_'+snk+src
            lbl = r'$z_{\rm %s}$' % snk
            eff = effective_mass(dsets[k], mtype=mtype, tau=tau)
            t = np.arange(eff.shape[0])

            if 'exp' in mtype:
                # we have to cut the final t-slice
                zeff = np.exp(eff * t) * dsets[k][:-1]
            elif mtype == 'cosh':
                zeff = dsets[k][:-1] / \
                    (np.exp(-eff * t) + np.exp(-eff * (len(t)-t)))
            # we don't want the last entry (wrap around effects)
            #zeff = zeff[:-1]
            if snk == src:
                z = [d.mean for d in np.sqrt(zeff)]
                dz = [d.sdev for d in np.sqrt(zeff)]
            else:
                k2 = key+'_'+src+src
                eff2 = effective_mass(dsets[k2], mtype=mtype, tau=tau)
                if 'exp' in mtype:
                    zeff2 = np.exp(eff*t) * dsets[k2][:-1]
                elif mtype == 'cosh':
                    zeff2 = dsets[k2][:-1] / \
                        (np.exp(-eff * t) + np.exp(-eff * (len(t)-t)))
                #zeff2 = zeff2[:-1]
                z = [d.mean for d in zeff / np.sqrt(zeff2)]
                dz = [d.sdev for d in zeff / np.sqrt(zeff2)]

            if colors is not None:
                ax.errorbar(t, z, yerr=dz, linestyle='None', marker='o',
                            color=colors[snk+src], mfc='None', label=lbl)
            else:
                ax.errorbar(t, z, yerr=dz, linestyle='None', marker='o',
                            mfc='None', label=lbl)

def plot_mres(ax, dsets, key, mtype='exp', tau=1, colors=None, offset=0, denom_key=None):
    lst = [k for k in dsets if key in k]
    mres = lst[0].split('_')[0]
    data = dsets[mres+'_MP'] / dsets[mres+'_PP']
    lbl  = mres
    x  = np.arange(data.shape[0]) + offset
    m  = [k.mean for k in data]
    dm = [k.sdev for k in data]
    if colors is not None:
        ax.errorbar(x, m, yerr=dm, linestyle='None', marker='o',
                    color=colors, mfc='None', label=lbl)
    else:
        ax.errorbar(x, m, yerr=dm, linestyle='None', marker='o',
                    mfc='None', label=label)


def make_stability_plot(states,x,fp,gv_data,stability,priors,scale,
                        svd_test,data_cfg,n_states,svd_nbs,es_stability,save_figs):
    
    #x,y,n_states,priors = make_fit_params(fp, states, gv_data)
    p = copy.deepcopy(fp.priors)
    for state in stability:
        if 'mres' in state:
            sys.exit("we don't support m_res stability with this function")
        
        print(state)
        if 't_sweep' in fp.corr_lst[state]:
            tmin = fp.corr_lst[state]['t_sweep']
        else:
            tmin = range(2, x[state]['t_range'][-1])
        if 'n_sweep' in fp.corr_lst[state]:
            n_states = fp.corr_lst[state]['n_sweep']
        else:
            n_states = range(1, 6)
        tn_opt = (fp.corr_lst[state]['t_range'][0],
                    fp.corr_lst[state]['n_state'])
        x_tmp = dict()
        for k in [kk for kk in x if kk.split('_')[0] == state]:
            x_tmp[k] = x[k].copy()

        fits = {}
        for ti in tmin:
            for k in x_tmp:
                x_tmp[k]['t_range'] = np.arange(ti, x[k]['t_range'][-1]+1)

            y_tmp = {k: v[x_tmp[k]['t_range']]
                    for (k, v) in gv_data.items() if k in x_tmp}
            if svd_test:
                y_chop = dict()
                for d in y_tmp:
                    if d in x_tmp:
                        y_chop[d] = data_cfg[d][:,x_tmp[d]['t_range']]
                s = gv.dataset.svd_diagnosis(y_chop, nbstrap=svd_nbs)
                svdcut = s.svdcut
                has_svd = True
            for k in x_tmp:
                print(k)
                if k.split('_')[0] not in states:
                    y_tmp.pop(k)
            if ti == tmin[0]:
                print([k for k in y_tmp])
            for ns in n_states:
                xx = copy.deepcopy(x_tmp)
                for k in xx:
                    ''' NOTE  - we are chaning n_s for pi, D and Dpi all together '''
                    xx[k]['n_state'] = ns
                #fit_funcs = cf.FitCorr()
                p_sweep = {}

                for k in p:
                    if 'mres' not in k:
                        if int(k.split('_')[-1].split(')')[0]) < ns:
                            p_sweep[k] = gv.gvar(p[k].mean,p[k].sdev)

                p0 = {k: v.mean for (k, v) in priors.items()}
                #print('t_min = %d  ns = %d' %(ti,ns))
                sys.stdout.write(
                    'sweeping t_min = %d n_s = %d\r' % (ti, ns))
                sys.stdout.flush()
                
                if has_svd:
                    f_tmp = lsqfit.nonlinear_fit(data=(xx, y_tmp),
                                                    prior=p_sweep, p0=p0,
                                                    fcn=fit_funcs.fit_function, svdcut=svdcut)
                else:
                    f_tmp = lsqfit.nonlinear_fit(data=(xx, y_tmp),
                                                    prior=p_sweep, p0=p0,
                                                    fcn=fit_funcs.fit_function)
                fits[(ti, ns)] = f_tmp
                #print(fits.keys())
                

        ylim = None
        #print(x_tmp[list(x_tmp.keys())[0]])
        if 'eff_ylim' in x_tmp[list(x_tmp.keys())[0]]:
            ylim = x_tmp[k]['eff_ylim']
        #import IPython; IPython.embed()
        plot_stability(fits=fits, tmin=tmin, n_states=n_states, tn_opt=tn_opt,
                            state=state, ylim=ylim, save=save_figs)
        if es_stability:
            for i_n in range(1, n_states[-1]):
                plot_stability(fits, tmin, n_states, tn_opt, state,
                                ylim=ylim, save=save_figs, n_plot=i_n, scale=scale)
    
    
    def plot_mres(self,ax, dsets, key, mtype='exp', tau=1, colors=None, offset=0, denom_key=None):
        lst = [k for k in dsets if key in k]
        mres = lst[0].split('_')[0]
        data = dsets[mres+'_MP'] / dsets[mres+'_PP']
        lbl  = mres
        x  = np.arange(data.shape[0]) + offset
        m  = [k.mean for k in data]
        dm = [k.sdev for k in data]
        if colors is not None:
            ax.errorbar(x, m, yerr=dm, linestyle='None', marker='o',
                        color=colors, mfc='None', label=lbl)
        else:
            ax.errorbar(x, m, yerr=dm, linestyle='None', marker='o',
                        mfc='None', label=label)


def plot_stability(fits, tmin, n_states, tn_opt, state,
                ylim=None, diff=False, save=True, n_plot=0, scale=None):
    fs = 20
    fs_ns = 16
    nn = str(n_plot)
    if n_plot == 0:
        kk = 'E'
    else:
        kk = 'dE'

    markers = dict()
    colors = dict()
    markers[1] = 'o'
    colors[1] = 'r'
    markers[2] = 's'
    colors[2] = 'orange'
    markers[3] = '^'
    colors[3] = 'g'
    markers[4] = 'D'
    colors[4] = 'b'
    markers[5] = '*'
    colors[5] = 'darkviolet'
    markers[6] = 'h'
    colors[6] = 'violet'
    markers[7] = 'X'
    colors[7] = 'gold'
    markers[8] = '8'
    colors[8] = 'darkred'

    fig = plt.figure(state+'_E_'+nn+'_stability', figsize=(7, 4))
    if scale:
        ax_e0 = plt.axes([0.14, 0.42, 0.75, 0.57])
        ax_Q = plt.axes([0.14, 0.27, 0.75, 0.15])
        ax_w = plt.axes([0.14, 0.12, 0.75, 0.15])
        s = float(scale[0])
        units = scale[1]
    else:
        ax_e0 = plt.axes([0.14, 0.42, 0.85, 0.57])
        ax_Q = plt.axes([0.14, 0.27, 0.85, 0.15])
        ax_w = plt.axes([0.14, 0.12, 0.85, 0.15])

    for ti in tmin:
        logGBF = []
        for ns in n_states:
            if n_plot <= ns-1:
                if ti == tmin[0]:
                    if ns == n_states[0]:
                        lbl = r'$n_s=%d$' % ns
                    else:
                        lbl = r'$%d$' % ns
                else:
                    lbl = ''
                if ns == tn_opt[1] and ti == tn_opt[0]:
                    mfc = 'k'
                    color = colors[ns]
                else:
                    mfc = 'None'
                    color = colors[ns]
                #print(fits[(ti, ns)].logGBF)
                logGBF.append(fits[(ti, ns)].logGBF)
                if diff:
                    e0 = fits[(ti, ns)].p[state+'_'+kk+'_'+nn] - fits[(ti, ns)
                                                                    ].p['pi_'+kk+'_'+nn] - fits[(ti, ns)].p['D_'+kk+'_'+nn]
                else:
                    e0 = fits[(ti, ns)].p[state+'_E_0']
                    if n_plot > 0:
                        for i_n in range(1, n_plot+1):
                            e0 += fits[(ti, ns)].p[state+'_'+kk+'_'+str(i_n)]
                            print(e0)
                ax_e0.errorbar(ti + 0.1*(ns-5), e0.mean, yerr=e0.sdev,
                            marker=markers[ns], color=color, mfc=mfc, linestyle='None', label=lbl)
                ax_Q.plot(ti + 0.1*(ns-5), fits[(ti, ns)].Q,
                        marker=markers[ns], color=color, mfc=mfc, linestyle='None')
        logGBF = np.array(logGBF)
        logGBF = logGBF - logGBF.max()
        weights = np.exp(logGBF)
        weights = weights / weights.sum()
        for i_s, ns in enumerate(n_states):
            if i_s < len(weights):
                if ns == tn_opt[1] and ti == tn_opt[0]:
                    mfc = 'k'
                    color = colors[ns]
                else:
                    mfc = 'None'
                    color = colors[ns]
                ax_w.plot(ti + 0.1*(ns-5), weights[i_s],
                        marker=markers[ns], color=color, mfc=mfc, linestyle='None')
    priors = fits[(tn_opt[0], tn_opt[1])].prior
    if diff:
        ax_e0.set_ylabel(r'$E^{\rm %s}_%s - M_D - M_\pi$' %
                        (state.replace('_', '\_'), nn), fontsize=fs)
        e0_opt = fits[(tn_opt[0], tn_opt[1])].p[state+'_'+kk+'_'+nn]
        e0_opt += -fits[(tn_opt[0], tn_opt[1])].p['pi_'+kk+'_'+nn] - \
            fits[(tn_opt[0], tn_opt[1])].p['D_'+kk+'_'+nn]
        ax_e0.set_ylim(-2*abs(e0.mean), 2*abs(e0.mean))
        ax_e0.axhline(0)
    else:
        ax_e0.set_ylabel(r'$E^{\rm %s}_%s$' % (state, nn), fontsize=fs)
        e0_opt = fits[(tn_opt[0], tn_opt[1])].p[state+'_E_0']
        if n_plot == 0:
            e0_prior = fits[(tn_opt[0], tn_opt[1])].prior[state+'_E_0']
            ax_e0.set_ylim(
                1.001*(e0_prior.mean-e0_prior.sdev),
                0.999*(e0_prior.mean+e0_prior.sdev))
        elif n_plot > 0 and n_plot < tn_opt[1]:
            e0_prior = fits[(tn_opt[0], tn_opt[1])].p[state+'_E_0']
            for i_n in range(1, n_plot):
                e0_prior += fits[(tn_opt[0], tn_opt[1])
                                ].p[state+'_'+kk+'_'+str(i_n)]
                e0_opt += fits[(tn_opt[0], tn_opt[1])
                            ].p[state+'_'+kk+'_'+str(i_n)]
            e0_prior += fits[(tn_opt[0], tn_opt[1])
                            ].prior[state+'_'+kk+'_'+str(n_plot)]
            e0_opt += fits[(tn_opt[0], tn_opt[1])].p[state
                                                    + '_'+kk+'_'+str(n_plot)]
        else:
            e0_prior = fits[(tn_opt[0], tn_opt[1])].p[state+'_E_0']
        if n_plot > 0 and n_plot < tn_opt[1]:
            ax_e0.set_ylim(
                0.75*(e0_prior.mean-e0_prior.sdev),
                1.25*(e0_prior.mean+e0_prior.sdev))
            ax_e0.axhline(e0_prior.mean-e0_prior.sdev,
                        linestyle='--', color='k', alpha=.5)
            ax_e0.axhline(e0_prior.mean+e0_prior.sdev,
                        linestyle='--', color='k', alpha=.5)

    if scale:
        ax_e0r = ax_e0.twinx()
        ax_e0r.set_ylim(ax_e0.get_ylim()[0]*s, ax_e0.get_ylim()[1]*s)
        ax_e0r.set_yticks([s*t for t in ax_e0.get_yticks()[1:-1]])
        if units in ['GeV', 'gev']:
            ax_e0r.set_yticklabels(["%.2f" % t for t in ax_e0r.get_yticks()])
        else:
            ax_e0r.set_yticklabels(["%.0f" % t for t in ax_e0r.get_yticks()])

    #ax_e0.text(0.05,0.1, text, transform=ax_e0.transAxes,
    #    bbox={'facecolor':ens_colors[a_str],'boxstyle':'round'},
    #    horizontalalignment='left', fontsize=fs)
    ax_e0.legend(loc=1, ncol=8, columnspacing=0,
                fontsize=fs_ns, handletextpad=0.1)
    if n_plot >= 0 and n_plot < tn_opt[1]:
        ax_e0.axhspan(e0_opt.mean-e0_opt.sdev, e0_opt.mean
                    + e0_opt.sdev, color=colors[tn_opt[1]], alpha=.2)
    ax_e0.set_xticks(tmin)
    if ylim is not None:
        ax_e0.set_ylim(ylim)

    ax_Q.set_xticks(tmin)
    ax_Q.set_yticks([0.1, 0.75])
    ax_w.set_xticks(tmin)
    ax_w.set_yticks([0.1, 0.75])
    ax_e0.tick_params(bottom=True, labelbottom=False, top=True, direction='in')
    ax_Q.tick_params(bottom=True, labelbottom=False, top=True, direction='in')
    ax_w.tick_params(bottom=True, top=True, direction='in')

    ax_w.set_ylabel(r'$w_{n_s}$', fontsize=fs)
    ax_w.set_xlabel(r'$t_{\rm min}$', fontsize=fs)
    ax_w.set_ylim(0, 1.2)
    ax_Q.set_ylabel(r'$Q$', fontsize=fs)
    ax_Q.set_ylim(0, 1.2)
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/'+state+'_E_'+nn+'_stability.pdf', transparent=True)
