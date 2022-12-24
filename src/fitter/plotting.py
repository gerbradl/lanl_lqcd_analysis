'''
plot raw correlators
effective mass plot 
effective wf overlap plot 
effective mass stability plot 
Summary plot 
TODO time history plot (correlator for each configuration on a single plot)
'''

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