import numpy as np
import gvar as gv
import re 
import pandas as pd 
import sys
import copy
# import tables as h5
import h5py
import os 
import time
import re
import argparse
import importlib
import corrfitter
import collections
import numpy as np
import lsqfit

import fitter.corr_functions as cf

from utilities.h5io import get_dsets 
from utilities.parsing import parse_t_info, parse_file_info 
from utilities.concat_ import concatenate,concat_dsets

def prepare_xyp(states, u_xp, gv_data):
    '''
    scrape input file 
    '''

    x = copy.deepcopy(u_xp.x)
    # prepare data y
    # y = {k: v[x[k]['t_range']]
    #      for (k, v) in gv_data.items() if k.split('_')[0] in states}
    
    n_states = dict()
    for state in states:
        for k in x:
            if state in k and 'mres' not in k:
                n_states[state] = x[k]['n_state']

    priors = dict()
    for k in u_xp.priors:
        for state in states:
            if 'mres' not in k:
                k_n = int(k.split('_')[-1].split(')')[0])
                if state == k.split('(')[-1].split('_')[0] and k_n < n_states[state]:
                    priors[k] = gv.gvar(u_xp.priors[k].mean, u_xp.priors[k].sdev)
            else:
                mres = k.split('_')[0]
                if mres in states:
                    priors[k] = gv.gvar(u_xp.priors[k].mean, u_xp.priors[k].sdev)

    return x,priors 

def prepare_xyp(states, u_xp, gv_data):
    '''
    scrape input file 
    '''

    x = copy.deepcopy(u_xp.x)
    # prepare data y
    # y = {k: v[x[k]['t_range']]
    #      for (k, v) in gv_data.items() if k.split('_')[0] in states}
    
    n_states = dict()
    for state in states:
        for k in x:
            if state in k and 'mres' not in k:
                n_states[state] = x[k]['n_state']

    priors = dict()
    for k in u_xp.priors:
        for state in states:
            if 'mres' not in k:
                k_n = int(k.split('_')[-1].split(')')[0])
                if state == k.split('(')[-1].split('_')[0] and k_n < n_states[state]:
                    priors[k] = gv.gvar(u_xp.priors[k].mean, u_xp.priors[k].sdev)
            else:
                mres = k.split('_')[0]
                if mres in states:
                    priors[k] = gv.gvar(u_xp.priors[k].mean, u_xp.priors[k].sdev)

    return x,priors 

''' 
Convert raw correlator data into agreeable format for fitter.
Apply statistical treatments to refactored raw data (jackknife,bootstrap) 

Args: 


Returns:
    Coalesced_Dataset object: 
        corr_gv : dictionary of correlated data
'''



def coalesce_data(corr_raw,bl=9,bl=9, skip_prelim=False,fold=False,nt=None):

    corr_binned = raw_to_binned(
        corr_raw,
        bl=bl,
        fold=fold,
        bl=bl,
        fold=fold
    )

    corr_gv = Coalesced_Dataset(
        corr_binned,
        skip_prelim = skip_prelim,
        nt = nt
    )

    return corr_gv

class Coalesced_Dataset(object):
    def __init__(self,data_dict,datatags=None,skip_prelim=False):
        self.datatags= datatags
        fit_data_3pt = {datatag: val for datatag,val in data_dict.items() if isinstance(datatag, int)}
        self.c_3pt = cf.C_3pt(datatag=None, fit_data_3pt=fit_data_3pt)

        self.c_2pt = {}
        for datatag in self.datatags:
            self.c_2pt[datatag] = cf.C_2pt(datatag, data_dict[datatag],nt = self.c_3pt.times.nt) 

    def check_timeslice(self):
        slices = [self.c_2pt.src.times.nt,self.c_2pt.snk.times.nt,self.c_3pt.times.nt]
        for tslice in slices:
            if not np.all(tslice == slices[0]):
                raise ValueError('check that your timeslices are equivalent for corrs')

def raw_to_binned(data, bl=9):
    ''' data shape is [Ncfg, others]
        bl = block length in configs
    '''
    if bl <= 1:
        return data

    # (1000, 32) --> (200, 5, 32)
    Ncfg = data.shape[0]
    data = np.reshape(data, (Ncfg//bl, bl))
    return np.average(data, axis=1) 
    # --> (200, 32)


    # ncfg, nt_gf = data.shape
    # if ncfg % bl == 0:
    #     nb = ncfg // bl
    # else:
    #     nb = ncfg // bl + 1
    # corr_bl = np.zeros([nb, nt_gf], dtype=data.dtype)
    # for b in range(nb-1):
    #     corr_bl[b] = data[b*bl:(b+1)*bl].mean(axis=0)
    # corr_bl[nb-1] = data[(nb-1)*bl:].mean(axis=0)

    # return corr_bl

def bs_to_gvar(data,corr, bs_N):
    ''' provide corr for dict key '''
    bs_M = data[corr].shape[0]
    print(bs_M)
    bs_list = np.random.randint(low=0, high=bs_M, size=(bs_N, bs_M))
    
    temp_dict = {}
    for key in data.keys():
        temp = data[key][bs_list[0, :],:]
        temp_dict[key] = np.mean(temp, axis=0)
        
    for k in range(1, bs_N):
        for key in data.keys():
            temp = data[key][bs_list[k, :],:]
            temp_dict[key] = np.vstack((temp_dict[key], np.mean(temp, axis=0)))
    
    output = {}
    for key in data.keys():
        mean = np.mean(data[key], axis=0)
        unc = np.cov(temp_dict[key], rowvar=False)
        output[key] = gv.gvar(mean, unc)
    return output

def correlate(data):
    mean = gv.mean(gv.dataset.avg_data(data))
    covariance = compute_covariance(data)
    return gv.gvar(mean, cov)


    def check_timeslice(self):
        slices = [self.c_2pt.src.times.nt,self.c_2pt.snk.times.nt,self.c_3pt.times.nt]
        for tslice in slices:
            if not np.all(tslice == slices[0]):
                raise ValueError('check that your timeslices are equivalent for corrs')



# def normalize_ff(curr,mom,m_snk):
#     #TODO 
#     normalize = np.float(1)
#     return normalize

# def normalize_R(ff,E_p,M):
#     ''' TODO ratio -> scalar form factor '''
#     if ff == '51':

#     elif ff == '52':
    
#     elif ff== '53':

#     elif ff== '54':

#     return np.sqrt(2*E_p(E_p)+M)

''' Statistics routines '''



def bs_corr(corr,Nbs,Mbs,seed=None):
        corr_bs = np.zeros(tuple([Nbs]) + corr.shape[1:],dtype=corr.dtype)
        np.random.seed(seed)
        # make bs_lst of shape (Nbs,Mbs)
        bs_lst = np.random.randint(0,corr.shape[0],(Nbs,Mbs))
        # use bs_lst to make corr_bs entries
        for bs in range(Nbs):
            corr_bs[bs] = corr[bs_lst[bs]].mean(axis=0)
        return corr_bs

def jackknife(data):
    ''' bin first, then jn '''
    N = data.shape[0]
    _jackknife_avg = (data.sum() - data)/(N-1)
    
    jackknife_avg = np.mean(_jackknife_avg)
    jackknife_err = np.sqrt((N - 1)*(np.mean(_jackknife_avg**2) - jackknife_avg**2))
    return jackknife_avg, jackknife_err

''' preliminary dset handling '''

def unpack_tuple(func):
    def func_wrapper(data, *args, **kwargs):
        if type(data[0]) is tuple:
            retvalue = ()
            for i in range(len(data[0])):
                obj_array = []
                for k in range(len(data)):
                    obj_array.append(data[k][i])
                retvalue += (func(obj_array, *args, **kwargs),)
            return retvalue
        else:
            return func(data, *args, **kwargs)
    return func_wrapper

''' properties of unpacked tuple '''

@unpack_tuple
def std_mean(data, axis=0):
    return np.mean(data, axis)


@unpack_tuple
def std_median(data, axis=0):
    return np.median(data, axis)

''' properties of unpacked tuple '''

@unpack_tuple
def std_err(data, axis=0):
    data = np.asarray(data)
    return std_dev(data, axis) / np.sqrt(data.shape[axis])

@unpack_tuple
def std_var(data, axis=0):
    data = np.asarray(data)
    return np.var(data, axis=axis, ddof=1)


@unpack_tuple
def std_dev(data, axis=0):
    data = np.asarray(data)
    return np.std(data, axis=axis, ddof=1)


def mean_and_err(data, axis=0):
    mean = std_mean(data, axis=axis)
    error = std_err(data, axis=axis)
    return mean, error


def mean_and_cov(data, axis=0):
    mean = std_mean(data, axis=axis)
    cov = calc_cov(data)
    return mean, cov


def mean_and_std_dev(data, axis=0):
    mean = std_mean(data, axis=axis)
    std = std_dev(data, axis=axis)
    return mean, std






''' misc. parsing routines to be implemented when database IO is built out '''

def parse_corr_block(file):
    """
    Parses a correlator data block into lists of t, Re(C), Im(C).
    Args:
        ifile: if .txt file, routine will work. TODO add to class of parsing routines
        based on type of input file (.h5,.txt,.dat, etc.). Namely, interface with 
        database IO.
    Returns:
        t, real, imag: three lists with the data
    """
    # Use regex to be picky about what constitutes data and to fail
    # early and noisily if something unexpected appears
    corr_block = r"(-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)"
    datum = re.compile(r"^(\d+)" + f" {corr_block} {corr_block}$")
    t, real, imag = [], [], []
    data_block = True
    while data_block:
        line = next(file)
        match = re.match(datum, line)
        if match:
            tokens = match.groups() # t, real, imag
            t.append(tokens[0])
            real.append(tokens[1])
            imag.append(tokens[2])
        else:
            LOGGER.error("ERROR: Unrecognized line in data block %s", line)
            raise ValueError(f"Unrecognized line in data block, {line}")
    return t, real, imag
            

        
       
       
if __name__== '__main__':
    main()