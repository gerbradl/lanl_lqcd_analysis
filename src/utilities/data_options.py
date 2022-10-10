""" data i/o utilities for run_data_options """

import numpy as np
import gvar as gv
import re 
# import pandas as pd 
import sys
import copy
import tables as h5
import h5py
import os 
import time
import collections

from utilities.h5io import get_dsets 
from utilities.parsing import parse_t_info, parse_file_info 

from utilities.concat_ import concatenate,concat_dsets
from utilities.utils import group_files,parse_dset_address


''' regex patterns dict '''
dset_replace_patterns = {}
dset_replace_patterns['pion'] = {
    '2pt' : '2pt',
    '(?P<corr>pion)':'pion',
    '(?P<cfg>E7.\w\_[0-9][0-9][0-9][0-9]+)\/': ''
    } # note the trailing /
dset_replace_patterns['pion_SP'] = {
    '2pt' : '2pt',
    #'(?P<corr>pion|pion_SP|proton|proton_SP)':'corr',
    '(?P<corr>pion_SP)':'pion_SP',
    '(?P<cfg>E7.\w\_[0-9][0-9][0-9][0-9]+)\/': ''
    } # note the trailing /
dset_replace_patterns['proton'] = {
    '2pt' : '2pt',
    #'(?P<corr>pion|pion_SP|proton|proton_SP)':'corr',
    '(?P<corr>proton)':'proton',
    '(?P<cfg>E7.\w\_[0-9][0-9][0-9][0-9]+)\/': ''
    } # note the trailing /
dset_replace_patterns['proton_SP'] = {
    '2pt' : '2pt',
    #'(?P<corr>pion|pion_SP|proton|proton_SP)':'corr',
    '(?P<corr>proton_SP)':'proton_SP',
    '(?P<cfg>E7.\w\_[0-9][0-9][0-9][0-9]+)\/': ''
    } # note the trailing /

'''
Chroma dict for 3pt corr data:
NUCL: nucleon
U: quark bilinear operator inserted on up-quark
D:  quark bilinear operator inserted on down-quark
MIXED: "mixed" type of spin projection is used
NONREL: non-relativistic proton is used
l0:  the separation of the quarks of the bilinear operator is zero (local operator);
l1:  quark bilinear operator separated by 1 lattice unit
g_{}: the gamma matrix of the quark bilinear operator
 
    0: scalar; I
    15: pseudoscalar; g_5
    1: vector;  g_x
    2: vector;  g_y
    4: vector;  g_z
    8: vector;  g_t
    14: axial;   g_x g_5
    13: axial;  -g_y g_5
    11: axial;   g_z g_5
    7: axial;  -g_t g_5
    9: tensor;  g_x g_t
    10: tensor;  g_y g_t
    12: tensor;  g_z g_t
    3: tensor;  g_x g_y
    6: tensor;  g_y g_z
    5: tensor;  g_x g_z
''' 

dset_replace_patterns['gA'] = {
    # '3pt' : '',
    "3pt_tsep(?P<tsep>[0-9][0-9]+)" : '\g<tsep>',
    "NUCL_(?P<quark>U|D)" : '',  # Store U or D in quark
    "_MIXED_NONREL" : '',  # Not sure if this changes. Not stored for now
    # "_l(?P<l>[0-9]+)": '',  # action parameters?
    # "_g(?P<g>[0-15]+)": '',
    "/src(?P<src>[0-9\.]+)":'',  # Stores numbers + . to store decimals. Must escape .
    "_snk(?P<snk>[0-9\.]+)":'',  # Stores numbers + . to store decimals. Must escape .
    "qz(?P<qz>[\+\-0-9]+)": "qz\g<qz>",
    "_qy(?P<qy>[\+\-0-9]+)": "_qy\g<qy>",
    "_qx(?P<qx>[\+\-0-9]+)": "_qx\g<qx>",
    '(?P<cfg>E7.\w\_[0-9][0-9][0-9][0-9]+)\/': ''
}

string_ = '3pt_tsep21/NUCL_D_MIXED_NONREL_l0_g11/src10.0_snk10.0/qz+1_qy+2_qx+0/E7.a_1716/AMA'
out_grp,meta_info = parse_dset_address(string_,dset_replace_patterns=dset_replace_patterns['gA'])
# print(out_grp,meta_info)

''' functions to parse/inspect data params from input file '''

def get_tsep(string):
    result = {}
    match = re.search(r"_tsep(?P<tsep>[0-9][0-9]+)", string)
    if match:
            for key, val in match.groupdict().items():
                result[key] = int(val)
    return result

def parse_cfg_arg(cfg_arg,params):
    allowed_cfgs = range(params['cfg_i'],params['cfg_f']+1,params['cfg_d'])
    if not cfg_arg:
        ci = params['cfg_i']
        cf = params['cfg_f']
        dc = params['cfg_d']
    else:
        if len(cfg_arg) == 3:
            cfgs = range(cfg_arg[0], cfg_arg[1]+1, cfg_arg[2])
        elif len(cfg_arg) == 1:
            cfgs = range(cfg_arg[0], cfg_arg[0]+1, 1)
        if not all([cfg in allowed_cfgs for cfg in cfgs]):
            print('you selected configs not allowed for %s:' %params['ENS_S'])
            print('       allowed cfgs = range(%d, %d, %d)' %(params['cfg_i'], params['cfg_f'], params['cfg_d']))
            sys.exit('  your choice: cfgs = range(%d, %d, %d)' %(cfgs[0],cfgs[-1],cfgs[1]-cfgs[0]))
        elif len(cfg_arg) == 1:
            ci = int(cfg_arg[0])
            cf = int(cfg_arg[0])
            dc = 1
        elif len(cfg_arg) == 3:
            ci = int(cfg_arg[0])
            cf = int(cfg_arg[1])
            dc = int(cfg_arg[2])
        else:
            print('unrecognized use of cfg arg')
            print('cfg_i [cfg_f cfg_d]')
            sys.exit()
    return range(ci,cf+1,dc)

def parse_baryon_tag(datatag):
    ''' Given a datatag, return dict with keys to pass into fitter 
    This is currently not implemented anywhere, ideally this would act on an 
    existing database of datatags with correlated data or fits '''
    datatag_split   = datatag.split('/')
    corr_type       = datatag_split[0]
    tsep            = int(corr_type.split('_tsep')[1])
    buffer          =  datatag_split[1]
    channel         = buffer.split('_')[0]
    quark_ins       = buffer.split('_')[1]
    spin_proj       = buffer.split('_')[2]
    quark_sep       = buffer.split('_')[3]
    gamma           = buffer.split('_')[4] #gamma matrix of quark bilinear operator in the CHROMA convention , value accessed via dict
    src_snk_sep     = datatag_split[2]
    mom             = datatag_split[3]
    mom0            = mom.split('_')[0]
    mom1            = mom.split('_')[1]
    mom2            = mom.split('_')[2]
    momentum        = (mom0,mom1,mom2)
    config          = datatag_split[4]

    data_dict = dict()
    data_dict['corr_type']   = corr_type
    data_dict['tsep']        = tsep
    data_dict['buffer']      = buffer
    data_dict['channel']     = channel
    data_dict['quark_ins']   = quark_ins
    data_dict['spin_proj']   = spin_proj
    data_dict['quark_sep']   = quark_sep
    data_dict['gamma']       = gamma
    data_dict['src_snk_sep'] = src_snk_sep
    data_dict['mom']         = momentum
    data_dict['config']      = config
    return data_dict

def q_simple_lst(n=4):
    r = [i for j in (range(-n,0), range(1,n+1)) for i in j]
    q_lst = []
    q_lst.append('qx0_qy0_qz0')
    for q in r:
        q_lst.append('qx%d_qy0_qz0' %q)
        q_lst.append('qx0_qy%d_qz0' %q)
        q_lst.append('qx0_qy0_qz%d' %q)
    return q_lst

def mom_avg(h5_data,state,mom_lst,weights=False):
    '''
    perform a momentum average of a state from an open h5 file
    data file is assumed to be of shape [Nt,Nz,Ny,Nx,[re,im]]
    data_mom = h5_data[state][:,qz,qy,qx]
    '''
    d_lst = []
    # w = []
    for mom in mom_lst:
        qx,qy,qz = mom['momentum']
        # w.append(mom['weight'])
        #print(state)
        d_lst.append(h5_data[state][:,qz,qy,qx])
    d_lst = np.array(d_lst)
    w = np.array(w)
    if weights:
        for wi,we in enumerate(w):
            d_lst[wi] = we*d_lst[wi]
        d_avg = np.sum(d_lst,axis=0) / np.sum(w)
    else:
        d_avg = np.mean(d_lst,axis=0)
    return d_avg

mom_lst = []
for qx in range(-2,3):
    for qy in range(-2,3):
        for qz in range(-2,3):
            if qx**2 + qy**2 + qz**2 <= 5:
                mom_lst.append('qz'+str(qz)+'_qy'+str(qy)+'_qx'+str(qx))

def unpack_tuple(data):
    # if type(data[0]) is tuple:
    for i in range(len(data[0])):
        obj_array = []
        for k in range(len(data)):
            obj_array.append(data[k][i])
    return obj_array
