import sys 
import argparse
import argcomplete
import numpy as np
import gvar as gv
import re 
import pandas as pd 
import copy
import tables as h5
import h5py
import os 
import time
import collections
import importlib
# sys.path.insert(0,'/home/gbradley/c51_corr_analysis')

from src.utilities.h5io import get_dsets 
from src.utilities.parsing import parse_t_info, parse_file_info 

from src.utilities.concat_ import concatenate,concat_dsets
from src.utilities.utils import group_files,parse_dset_address
from src.utilities.data_options import *

class Format(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()  
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)

def main():
    # formatting for help messages 
    formatter = lambda form: argparse.RawTextHelpFormatter(form) 

    parser = argparse.ArgumentParser(description= "Compute averages of provided data"
    "with several statistical options to choose from")

    parser.add_argument('-input_file',help='input file to specify fit')
    parser.add_argument('-output_dir',help='dir to save concat. h5 files')
    
    

    parser.add_argument('-bs', '--bootstrap')
    parser.add_argument('-jn', '--jackknife')
    parser.add_argument('-sdev', '--standard_deviation')

    parser.add_argument('-columns', '--data_columns')
    parser.add_argument('-jn_nbs', '--jackknife_blocks')

    args = parser.parse_args()
    # add path to the input file and load it
    # if no sys arg given for filename but input file instead:
    sys.path.append(os.path.dirname(os.path.abspath(args.input_file)))
    fp = importlib.import_module(
        args.input_file.split('/')[-1].split('.py')[0])

    ens = fp.params['ENS']
    data_dir = fp.file_params['data_dir']
    dirs = os.listdir( data_dir )
    cnf_abbr = [files.split(".ama.h5")[0] for files in dirs]
    cnf_abbr = [cnf.replace('-','.') for cnf in cnf_abbr]
    # cnf_abbr_ascend = {}
    cnf_abbr_ascend = [cnf_.split('_')[1] for cnf_ in cnf_abbr]
    cfg_abbr_sorted = np.sort(cnf_abbr_ascend,axis=None)
    with open("cfg_list.txt","a") as f: 
        print(cfg_abbr_sorted.astype(int).tolist(),file=f)
    # embed()
    data_file_list = list()
    for dirpath,_,filenames in os.walk(data_dir):
        for f in filenames:
            data_file_list.append(os.path.abspath(os.path.join(dirpath, f)))
    sorted_files = np.sort(data_file_list)


    """ read in pure """ 
    out_file = {}
    out_file['pion'] = os.path.join(os.getcwd(),"pion.h5")
    out_file['pion_SP'] = os.path.join(os.getcwd(),"pion_SP.h5")
    out_file['proton'] = os.path.join(os.getcwd(),"proton.h5")
    out_file['proton_all'] = os.path.join(os.getcwd(),"proton_all.h5")
    out_file['proton_SP'] = os.path.join(os.getcwd(),"proton_SP.h5")
    out_file['3pt'] = os.path.join(os.getcwd(),"3pt_out.h5")

    print("if want to override regex patterns, edit in data_options")
    # TODO this needs to be in a loop, but not working with external module concat_dsets
    # for corr in dset_replace_patterns.keys():
    # concat_dsets(data_file_list[0:20], out_file['proton_all'], dset_replace_patterns=dset_replace_patterns['proton'],overwrite=False,write_unpaired_dsets=True)
    # concat_dsets(data_file_list[0:4], out_file['pion'], dset_replace_patterns=dset_replace_patterns['pion'],overwrite=False,write_unpaired_dsets=False)
    # concat_dsets(data_file_list[0:4], out_file['pion'], dset_replace_patterns=dset_replace_patterns['pion_SP'],overwrite=False,write_unpaired_dsets=True)
    # concat_dsets(data_file_list[0:4], out_file['proton_SP'], dset_replace_patterns=dset_replace_patterns['proton_SP'],overwrite=False,write_unpaired_dsets=True)
    concat_dsets(data_file_list[0:100], out_file['3pt'], dset_replace_patterns=dset_replace_patterns['gA'],overwrite=True,write_unpaired_dsets=True)
    

    
    """ naive read in """
    # if args.bootstrap:
    
    # if args.jackknife:

    # if args.standard_deviation:

if __name__ == '__main__':
    main()