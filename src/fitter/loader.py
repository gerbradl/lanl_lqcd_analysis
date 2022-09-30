import numpy as np
import os 
import h5py

directory = './data/C13/'
N_cnf = len([name for name in os.listdir(directory) if os.path.isfile(name)])

dirs = os.listdir( directory )
# print(dirs)
# data_file_list = os.path.realpath(dirs)
data_file_list = []
for dirpath,_,filenames in os.walk(directory):
    for f in filenames:
        data_file_list.append(os.path.abspath(os.path.join(dirpath, f)))
        print(data_file_list)

def load_columns(filename, expected_column_count=None):
    data = h5py.File(filename)
    data2d = np.atleast_2d(data)
    shape = data2d.shape
    num_cols = shape[1]

    if num_cols == 0 and expected_column_count is not None:
        cols = [np.array([]) for i in range(expected_column_count)]
    else:
        cols = [data2d[:, i] for i in range(num_cols)]
    return cols

def folded_list_loader(filenames):
    all_folded = []

    for filename in filenames:
        t, re, im = load_columns(filename)

        n = len(t)

        a = re

        second_rev = a[n//2+1:][::-1]
        first = a[:n//2+1]
        first[1:-1] += second_rev
        first[1:-1] /= 2.
        a = first

        all_folded.append(a)

    return all_folded

folded_list_loader(data_file_list)