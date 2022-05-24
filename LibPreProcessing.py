import os
import glob
import collections
import numpy as np
import nibabel
import natsort
import scipy
import json
import umap

from sklearn import preprocessing


def process_files(subpath, filetype, index):
    if index == 0:
        PATH = os.getcwd() + subpath
        FILES = os.listdir(PATH)
        FILES = glob.glob(PATH + '/*.' + filetype)
    if index == 1:
        PATH = os.getcwd() + subpath
        FILES = os.listdir(PATH)
        FILES = glob.glob(PATH + '/**/**.' + filetype)
    if index == 2:
        PATH = os.getcwd() + subpath
        FILES = os.listdir(PATH)
        FILES = glob.glob(PATH + '/**/**/**.' + filetype)
    if index == 'A':
        PATH = os.getcwd() + subpath
        FILES = os.listdir(PATH)
        FILES = glob.glob(PATH + '/**')
    if index == 'B':
        PATH = os.getcwd() + subpath
        FILES = os.listdir(PATH)
        FILES = glob.glob(PATH + '/**/**')
    if index == 'C':
        PATH = os.getcwd() + subpath
        FILES = os.listdir(PATH)
        FILES = glob.glob(PATH + '/**/**/**')
    return natsort.natsorted(FILES)


def label_encoder():
    return preprocessing.LabelEncoder()


def standard_scaler(param):
    pass


def reducer_umap(param):
    reducer = umap.UMAP(param)
    return reducer


def find_most_common(lst):
    return max(set(lst), key=lst.count)


def find_duplicates(list_input):
    list_pure = [
        item for item, count in collections.Counter(
            list_input
        ).items() if count > 1
    ]
    return list_pure


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def nifti2array(data):
    data = nibabel.load(
        data
    ).get_fdata()[:, :, :]
    return data


def scale(data, scaler):
    if scaler == 'standard':
        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        data_scaled = scaler.transform(data)
    if scaler == 'minmax':
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(data)
        data_scaled = scaler.transform(data)
    return data_scaled


def encode(data):
    encoder = preprocessing.LabelEncoder()
    y_encoded = encoder.fit_transform(data)
    y_decoded = encoder.inverse_transform(y_encoded)
    return (y_encoded, y_decoded)


def zscore(data):
    data = data.apply(scipy.stats.zscore)
    return data


def load_json(data):
    data = json.load(data)
    return data


def load_nifti(data):
    data = nibabel.load(data)
    return data
