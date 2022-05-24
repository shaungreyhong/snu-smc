import itertools
import operator
import pickle
import pandas as pd
import numpy as np
import os
import glob
import collections
import tensorflow as tf
import pydicom
import scipy
# from shutil import copyfile
# import nibabel as nib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    jaccard_score,
    confusion_matrix,
    plot_confusion_matrix,
    davies_bouldin_score,
    silhouette_score
)
from sklearn.metrics.cluster import (
    normalized_mutual_info_score,
    adjusted_rand_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from natsort import natsorted


def showModelPerformance(y_test, y_pred):
    print('Model accuracy score : {0:0.9f}'. format(accuracy_score(y_test, y_pred)))
    print('Model f1 score : {0:0.9f}'. format(f1_score(y_test, y_pred, pos_label='positive', average='weighted')))
    print('Model precision score : {0:0.9f}'. format(precision_score(y_test, y_pred, pos_label='positive', average='weighted')))
    print('Model recall score : {0:0.9f}'. format(recall_score(y_test, y_pred, pos_label='positive', average='weighted')))
    # print('Model ROC AUC score : {0:0.9f}'. format(roc_auc_score(y, CV.predict_proba(X), multi_class='ovr')))
    # scores = cross_val_score(CV, X, y, cv=3, scoring='accuracy')
    # print('CV %0.2f accuracy with a standard deviation of %0.2f' % (scores.mean(), scores.std()))


def sortMostCommon(list_input):
    sorted_list = sorted((x, i) for i, x in enumerate(list_input))
    groups = itertools.groupby(sorted_list, key=operator.itemgetter(0))

    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(list_input)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        return count, -min_index
    return max(groups, key=_auxfun)[0]


def sortSymmetry(list_input):
    classes_Rt = pd.DataFrame(list_input)[pd.Series(list_input).str.contains('Rt')]
    classes_Lt = pd.DataFrame(list_input)[pd.Series(list_input).str.contains('Lt')]
    classes_mirror = pd.concat([classes_Rt, classes_Lt])[0].tolist()
    classes_center = pd.DataFrame(list_input)[~pd.Series(list_input).str.contains('Rt|Lt')]
    if len(list_input) % 2 == 0:
        classes_mirror.insert(int(len(classes_mirror)/2), ''.join(classes_center.values.tolist()[0]))
        classes_mirror.insert(int(len(classes_mirror)/2+1), ''.join(classes_center.values.tolist()[1]))
    if len(list_input) % 2 == 1:
        classes_mirror.insert(int(len(classes_mirror)/2+1), ''.join(classes_center.values.tolist()[0]))
    return classes_mirror


def findDuplicates(list_input):
    list_pure = [item for item, count in collections.Counter(list_input).items() if count > 1]
    return list_pure


def findPlaneOrientation(path):
    data = pydicom.read_file(path)
    IOP = data.ImageOrientationPatient
    IOP_round = [round(x) for x in IOP]
    plane = np.cross(IOP_round[0:3], IOP_round[3:6])
    plane = [abs(x) for x in plane]
    # ['0', '1', '0', '0', '0', '-1'] you are dealing with Sagittal plane view
    if plane[0] == 1:
        return 'sagittal'
    # ['1', '0', '0', '0', '0', '-1'] you are dealing with Coronal plane view
    elif plane[1] == 1:
        return 'coronal'
    # ['1', '0', '0', '0', '1', '0'] you are dealing with Axial plane view
    elif plane[2] == 1:
        return 'transverse'


def findNearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
    

def scaleData(data, scaler):
    if scaler == 'standard':
        scaler = StandardScaler()
        scaler.fit(data)
        data_scaled = scaler.transform(data)
    if scaler == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(data)
        data_scaled = scaler.transform(data)
    return data_scaled


def encodeData(data):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_decoded = encoder.inverse_transform(y_encoded)
    return (y_encoded, y_decoded)


def reconstructVesselAcronym(feature_vector):
    feature_vector.iloc[:, -1] = feature_vector.iloc[:, -1].replace(
        [
            'Rt_ICA', 'Rt_Ophthalmic', 'Rt_AChA',
            'Lt_ICA', 'Lt_Ophthalmic', 'Lt_AChA',
            'Rt_M1', 'Rt_MCA_Superior', 'Rt_MCA_Inferior',
            'Lt_M1', 'Lt_MCA_Superior', 'Lt_MCA_Inferior',
            'Rt_MCA_lat_OFA', 'Rt_MCA_PreRolandic', 'Rt_MCA_Rolandic',
            'Rt_MCA_AntPerietal', 'Rt_MCA_PostParietal', 'Rt_MCA_Angular',
            'Rt_MCA_PostTemporal', 'Rt_MCA_MidTemporal', 'Rt_MCA_AntTemporal', 'Rt_MCA_Prefrontal',
            'Lt_MCA_lat_OFA', 'Lt_MCA_PreRolandic', 'Lt_MCA_Rolandic',
            'Lt_MCA_AntPerietal', 'Lt_MCA_PostParietal', 'Lt_MCA_Angular',
            'Lt_MCA_PostTemporal', 'Lt_MCA_MidTemporal', 'Lt_MCA_AntTemporal', 'Lt_MCA_Prefrontal',
            'Rt_A1', 'Rt_A2', 'Rt_A1+A2',
            'Lt_A1', 'Lt_A2', 'Lt_A1+A2',
            'Rt_ACA_med_OFA', 'Rt_A2_Frontopolar', 'Rt_ACA_Callosamarginal',
            'Rt_ACA_Pericallosal', 'Lt_ACA_med_OFA', 'Lt_A2_Frontopolar',
            'Lt_ACA_Callosamarginal', 'Lt_ACA_Pericallosal', 'Rt_VA',
            'Lt_VA', 'Rt_P1', 'Rt_P2',
            'Rt_P1+P2', 'Rt_P3,P4', 'Lt_P1',
            'Lt_P2', 'Lt_P1+P2', 'Lt_P3,P4',
            'Rt_PPA', 'Rt_Hippocampal artery', 'Rt_PCA_AnteriorTemporal',
            'Rt_PCA_PosteriorTemporal', 'Rt_PCA_lat_Pca', 'Rt_PCoA', 'Lt_PPA',
            'Lt_Hippocampal artery', 'Lt_PCA_AnteriorTemporal', 'Lt_PCA_PosteriorTemporal',
            'Lt_PCA_lat_Pca', 'Lt_PCoA', 'Rt_PICA', 'Rt_AICA',
            'Rt_IAA', 'Rt_SCA', 'Lt_PICA',
            'Lt_AICA', 'Lt_IAA', 'Lt_SCA',
            'BA', 'ACoA'
        ], [
            'Rt_ICA', 'Rt_Ophthalmic', 'Rt_AChA',
            'Lt_ICA', 'Lt_Ophthalmic', 'Lt_AChA',
            'Rt_M1', 'Rt_MCA_Superior', 'Rt_MCA_Inferior',
            'Lt_M1', 'Lt_MCA_Superior', 'Lt_MCA_Inferior',
            'Rt_MCA_lat_OFA', 'Rt_MCA_PreRolandic', 'Rt_MCA_Rolandic',
            'Rt_MCA_AntPerietal', 'Rt_MCA_PostParietal', 'Rt_MCA_Angular',
            'Rt_MCA_PostTemporal', 'Rt_MCA_MidTemporal', 'Rt_MCA_AntTemporal', 'Rt_MCA_Prefrontal',
            'Lt_MCA_lat_OFA', 'Lt_MCA_PreRolandic', 'Lt_MCA_Rolandic',
            'Lt_MCA_AntPerietal', 'Lt_MCA_PostParietal', 'Lt_MCA_Angular',
            'Lt_MCA_PostTemporal', 'Lt_MCA_MidTemporal', 'Lt_MCA_AntTemporal', 'Lt_MCA_Prefrontal',
            'Rt_A1', 'Rt_A2', 'Rt_A1+A2',
            'Lt_A1', 'Lt_A2', 'Lt_A1+A2',
            'Rt_ACA_med_OFA', 'Rt_A2_Frontopolar', 'Rt_ACA_Callosamarginal',
            'Rt_ACA_Pericallosal', 'Lt_ACA_med_OFA', 'Lt_A2_Frontopolar',
            'Lt_ACA_Callosamarginal', 'Lt_ACA_Pericallosal', 'Rt_VA',
            'Lt_VA', 'Rt_P1', 'Rt_P2',
            'Rt_P1+P2', 'Rt_P3,P4', 'Lt_P1',
            'Lt_P2', 'Lt_P1+P2', 'Lt_P3,P4',
            'Rt_PPA', 'Rt_Hippocampal artery', 'Rt_PCA_AnteriorTemporal',
            'Rt_PCA_PosteriorTemporal', 'Rt_PCA_lat_Pca', 'Rt_PCoA', 'Lt_PPA',
            'Lt_Hippocampal artery', 'Lt_PCA_AnteriorTemporal', 'Lt_PCA_PosteriorTemporal',
            'Lt_PCA_lat_Pca', 'Lt_PCoA', 'Rt_PICA', 'Rt_AICA',
            'Rt_IAA', 'Rt_SCA', 'Lt_PICA',
            'Lt_AICA', 'Lt_IAA', 'Lt_SCA',
            'BA', 'ACoA'
        ]
    )
    
    '''
    'G10', 'G11', 'G12',
    'G20', 'G21', 'G22',
    'G30', 'G31', 'G32',
    'G40', 'G41', 'G42',
    'G50', 'G51', 'G52',
    'G53', 'G54', 'G55',
    'G56', 'G57', 'G58', 'G59',
    'G60', 'G61', 'G62',
    'G63', 'G64', 'G65',
    'G66', 'G67', 'G68', 'G69',
    'G70', 'G71', 'G72',
    'G80', 'G81', 'G82',
    'G90', 'G91', 'G92',
    'G93', 'G100', 'G101',
    'G102', 'G103', 'G110',
    'G120', 'G130', 'G131',
    'G132', 'G133', 'G140',
    'G141', 'G142', 'G143',
    'G150', 'G151', 'G152',
    'G153', 'G154', 'G155', 'G160',
    'G161', 'G162', 'G163',
    'G164', 'G165', 'G170', 'G171',
    'G172', 'G173', 'G180',
    'G181', 'G182', 'G183',
    'G190', 'G200'
    '''
    
    return feature_vector


def reconstructVesselChunk(feature_vector, n_components):
    if n_components == '20A':
        df_all_modeling = feature_vector

        df_all_modeling_group1 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_ICA') |
            (df_all_modeling.VesselName == 'Rt_Ophthalmic')
        ]
        df_all_modeling_group1_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_ICA') |
            (df_all_modeling.VesselName == 'Rt_Ophthalmic')
        ]
        df_all_modeling_group1['VesselName'] = 'G01-Rt_ICA'
        df_all_modeling_group1_r = reconstructVesselAcronym(df_all_modeling_group1_r)

        df_all_modeling_group2 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ICA') |
            (df_all_modeling.VesselName == 'Lt_Ophthalmic')
        ]
        df_all_modeling_group2_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ICA') |
            (df_all_modeling.VesselName == 'Lt_Ophthalmic')
        ]
        df_all_modeling_group2['VesselName'] = 'G02-Lt_ICA'
        df_all_modeling_group2_r = reconstructVesselAcronym(df_all_modeling_group2_r)

        df_all_modeling_group3 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Rt_M1') |
            (df_all_modeling.VesselName == 'Rt_MCA_Inferior')
        ]
        df_all_modeling_group3_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Rt_M1') |
            (df_all_modeling.VesselName == 'Rt_MCA_Inferior')
        ]
        df_all_modeling_group3['VesselName'] = 'G03-Rt_Ant-MCA-Basal'
        df_all_modeling_group3_r = reconstructVesselAcronym(df_all_modeling_group3_r)

        df_all_modeling_group4 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Lt_M1') |
            (df_all_modeling.VesselName == 'Lt_MCA_Inferior')
        ]
        df_all_modeling_group4_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Lt_M1') |
            (df_all_modeling.VesselName == 'Lt_MCA_Inferior')
        ]
        df_all_modeling_group4['VesselName'] = 'G04-Lt_Ant-MCA-Basal'
        df_all_modeling_group4_r = reconstructVesselAcronym(df_all_modeling_group4_r)

        df_all_modeling_group5 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Prefrontal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Angular')
        ]
        df_all_modeling_group5_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Prefrontal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Angular')
        ]
        df_all_modeling_group5['VesselName'] = 'G05-Rt_Ant-MCA-Pial'
        df_all_modeling_group5_r = reconstructVesselAcronym(df_all_modeling_group5_r)

        df_all_modeling_group6 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_Prefrontal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_Angular')
        ]
        df_all_modeling_group6_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_Prefrontal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_Angular')
        ]
        df_all_modeling_group6['VesselName'] = 'G06-Lt_Ant-MCA-Pial'
        df_all_modeling_group6_r = reconstructVesselAcronym(df_all_modeling_group6_r)

        df_all_modeling_group7 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A1') |
            (df_all_modeling.VesselName == 'Rt_A2') |
            (df_all_modeling.VesselName == 'Rt_A1+A2')
        ]
        df_all_modeling_group7_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A1') |
            (df_all_modeling.VesselName == 'Rt_A2') |
            (df_all_modeling.VesselName == 'Rt_A1+A2')
        ]
        df_all_modeling_group7['VesselName'] = 'G07-Rt_Ant-ACA-Basal'
        df_all_modeling_group7_r = reconstructVesselAcronym(df_all_modeling_group7_r)

        df_all_modeling_group8 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_A1') |
            (df_all_modeling.VesselName == 'Lt_A2') |
            (df_all_modeling.VesselName == 'Lt_A1+A2')
        ]
        df_all_modeling_group8_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_A1') |
            (df_all_modeling.VesselName == 'Lt_A2') |
            (df_all_modeling.VesselName == 'Lt_A1+A2')
        ]
        df_all_modeling_group8['VesselName'] = 'G08-Lt_Ant-ACA-Basal'
        df_all_modeling_group8_r = reconstructVesselAcronym(df_all_modeling_group8_r)

        df_all_modeling_group9 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Rt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Rt_ACA_Pericallosal')
        ]
        df_all_modeling_group9_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Rt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Rt_ACA_Pericallosal')
        ]
        df_all_modeling_group9['VesselName'] = 'G09-Rt_Ant-ACA-Pial'
        df_all_modeling_group9_r = reconstructVesselAcronym(df_all_modeling_group9_r)

        df_all_modeling_group10 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Lt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Lt_ACA_Pericallosal')
        ]
        df_all_modeling_group10_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Lt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Lt_ACA_Pericallosal')
        ]
        df_all_modeling_group10['VesselName'] = 'G10-Lt_Ant-ACA-Pial'
        df_all_modeling_group10_r = reconstructVesselAcronym(df_all_modeling_group10_r)

        df_all_modeling_group11 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11['VesselName'] = 'G11-Rt_Post-VA'
        df_all_modeling_group11_r = reconstructVesselAcronym(df_all_modeling_group11_r)

        df_all_modeling_group12 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12['VesselName'] = 'G12-Lt_Post-VA'
        df_all_modeling_group12_r = reconstructVesselAcronym(df_all_modeling_group12_r)

        df_all_modeling_group13 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_P1') |
            (df_all_modeling.VesselName == 'Rt_P2') |
            (df_all_modeling.VesselName == 'Rt_P3,P4') |
            (df_all_modeling.VesselName == 'Rt_P1+P2')
        ]
        df_all_modeling_group13_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_P1') |
            (df_all_modeling.VesselName == 'Rt_P2') |
            (df_all_modeling.VesselName == 'Rt_P3,P4') |
            (df_all_modeling.VesselName == 'Rt_P1+P2')
        ]
        df_all_modeling_group13['VesselName'] = 'G13-Rt_Post-PCA-Basal'
        df_all_modeling_group13_r = reconstructVesselAcronym(df_all_modeling_group13_r)

        df_all_modeling_group14 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_P1') |
            (df_all_modeling.VesselName == 'Lt_P2') |
            (df_all_modeling.VesselName == 'Lt_P3,P4') |
            (df_all_modeling.VesselName == 'Lt_P1+P2')
        ]
        df_all_modeling_group14_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_P1') |
            (df_all_modeling.VesselName == 'Lt_P2') |
            (df_all_modeling.VesselName == 'Lt_P3,P4') |
            (df_all_modeling.VesselName == 'Lt_P1+P2')
        ]
        df_all_modeling_group14['VesselName'] = 'G14-Lt_Post-PCA-Basal'
        df_all_modeling_group14_r = reconstructVesselAcronym(df_all_modeling_group14_r)

        df_all_modeling_group15 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PCoA') |
            (df_all_modeling.VesselName == 'Rt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Rt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group15_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PCoA') |
            (df_all_modeling.VesselName == 'Rt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Rt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group15['VesselName'] = 'G15-Rt_Post-PCA-Pial'
        df_all_modeling_group15_r = reconstructVesselAcronym(df_all_modeling_group15_r)

        df_all_modeling_group16 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PCoA') |
            (df_all_modeling.VesselName == 'Lt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Lt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group16_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PCoA') |
            (df_all_modeling.VesselName == 'Lt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Lt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group16['VesselName'] = 'G16-Lt_Post-PCA-Pial'
        df_all_modeling_group16_r = reconstructVesselAcronym(df_all_modeling_group16_r)

        df_all_modeling_group17 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PICA') |
            (df_all_modeling.VesselName == 'Rt_AICA') |
            (df_all_modeling.VesselName == 'Rt_SCA')
        ]
        df_all_modeling_group17_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PICA') |
            (df_all_modeling.VesselName == 'Rt_AICA') |
            (df_all_modeling.VesselName == 'Rt_SCA')
        ]
        df_all_modeling_group17['VesselName'] = 'G17-Rt_SCA,AICA,PICA'
        df_all_modeling_group17_r = reconstructVesselAcronym(df_all_modeling_group17_r)

        df_all_modeling_group18 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PICA') |
            (df_all_modeling.VesselName == 'Lt_AICA') |
            (df_all_modeling.VesselName == 'Lt_SCA')
        ]
        df_all_modeling_group18_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PICA') |
            (df_all_modeling.VesselName == 'Lt_AICA') |
            (df_all_modeling.VesselName == 'Lt_SCA')
        ]
        df_all_modeling_group18['VesselName'] = 'G18-Lt_SCA,AICA,PICA'
        df_all_modeling_group18_r = reconstructVesselAcronym(df_all_modeling_group18_r)

        df_all_modeling_group19 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19['VesselName'] = 'G19-BA'
        df_all_modeling_group19_r = reconstructVesselAcronym(df_all_modeling_group19_r)

        df_all_modeling_group20 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20['VesselName'] = 'G20-ACoA'
        df_all_modeling_group20_r = reconstructVesselAcronym(df_all_modeling_group20_r)

        df_all_modeling_group = pd.concat(
            [
                df_all_modeling_group1,
                df_all_modeling_group2,
                df_all_modeling_group3,
                df_all_modeling_group4,
                df_all_modeling_group5,
                df_all_modeling_group6,
                df_all_modeling_group7,
                df_all_modeling_group8,
                df_all_modeling_group9,
                df_all_modeling_group10,
                df_all_modeling_group11,
                df_all_modeling_group12,
                df_all_modeling_group13,
                df_all_modeling_group14,
                df_all_modeling_group15,
                df_all_modeling_group16,
                df_all_modeling_group17,
                df_all_modeling_group18,
                df_all_modeling_group19,
                df_all_modeling_group20
            ], ignore_index=True, axis=0
        )
    
    if n_components == '20B':
        df_all_modeling = feature_vector
        
        df_all_modeling_group1 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_ICA') |
            (df_all_modeling.VesselName == 'Rt_Ophthalmic') |
            (df_all_modeling.VesselName == 'Rt_AChA')
        ]
        df_all_modeling_group1_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_ICA') |
            (df_all_modeling.VesselName == 'Rt_Ophthalmic') |
            (df_all_modeling.VesselName == 'Rt_AChA')
        ]
        df_all_modeling_group1['VesselName'] = 'G01-Rt_ICA'
        df_all_modeling_group1_r = reconstructVesselAcronym(df_all_modeling_group1_r)

        df_all_modeling_group2 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ICA') |
            (df_all_modeling.VesselName == 'Lt_Ophthalmic') |
            (df_all_modeling.VesselName == 'Lt_AChA')
        ]
        df_all_modeling_group2_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ICA') |
            (df_all_modeling.VesselName == 'Lt_Ophthalmic') |
            (df_all_modeling.VesselName == 'Lt_AChA')
        ]
        df_all_modeling_group2['VesselName'] = 'G02-Lt_ICA'
        df_all_modeling_group2_r = reconstructVesselAcronym(df_all_modeling_group2_r)

        df_all_modeling_group3 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Rt_M1') |
            (df_all_modeling.VesselName == 'Rt_MCA_Inferior')
        ]
        df_all_modeling_group3_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Rt_M1') |
            (df_all_modeling.VesselName == 'Rt_MCA_Inferior')
        ]
        df_all_modeling_group3['VesselName'] = 'G03-Rt_Ant-MCA-Basal'
        df_all_modeling_group3_r = reconstructVesselAcronym(df_all_modeling_group3_r)

        df_all_modeling_group4 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Lt_M1') |
            (df_all_modeling.VesselName == 'Lt_MCA_Inferior')
        ]
        df_all_modeling_group4_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Lt_M1') |
            (df_all_modeling.VesselName == 'Lt_MCA_Inferior')
        ]
        df_all_modeling_group4['VesselName'] = 'G04-Lt_Ant-MCA-Basal'
        df_all_modeling_group4_r = reconstructVesselAcronym(df_all_modeling_group4_r)

        df_all_modeling_group5 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_lat_OFA') |
            (df_all_modeling.VesselName == 'Rt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Angular') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_MidTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntTemporal')
        ]
        df_all_modeling_group5_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_lat_OFA') |
            (df_all_modeling.VesselName == 'Rt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Angular') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_MidTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntTemporal')
        ]
        df_all_modeling_group5['VesselName'] = 'G05-Rt_Ant-MCA-Pial'
        df_all_modeling_group5_r = reconstructVesselAcronym(df_all_modeling_group5_r)

        df_all_modeling_group6 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_lat_OFA') |
            (df_all_modeling.VesselName == 'Lt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_Angular') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_MidTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntTemporal')
        ]
        df_all_modeling_group6_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_lat_OFA') |
            (df_all_modeling.VesselName == 'Lt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_Angular') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_MidTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntTemporal')
        ]
        df_all_modeling_group6['VesselName'] = 'G06-Lt_Ant-MCA-Pial'
        df_all_modeling_group6_r = reconstructVesselAcronym(df_all_modeling_group6_r)

        df_all_modeling_group7 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A1') |
            (df_all_modeling.VesselName == 'Rt_A2') |
            (df_all_modeling.VesselName == 'Rt_A1+A2')
        ]
        df_all_modeling_group7_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A1') |
            (df_all_modeling.VesselName == 'Rt_A2') |
            (df_all_modeling.VesselName == 'Rt_A1+A2')
        ]
        df_all_modeling_group7['VesselName'] = 'G07-Rt_Ant-ACA-Basal'
        df_all_modeling_group7_r = reconstructVesselAcronym(df_all_modeling_group7_r)

        df_all_modeling_group8 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_A1') |
            (df_all_modeling.VesselName == 'Lt_A2') |
            (df_all_modeling.VesselName == 'Lt_A1+A2')
        ]
        df_all_modeling_group8_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_A1') |
            (df_all_modeling.VesselName == 'Lt_A2') |
            (df_all_modeling.VesselName == 'Lt_A1+A2')
        ]
        df_all_modeling_group8['VesselName'] = 'G08-Lt_Ant-ACA-Basal'
        df_all_modeling_group8_r = reconstructVesselAcronym(df_all_modeling_group8_r)

        df_all_modeling_group9 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_ACA_med_OFA') |
            (df_all_modeling.VesselName == 'Rt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Rt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Rt_ACA_Pericallosal')
        ]
        df_all_modeling_group9_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_ACA_med_OFA') |
            (df_all_modeling.VesselName == 'Rt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Rt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Rt_ACA_Pericallosal')
        ]
        df_all_modeling_group9['VesselName'] = 'G09-Rt_Ant-ACA-Pial'
        df_all_modeling_group9_r = reconstructVesselAcronym(df_all_modeling_group9_r)

        df_all_modeling_group10 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ACA_med_OFA') |
            (df_all_modeling.VesselName == 'Lt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Lt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Lt_ACA_Pericallosal')
        ]
        df_all_modeling_group10_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ACA_med_OFA') |
            (df_all_modeling.VesselName == 'Lt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Lt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Lt_ACA_Pericallosal')
        ]
        df_all_modeling_group10['VesselName'] = 'G10-Lt_Ant-ACA-Pial'
        df_all_modeling_group10_r = reconstructVesselAcronym(df_all_modeling_group10_r)

        df_all_modeling_group11 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11['VesselName'] = 'G11-Rt_Post-VA'
        df_all_modeling_group11_r = reconstructVesselAcronym(df_all_modeling_group11_r)

        df_all_modeling_group12 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12['VesselName'] = 'G12-Lt_Post-VA'
        df_all_modeling_group12_r = reconstructVesselAcronym(df_all_modeling_group12_r)

        df_all_modeling_group13 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_P1') |
            (df_all_modeling.VesselName == 'Rt_P2') |
            (df_all_modeling.VesselName == 'Rt_P3,P4') |
            (df_all_modeling.VesselName == 'Rt_P1+P2')
        ]
        df_all_modeling_group13_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_P1') |
            (df_all_modeling.VesselName == 'Rt_P2') |
            (df_all_modeling.VesselName == 'Rt_P3,P4') |
            (df_all_modeling.VesselName == 'Rt_P1+P2')
        ]
        df_all_modeling_group13['VesselName'] = 'G13-Rt_Post-PCA-Basal'
        df_all_modeling_group13_r = reconstructVesselAcronym(df_all_modeling_group13_r)

        df_all_modeling_group14 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_P1') |
            (df_all_modeling.VesselName == 'Lt_P2') |
            (df_all_modeling.VesselName == 'Lt_P3,P4') |
            (df_all_modeling.VesselName == 'Lt_P1+P2')
        ]
        df_all_modeling_group14_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_P1') |
            (df_all_modeling.VesselName == 'Lt_P2') |
            (df_all_modeling.VesselName == 'Lt_P3,P4') |
            (df_all_modeling.VesselName == 'Lt_P1+P2')
        ]
        df_all_modeling_group14['VesselName'] = 'G14-Lt_Post-PCA-Basal'
        df_all_modeling_group14_r = reconstructVesselAcronym(df_all_modeling_group14_r)

        df_all_modeling_group15 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PCoA') |
            (df_all_modeling.VesselName == 'Rt_PPA') |
            (df_all_modeling.VesselName == 'Rt_Hippocampal artery') |
            (df_all_modeling.VesselName == 'Rt_PCA_lat_Pca') |
            (df_all_modeling.VesselName == 'Rt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Rt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group15_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PCoA') |
            (df_all_modeling.VesselName == 'Rt_PPA') |
            (df_all_modeling.VesselName == 'Rt_Hippocampal artery') |
            (df_all_modeling.VesselName == 'Rt_PCA_lat_Pca') |
            (df_all_modeling.VesselName == 'Rt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Rt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group15['VesselName'] = 'G15-Rt_Post-PCA-Pial'
        df_all_modeling_group15_r = reconstructVesselAcronym(df_all_modeling_group15_r)

        df_all_modeling_group16 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PCoA') |
            (df_all_modeling.VesselName == 'Lt_PPA') |
            (df_all_modeling.VesselName == 'Lt_Hippocampal artery') |
            (df_all_modeling.VesselName == 'Lt_PCA_lat_Pca') |
            (df_all_modeling.VesselName == 'Lt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Lt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group16_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PCoA') |
            (df_all_modeling.VesselName == 'Lt_PPA') |
            (df_all_modeling.VesselName == 'Lt_Hippocampal artery') |
            (df_all_modeling.VesselName == 'Lt_PCA_lat_Pca') |
            (df_all_modeling.VesselName == 'Lt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Lt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group16['VesselName'] = 'G16-Lt_Post-PCA-Pial'
        df_all_modeling_group16_r = reconstructVesselAcronym(df_all_modeling_group16_r)

        df_all_modeling_group17 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PICA') |
            (df_all_modeling.VesselName == 'Rt_AICA') |
            (df_all_modeling.VesselName == 'Rt_SCA') |
            (df_all_modeling.VesselName == 'Rt_IAA')
        ]
        df_all_modeling_group17_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PICA') |
            (df_all_modeling.VesselName == 'Rt_AICA') |
            (df_all_modeling.VesselName == 'Rt_SCA') |
            (df_all_modeling.VesselName == 'Rt_IAA')
        ]
        df_all_modeling_group17['VesselName'] = 'G17-Rt_SCA,AICA,PICA'
        df_all_modeling_group17_r = reconstructVesselAcronym(df_all_modeling_group17_r)

        df_all_modeling_group18 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PICA') |
            (df_all_modeling.VesselName == 'Lt_AICA') |
            (df_all_modeling.VesselName == 'Lt_SCA') |
            (df_all_modeling.VesselName == 'Lt_IAA')
        ]
        df_all_modeling_group18_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PICA') |
            (df_all_modeling.VesselName == 'Lt_AICA') |
            (df_all_modeling.VesselName == 'Lt_SCA') |
            (df_all_modeling.VesselName == 'Lt_IAA')
        ]
        df_all_modeling_group18['VesselName'] = 'G18-Lt_SCA,AICA,PICA'
        df_all_modeling_group18_r = reconstructVesselAcronym(df_all_modeling_group18_r)

        df_all_modeling_group19 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19['VesselName'] = 'G19-BA'
        df_all_modeling_group19_r = reconstructVesselAcronym(df_all_modeling_group19_r)

        df_all_modeling_group20 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20['VesselName'] = 'G20-ACoA'
        df_all_modeling_group20_r = reconstructVesselAcronym(df_all_modeling_group20_r)

        df_all_modeling_group = pd.concat(
            [
                df_all_modeling_group1,
                df_all_modeling_group2,
                df_all_modeling_group3,
                df_all_modeling_group4,
                df_all_modeling_group5,
                df_all_modeling_group6,
                df_all_modeling_group7,
                df_all_modeling_group8,
                df_all_modeling_group9,
                df_all_modeling_group10,
                df_all_modeling_group11,
                df_all_modeling_group12,
                df_all_modeling_group13,
                df_all_modeling_group14,
                df_all_modeling_group15,
                df_all_modeling_group16,
                df_all_modeling_group17,
                df_all_modeling_group18,
                df_all_modeling_group19,
                df_all_modeling_group20
            ], ignore_index=True, axis=0
        )
        
    return df_all_modeling_group


def reconstructVesselGroup(feature_vector, n_components):
    if n_components == '4A':
        df_all_modeling = feature_vector
  
        df_all_modeling_large_anterior = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A1') |
            (df_all_modeling.VesselName == 'Rt_A2') |
            (df_all_modeling.VesselName == 'Rt_ICA') |
            (df_all_modeling.VesselName == 'Rt_M1') |
            (df_all_modeling.VesselName == 'Rt_MCA_Inferior') |
            (df_all_modeling.VesselName == 'Rt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Lt_A1') |
            (df_all_modeling.VesselName == 'Lt_A2') |
            (df_all_modeling.VesselName == 'Lt_ICA') |
            (df_all_modeling.VesselName == 'Lt_M1') |
            (df_all_modeling.VesselName == 'Lt_MCA_Inferior') |
            (df_all_modeling.VesselName == 'Lt_MCA_Superior')
        ]
        df_all_modeling_large_anterior['VesselName'] = 'Large Anterior'

        df_all_modeling_large_posterior = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA') |
            (df_all_modeling.VesselName == 'Rt_P1') |
            (df_all_modeling.VesselName == 'Rt_P1+P2') |
            (df_all_modeling.VesselName == 'Rt_P2') |
            (df_all_modeling.VesselName == 'Rt_VA') |
            (df_all_modeling.VesselName == 'Lt_P1') |
            (df_all_modeling.VesselName == 'Lt_P1+P2') |
            (df_all_modeling.VesselName == 'Lt_P2') |
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_large_posterior['VesselName'] = 'Large Posterior'

        df_all_modeling_small_anterior = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Rt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Rt_ACA_Pericallosal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Angular') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntPerietal') | 
            (df_all_modeling.VesselName == 'Rt_MCA_AntTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Prefrontal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Lt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Lt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Lt_ACA_Pericallosal') |
            (df_all_modeling.VesselName == 'Lt_MCA_Angular') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntPerietal') | 
            (df_all_modeling.VesselName == 'Lt_MCA_AntTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_Prefrontal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_Rolandic')
        ]
        df_all_modeling_small_anterior['VesselName'] = 'Small Anterior'

        df_all_modeling_small_posterior = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_AICA') |
            (df_all_modeling.VesselName == 'Rt_P3,P4') |
            (df_all_modeling.VesselName == 'Rt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Rt_PCA_PosteriorTemporal') |
            (df_all_modeling.VesselName == 'Rt_PICA') |
            (df_all_modeling.VesselName == 'Rt_SCA') |
            (df_all_modeling.VesselName == 'Lt_AICA') |
            (df_all_modeling.VesselName == 'Lt_P3,P4') |
            (df_all_modeling.VesselName == 'Lt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Lt_PCA_PosteriorTemporal') |
            (df_all_modeling.VesselName == 'Lt_PICA') |
            (df_all_modeling.VesselName == 'Lt_SCA')
        ]
        df_all_modeling_small_posterior['VesselName'] = 'Small Posterior'

        df_all_modeling_group = pd.concat(
            [
                df_all_modeling_large_anterior,
                df_all_modeling_large_posterior,
                df_all_modeling_small_anterior,
                df_all_modeling_small_posterior
            ], ignore_index=True, axis=0
        )

        classes = pd.DataFrame(df_all_modeling_group.VesselName.value_counts()).index.tolist()
        classes = sorted(classes, key=lambda x: int(x.split('-')[0][1:]), reverse=False)
        classes_r = sorted(classes, key=lambda x: int(x.split('-')[0][1:]), reverse=True)
        classes_mirror = sortSymmetry(classes)
        classes_r_mirror = sortSymmetry(classes_r)
        X = df_all_modeling_group

    if n_components == '9A':
        df_all_modeling = feature_vector
        
        df_all_modeling_group1 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_ICA') |
            (df_all_modeling.VesselName == 'Lt_ICA') |
            (df_all_modeling.VesselName == 'Rt_Ophthalmic') |
            (df_all_modeling.VesselName == 'Lt_Ophthalmic')
        ]
        df_all_modeling_group1['VesselName'] = 'G1-ICA'

        df_all_modeling_group2 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Lt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Rt_M1') |
            (df_all_modeling.VesselName == 'Lt_M1')
        ]
        df_all_modeling_group2['VesselName'] = 'G2-Ant-MCA-Basal'

        df_all_modeling_group3 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Prefrontal') |
            (df_all_modeling.VesselName == 'Lt_MCA_Prefrontal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Angular') |
            (df_all_modeling.VesselName == 'Lt_MCA_Angular')
        ]
        df_all_modeling_group3['VesselName'] = 'G3-Ant-MCA-Pial'

        df_all_modeling_group4 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A1') |
            (df_all_modeling.VesselName == 'Lt_A1') |
            (df_all_modeling.VesselName == 'Rt_A2') |
            (df_all_modeling.VesselName == 'Lt_A2') |
            (df_all_modeling.VesselName == 'Rt_A1+A2') |
            (df_all_modeling.VesselName == 'Lt_A1+A2') |
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group4['VesselName'] = 'G4-Ant-ACA-Basal'

        df_all_modeling_group5 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Lt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Rt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Lt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Rt_ACA_Pericallosal') |
            (df_all_modeling.VesselName == 'Lt_ACA_Pericallosal')
        ]
        df_all_modeling_group5['VesselName'] = 'G5-Ant-ACA-Pial'

        df_all_modeling_group6 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA') |
            (df_all_modeling.VesselName == 'Lt_VA') |
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group6['VesselName'] = 'G6-Post-VA,BA'

        df_all_modeling_group7 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_P1') |
            (df_all_modeling.VesselName == 'Lt_P1') |
            (df_all_modeling.VesselName == 'Rt_P2') |
            (df_all_modeling.VesselName == 'Lt_P2') |
            (df_all_modeling.VesselName == 'Rt_P1+P2') |
            (df_all_modeling.VesselName == 'Lt_P1+P2')
        ]
        df_all_modeling_group7['VesselName'] = 'G7-Post-PCA-Basal'

        df_all_modeling_group8 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PCoA') |
            (df_all_modeling.VesselName == 'Lt_PCoA') |
            (df_all_modeling.VesselName == 'Rt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Lt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Rt_PCA_PosteriorTemporal') |
            (df_all_modeling.VesselName == 'Lt_PCA_PosteriorTemporal') |
            (df_all_modeling.VesselName == 'Rt_P3,P4') |
            (df_all_modeling.VesselName == 'Lt_P3,P4')
        ]
        df_all_modeling_group8['VesselName'] = 'G8-Post-PCA-Pial'

        df_all_modeling_group9 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PICA') |
            (df_all_modeling.VesselName == 'Lt_PICA') |
            (df_all_modeling.VesselName == 'Rt_AICA') |
            (df_all_modeling.VesselName == 'Lt_AICA') |
            (df_all_modeling.VesselName == 'Rt_SCA') |
            (df_all_modeling.VesselName == 'Lt_SCA')
        ]
        df_all_modeling_group9['VesselName'] = 'G9-SCA,AICA,PICA'

        df_all_modeling_group = pd.concat(
            [
                df_all_modeling_group1,
                df_all_modeling_group2,
                df_all_modeling_group3,
                df_all_modeling_group4,
                df_all_modeling_group5,
                df_all_modeling_group6,
                df_all_modeling_group7,
                df_all_modeling_group8,
                df_all_modeling_group9
            ], ignore_index=True, axis=0
        )
        
        classes = pd.DataFrame(df_all_modeling_group.VesselName.value_counts()).index.tolist()
        classes = sorted(classes, key=lambda x: int(x.split('-')[0][1:]), reverse=False)
        classes_r = sorted(classes, key=lambda x: int(x.split('-')[0][1:]), reverse=True)
        classes_mirror = sortSymmetry(classes)
        classes_r_mirror = sortSymmetry(classes_r)
        X = df_all_modeling_group
        
    if n_components == '20A':
        df_all_modeling = feature_vector

        df_all_modeling_group1 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_ICA') |
            (df_all_modeling.VesselName == 'Rt_Ophthalmic')
        ]
        df_all_modeling_group1_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_ICA') |
            (df_all_modeling.VesselName == 'Rt_Ophthalmic')
        ]
        df_all_modeling_group1['VesselName'] = 'G01-Rt_ICA'
        df_all_modeling_group1_r = reconstructVesselAcronym(df_all_modeling_group1_r)

        df_all_modeling_group2 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ICA') |
            (df_all_modeling.VesselName == 'Lt_Ophthalmic')
        ]
        df_all_modeling_group2_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ICA') |
            (df_all_modeling.VesselName == 'Lt_Ophthalmic')
        ]
        df_all_modeling_group2['VesselName'] = 'G02-Lt_ICA'
        df_all_modeling_group2_r = reconstructVesselAcronym(df_all_modeling_group2_r)

        df_all_modeling_group3 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Rt_M1') |
            (df_all_modeling.VesselName == 'Rt_MCA_Inferior')
        ]
        df_all_modeling_group3_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Rt_M1') |
            (df_all_modeling.VesselName == 'Rt_MCA_Inferior')
        ]
        df_all_modeling_group3['VesselName'] = 'G03-Rt_Ant-MCA-Basal'
        df_all_modeling_group3_r = reconstructVesselAcronym(df_all_modeling_group3_r)

        df_all_modeling_group4 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Lt_M1') |
            (df_all_modeling.VesselName == 'Lt_MCA_Inferior')
        ]
        df_all_modeling_group4_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Lt_M1') |
            (df_all_modeling.VesselName == 'Lt_MCA_Inferior')
        ]
        df_all_modeling_group4['VesselName'] = 'G04-Lt_Ant-MCA-Basal'
        df_all_modeling_group4_r = reconstructVesselAcronym(df_all_modeling_group4_r)

        df_all_modeling_group5 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Prefrontal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Angular')
        ]
        df_all_modeling_group5_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Prefrontal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Angular')
        ]
        df_all_modeling_group5['VesselName'] = 'G05-Rt_Ant-MCA-Pial'
        df_all_modeling_group5_r = reconstructVesselAcronym(df_all_modeling_group5_r)

        df_all_modeling_group6 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_Prefrontal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_Angular')
        ]
        df_all_modeling_group6_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_Prefrontal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_Angular')
        ]
        df_all_modeling_group6['VesselName'] = 'G06-Lt_Ant-MCA-Pial'
        df_all_modeling_group6_r = reconstructVesselAcronym(df_all_modeling_group6_r)

        df_all_modeling_group7 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A1') |
            (df_all_modeling.VesselName == 'Rt_A2') |
            (df_all_modeling.VesselName == 'Rt_A1+A2')
        ]
        df_all_modeling_group7_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A1') |
            (df_all_modeling.VesselName == 'Rt_A2') |
            (df_all_modeling.VesselName == 'Rt_A1+A2')
        ]
        df_all_modeling_group7['VesselName'] = 'G07-Rt_Ant-ACA-Basal'
        df_all_modeling_group7_r = reconstructVesselAcronym(df_all_modeling_group7_r)

        df_all_modeling_group8 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_A1') |
            (df_all_modeling.VesselName == 'Lt_A2') |
            (df_all_modeling.VesselName == 'Lt_A1+A2')
        ]
        df_all_modeling_group8_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_A1') |
            (df_all_modeling.VesselName == 'Lt_A2') |
            (df_all_modeling.VesselName == 'Lt_A1+A2')
        ]
        df_all_modeling_group8['VesselName'] = 'G08-Lt_Ant-ACA-Basal'
        df_all_modeling_group8_r = reconstructVesselAcronym(df_all_modeling_group8_r)

        df_all_modeling_group9 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Rt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Rt_ACA_Pericallosal')
        ]
        df_all_modeling_group9_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Rt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Rt_ACA_Pericallosal')
        ]
        df_all_modeling_group9['VesselName'] = 'G09-Rt_Ant-ACA-Pial'
        df_all_modeling_group9_r = reconstructVesselAcronym(df_all_modeling_group9_r)

        df_all_modeling_group10 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Lt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Lt_ACA_Pericallosal')
        ]
        df_all_modeling_group10_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Lt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Lt_ACA_Pericallosal')
        ]
        df_all_modeling_group10['VesselName'] = 'G10-Lt_Ant-ACA-Pial'
        df_all_modeling_group10_r = reconstructVesselAcronym(df_all_modeling_group10_r)

        df_all_modeling_group11 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11['VesselName'] = 'G11-Rt_Post-VA'
        df_all_modeling_group11_r = reconstructVesselAcronym(df_all_modeling_group11_r)

        df_all_modeling_group12 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12['VesselName'] = 'G12-Lt_Post-VA'
        df_all_modeling_group12_r = reconstructVesselAcronym(df_all_modeling_group12_r)

        df_all_modeling_group13 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_P1') |
            (df_all_modeling.VesselName == 'Rt_P2') |
            (df_all_modeling.VesselName == 'Rt_P3,P4') |
            (df_all_modeling.VesselName == 'Rt_P1+P2')
        ]
        df_all_modeling_group13_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_P1') |
            (df_all_modeling.VesselName == 'Rt_P2') |
            (df_all_modeling.VesselName == 'Rt_P3,P4') |
            (df_all_modeling.VesselName == 'Rt_P1+P2')
        ]
        df_all_modeling_group13['VesselName'] = 'G13-Rt_Post-PCA-Basal'
        df_all_modeling_group13_r = reconstructVesselAcronym(df_all_modeling_group13_r)

        df_all_modeling_group14 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_P1') |
            (df_all_modeling.VesselName == 'Lt_P2') |
            (df_all_modeling.VesselName == 'Lt_P3,P4') |
            (df_all_modeling.VesselName == 'Lt_P1+P2')
        ]
        df_all_modeling_group14_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_P1') |
            (df_all_modeling.VesselName == 'Lt_P2') |
            (df_all_modeling.VesselName == 'Lt_P3,P4') |
            (df_all_modeling.VesselName == 'Lt_P1+P2')
        ]
        df_all_modeling_group14['VesselName'] = 'G14-Lt_Post-PCA-Basal'
        df_all_modeling_group14_r = reconstructVesselAcronym(df_all_modeling_group14_r)

        df_all_modeling_group15 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PCoA') |
            (df_all_modeling.VesselName == 'Rt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Rt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group15_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PCoA') |
            (df_all_modeling.VesselName == 'Rt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Rt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group15['VesselName'] = 'G15-Rt_Post-PCA-Pial'
        df_all_modeling_group15_r = reconstructVesselAcronym(df_all_modeling_group15_r)

        df_all_modeling_group16 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PCoA') |
            (df_all_modeling.VesselName == 'Lt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Lt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group16_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PCoA') |
            (df_all_modeling.VesselName == 'Lt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Lt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group16['VesselName'] = 'G16-Lt_Post-PCA-Pial'
        df_all_modeling_group16_r = reconstructVesselAcronym(df_all_modeling_group16_r)

        df_all_modeling_group17 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PICA') |
            (df_all_modeling.VesselName == 'Rt_AICA') |
            (df_all_modeling.VesselName == 'Rt_SCA')
        ]
        df_all_modeling_group17_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PICA') |
            (df_all_modeling.VesselName == 'Rt_AICA') |
            (df_all_modeling.VesselName == 'Rt_SCA')
        ]
        df_all_modeling_group17['VesselName'] = 'G17-Rt_SCA,AICA,PICA'
        df_all_modeling_group17_r = reconstructVesselAcronym(df_all_modeling_group17_r)

        df_all_modeling_group18 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PICA') |
            (df_all_modeling.VesselName == 'Lt_AICA') |
            (df_all_modeling.VesselName == 'Lt_SCA')
        ]
        df_all_modeling_group18_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PICA') |
            (df_all_modeling.VesselName == 'Lt_AICA') |
            (df_all_modeling.VesselName == 'Lt_SCA')
        ]
        df_all_modeling_group18['VesselName'] = 'G18-Lt_SCA,AICA,PICA'
        df_all_modeling_group18_r = reconstructVesselAcronym(df_all_modeling_group18_r)

        df_all_modeling_group19 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19['VesselName'] = 'G19-BA'
        df_all_modeling_group19_r = reconstructVesselAcronym(df_all_modeling_group19_r)

        df_all_modeling_group20 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20['VesselName'] = 'G20-ACoA'
        df_all_modeling_group20_r = reconstructVesselAcronym(df_all_modeling_group20_r)

        df_all_modeling_group = pd.concat(
            [
                df_all_modeling_group1,
                df_all_modeling_group2,
                df_all_modeling_group3,
                df_all_modeling_group4,
                df_all_modeling_group5,
                df_all_modeling_group6,
                df_all_modeling_group7,
                df_all_modeling_group8,
                df_all_modeling_group9,
                df_all_modeling_group10,
                df_all_modeling_group11,
                df_all_modeling_group12,
                df_all_modeling_group13,
                df_all_modeling_group14,
                df_all_modeling_group15,
                df_all_modeling_group16,
                df_all_modeling_group17,
                df_all_modeling_group18,
                df_all_modeling_group19,
                df_all_modeling_group20
            ], ignore_index=True, axis=0
        )
        
        df_all_modeling_group_r = pd.concat(
            [
                df_all_modeling_group1_r,
                df_all_modeling_group2_r,
                df_all_modeling_group3_r,
                df_all_modeling_group4_r,
                df_all_modeling_group5_r,
                df_all_modeling_group6_r,
                df_all_modeling_group7_r,
                df_all_modeling_group8_r,
                df_all_modeling_group9_r,
                df_all_modeling_group10_r,
                df_all_modeling_group11_r,
                df_all_modeling_group12_r,
                df_all_modeling_group13_r,
                df_all_modeling_group14_r,
                df_all_modeling_group15_r,
                df_all_modeling_group16_r,
                df_all_modeling_group17_r,
                df_all_modeling_group18_r,
                df_all_modeling_group19_r,
                df_all_modeling_group20_r
            ], ignore_index=True, axis=0
        )
        
        '''
        classes = pd.DataFrame(df_all_modeling_group.VesselName.value_counts()).index.tolist()
        classes = sorted(classes, key=lambda x: int(x.split('-')[0][1:]), reverse=False)
        classes_r = sorted(classes, key=lambda x: int(x.split('-')[0][1:]), reverse=True)
        classes_mirror = sortSymmetry(classes)
        classes_r_mirror = sortSymmetry(classes_r)
        '''
        X = df_all_modeling_group
        X_r = df_all_modeling_group_r
    
    if n_components == '20B':
        df_all_modeling = feature_vector
        
        df_all_modeling_group1 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_ICA') |
            (df_all_modeling.VesselName == 'Rt_Ophthalmic') |
            (df_all_modeling.VesselName == 'Rt_AChA')
        ]
        df_all_modeling_group1_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_ICA') |
            (df_all_modeling.VesselName == 'Rt_Ophthalmic') |
            (df_all_modeling.VesselName == 'Rt_AChA')
        ]
        df_all_modeling_group1['VesselName'] = 'G01-Rt_ICA'
        df_all_modeling_group1_r = reconstructVesselAcronym(df_all_modeling_group1_r)

        df_all_modeling_group2 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ICA') |
            (df_all_modeling.VesselName == 'Lt_Ophthalmic') |
            (df_all_modeling.VesselName == 'Lt_AChA')
        ]
        df_all_modeling_group2_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ICA') |
            (df_all_modeling.VesselName == 'Lt_Ophthalmic') |
            (df_all_modeling.VesselName == 'Lt_AChA')
        ]
        df_all_modeling_group2['VesselName'] = 'G02-Lt_ICA'
        df_all_modeling_group2_r = reconstructVesselAcronym(df_all_modeling_group2_r)

        df_all_modeling_group3 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Rt_M1') |
            (df_all_modeling.VesselName == 'Rt_MCA_Inferior')
        ]
        df_all_modeling_group3_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Rt_M1') |
            (df_all_modeling.VesselName == 'Rt_MCA_Inferior')
        ]
        df_all_modeling_group3['VesselName'] = 'G03-Rt_Ant-MCA-Basal'
        df_all_modeling_group3_r = reconstructVesselAcronym(df_all_modeling_group3_r)

        df_all_modeling_group4 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Lt_M1') |
            (df_all_modeling.VesselName == 'Lt_MCA_Inferior')
        ]
        df_all_modeling_group4_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_Superior') |
            (df_all_modeling.VesselName == 'Lt_M1') |
            (df_all_modeling.VesselName == 'Lt_MCA_Inferior')
        ]
        df_all_modeling_group4['VesselName'] = 'G04-Lt_Ant-MCA-Basal'
        df_all_modeling_group4_r = reconstructVesselAcronym(df_all_modeling_group4_r)

        df_all_modeling_group5 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_lat_OFA') |
            (df_all_modeling.VesselName == 'Rt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Angular') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_MidTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntTemporal')
        ]
        df_all_modeling_group5_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_MCA_lat_OFA') |
            (df_all_modeling.VesselName == 'Rt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Rt_MCA_Angular') |
            (df_all_modeling.VesselName == 'Rt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_MidTemporal') |
            (df_all_modeling.VesselName == 'Rt_MCA_AntTemporal')
        ]
        df_all_modeling_group5['VesselName'] = 'G05-Rt_Ant-MCA-Pial'
        df_all_modeling_group5_r = reconstructVesselAcronym(df_all_modeling_group5_r)

        df_all_modeling_group6 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_lat_OFA') |
            (df_all_modeling.VesselName == 'Lt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_Angular') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_MidTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntTemporal')
        ]
        df_all_modeling_group6_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_MCA_lat_OFA') |
            (df_all_modeling.VesselName == 'Lt_MCA_PreRolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_Rolandic') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntPerietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostParietal') |
            (df_all_modeling.VesselName == 'Lt_MCA_Angular') |
            (df_all_modeling.VesselName == 'Lt_MCA_PostTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_MidTemporal') |
            (df_all_modeling.VesselName == 'Lt_MCA_AntTemporal')
        ]
        df_all_modeling_group6['VesselName'] = 'G06-Lt_Ant-MCA-Pial'
        df_all_modeling_group6_r = reconstructVesselAcronym(df_all_modeling_group6_r)

        df_all_modeling_group7 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A1') |
            (df_all_modeling.VesselName == 'Rt_A2') |
            (df_all_modeling.VesselName == 'Rt_A1+A2')
        ]
        df_all_modeling_group7_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_A1') |
            (df_all_modeling.VesselName == 'Rt_A2') |
            (df_all_modeling.VesselName == 'Rt_A1+A2')
        ]
        df_all_modeling_group7['VesselName'] = 'G07-Rt_Ant-ACA-Basal'
        df_all_modeling_group7_r = reconstructVesselAcronym(df_all_modeling_group7_r)

        df_all_modeling_group8 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_A1') |
            (df_all_modeling.VesselName == 'Lt_A2') |
            (df_all_modeling.VesselName == 'Lt_A1+A2')
        ]
        df_all_modeling_group8_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_A1') |
            (df_all_modeling.VesselName == 'Lt_A2') |
            (df_all_modeling.VesselName == 'Lt_A1+A2')
        ]
        df_all_modeling_group8['VesselName'] = 'G08-Lt_Ant-ACA-Basal'
        df_all_modeling_group8_r = reconstructVesselAcronym(df_all_modeling_group8_r)

        df_all_modeling_group9 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_ACA_med_OFA') |
            (df_all_modeling.VesselName == 'Rt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Rt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Rt_ACA_Pericallosal')
        ]
        df_all_modeling_group9_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_ACA_med_OFA') |
            (df_all_modeling.VesselName == 'Rt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Rt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Rt_ACA_Pericallosal')
        ]
        df_all_modeling_group9['VesselName'] = 'G09-Rt_Ant-ACA-Pial'
        df_all_modeling_group9_r = reconstructVesselAcronym(df_all_modeling_group9_r)

        df_all_modeling_group10 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ACA_med_OFA') |
            (df_all_modeling.VesselName == 'Lt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Lt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Lt_ACA_Pericallosal')
        ]
        df_all_modeling_group10_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ACA_med_OFA') |
            (df_all_modeling.VesselName == 'Lt_A2_Frontopolar') |
            (df_all_modeling.VesselName == 'Lt_ACA_Callosamarginal') |
            (df_all_modeling.VesselName == 'Lt_ACA_Pericallosal')
        ]
        df_all_modeling_group10['VesselName'] = 'G10-Lt_Ant-ACA-Pial'
        df_all_modeling_group10_r = reconstructVesselAcronym(df_all_modeling_group10_r)

        df_all_modeling_group11 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11['VesselName'] = 'G11-Rt_Post-VA'
        df_all_modeling_group11_r = reconstructVesselAcronym(df_all_modeling_group11_r)

        df_all_modeling_group12 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12['VesselName'] = 'G12-Lt_Post-VA'
        df_all_modeling_group12_r = reconstructVesselAcronym(df_all_modeling_group12_r)

        df_all_modeling_group13 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_P1') |
            (df_all_modeling.VesselName == 'Rt_P2') |
            (df_all_modeling.VesselName == 'Rt_P3,P4') |
            (df_all_modeling.VesselName == 'Rt_P1+P2')
        ]
        df_all_modeling_group13_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_P1') |
            (df_all_modeling.VesselName == 'Rt_P2') |
            (df_all_modeling.VesselName == 'Rt_P3,P4') |
            (df_all_modeling.VesselName == 'Rt_P1+P2')
        ]
        df_all_modeling_group13['VesselName'] = 'G13-Rt_Post-PCA-Basal'
        df_all_modeling_group13_r = reconstructVesselAcronym(df_all_modeling_group13_r)

        df_all_modeling_group14 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_P1') |
            (df_all_modeling.VesselName == 'Lt_P2') |
            (df_all_modeling.VesselName == 'Lt_P3,P4') |
            (df_all_modeling.VesselName == 'Lt_P1+P2')
        ]
        df_all_modeling_group14_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_P1') |
            (df_all_modeling.VesselName == 'Lt_P2') |
            (df_all_modeling.VesselName == 'Lt_P3,P4') |
            (df_all_modeling.VesselName == 'Lt_P1+P2')
        ]
        df_all_modeling_group14['VesselName'] = 'G14-Lt_Post-PCA-Basal'
        df_all_modeling_group14_r = reconstructVesselAcronym(df_all_modeling_group14_r)

        df_all_modeling_group15 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PCoA') |
            (df_all_modeling.VesselName == 'Rt_PPA') |
            (df_all_modeling.VesselName == 'Rt_Hippocampal artery') |
            (df_all_modeling.VesselName == 'Rt_PCA_lat_Pca') |
            (df_all_modeling.VesselName == 'Rt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Rt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group15_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PCoA') |
            (df_all_modeling.VesselName == 'Rt_PPA') |
            (df_all_modeling.VesselName == 'Rt_Hippocampal artery') |
            (df_all_modeling.VesselName == 'Rt_PCA_lat_Pca') |
            (df_all_modeling.VesselName == 'Rt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Rt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group15['VesselName'] = 'G15-Rt_Post-PCA-Pial'
        df_all_modeling_group15_r = reconstructVesselAcronym(df_all_modeling_group15_r)

        df_all_modeling_group16 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PCoA') |
            (df_all_modeling.VesselName == 'Lt_PPA') |
            (df_all_modeling.VesselName == 'Lt_Hippocampal artery') |
            (df_all_modeling.VesselName == 'Lt_PCA_lat_Pca') |
            (df_all_modeling.VesselName == 'Lt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Lt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group16_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PCoA') |
            (df_all_modeling.VesselName == 'Lt_PPA') |
            (df_all_modeling.VesselName == 'Lt_Hippocampal artery') |
            (df_all_modeling.VesselName == 'Lt_PCA_lat_Pca') |
            (df_all_modeling.VesselName == 'Lt_PCA_AnteriorTemporal') |
            (df_all_modeling.VesselName == 'Lt_PCA_PosteriorTemporal')
        ]
        df_all_modeling_group16['VesselName'] = 'G16-Lt_Post-PCA-Pial'
        df_all_modeling_group16_r = reconstructVesselAcronym(df_all_modeling_group16_r)

        df_all_modeling_group17 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PICA') |
            (df_all_modeling.VesselName == 'Rt_AICA') |
            (df_all_modeling.VesselName == 'Rt_SCA') |
            (df_all_modeling.VesselName == 'Rt_IAA')
        ]
        df_all_modeling_group17_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_PICA') |
            (df_all_modeling.VesselName == 'Rt_AICA') |
            (df_all_modeling.VesselName == 'Rt_SCA') |
            (df_all_modeling.VesselName == 'Rt_IAA')
        ]
        df_all_modeling_group17['VesselName'] = 'G17-Rt_SCA,AICA,PICA'
        df_all_modeling_group17_r = reconstructVesselAcronym(df_all_modeling_group17_r)

        df_all_modeling_group18 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PICA') |
            (df_all_modeling.VesselName == 'Lt_AICA') |
            (df_all_modeling.VesselName == 'Lt_SCA') |
            (df_all_modeling.VesselName == 'Lt_IAA')
        ]
        df_all_modeling_group18_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_PICA') |
            (df_all_modeling.VesselName == 'Lt_AICA') |
            (df_all_modeling.VesselName == 'Lt_SCA') |
            (df_all_modeling.VesselName == 'Lt_IAA')
        ]
        df_all_modeling_group18['VesselName'] = 'G18-Lt_SCA,AICA,PICA'
        df_all_modeling_group18_r = reconstructVesselAcronym(df_all_modeling_group18_r)

        df_all_modeling_group19 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19['VesselName'] = 'G19-BA'
        df_all_modeling_group19_r = reconstructVesselAcronym(df_all_modeling_group19_r)

        df_all_modeling_group20 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20['VesselName'] = 'G20-ACoA'
        df_all_modeling_group20_r = reconstructVesselAcronym(df_all_modeling_group20_r)

        df_all_modeling_group = pd.concat(
            [
                df_all_modeling_group1,
                df_all_modeling_group2,
                df_all_modeling_group3,
                df_all_modeling_group4,
                df_all_modeling_group5,
                df_all_modeling_group6,
                df_all_modeling_group7,
                df_all_modeling_group8,
                df_all_modeling_group9,
                df_all_modeling_group10,
                df_all_modeling_group11,
                df_all_modeling_group12,
                df_all_modeling_group13,
                df_all_modeling_group14,
                df_all_modeling_group15,
                df_all_modeling_group16,
                df_all_modeling_group17,
                df_all_modeling_group18,
                df_all_modeling_group19,
                df_all_modeling_group20
            ], ignore_index=True, axis=0
        )
        
        df_all_modeling_group_r = pd.concat(
            [
                df_all_modeling_group1_r,
                df_all_modeling_group2_r,
                df_all_modeling_group3_r,
                df_all_modeling_group4_r,
                df_all_modeling_group5_r,
                df_all_modeling_group6_r,
                df_all_modeling_group7_r,
                df_all_modeling_group8_r,
                df_all_modeling_group9_r,
                df_all_modeling_group10_r,
                df_all_modeling_group11_r,
                df_all_modeling_group12_r,
                df_all_modeling_group13_r,
                df_all_modeling_group14_r,
                df_all_modeling_group15_r,
                df_all_modeling_group16_r,
                df_all_modeling_group17_r,
                df_all_modeling_group18_r,
                df_all_modeling_group19_r,
                df_all_modeling_group20_r
            ], ignore_index=True, axis=0
        )

        '''
        classes = pd.DataFrame(df_all_modeling_group.VesselName.value_counts()).index.tolist()
        classes = sorted(classes, key=lambda x: int(x.split('-')[0][1:]), reverse=False)
        classes_r = sorted(classes, key=lambda x: int(x.split('-')[0][1:]), reverse=True)
        classes_mirror = sortSymmetry(classes)
        classes_r_mirror = sortSymmetry(classes_r)
        '''
        X = df_all_modeling_group
        X_r = df_all_modeling_group_r
    
    if n_components == '20C':
        REPLACEMENTS = {
            'ACoA': 'A0.01',
            'Rt_ICA': 'A1.01',
            'Rt_Ophthalmic': 'A1.02',
            'Lt_ICA': 'A2.01',
            'Lt_Ophthalmic': 'A2.02',
            'Rt_M1': 'A3.01',
            'Rt_MCA_Superior': 'A3.02',
            'Rt_MCA_Inferior': 'A3.03',
            'Lt_M1': 'A4.01',
            'Lt_MCA_Superior': 'A4.02',
            'Lt_MCA_Inferior': 'A4.03',
            'Rt_A1,A2': 'A5.01',
            'Lt_A1,A2': 'A6.01',
            'Rt_MCA_lat_OFA': 'A7.01',
            'Rt_MCA_PreRolandic': 'A7.02',
            'Rt_MCA_Rolandic': 'A7.03',
            'Rt_MCA_AntPerietal': 'A7.04',
            'Rt_MCA_PostParietal': 'A7.05',
            'Rt_MCA_Angular': 'A7.06',
            'Rt_MCA_PostTemporal': 'A7.07',
            'Rt_MCA_MidTemporal': 'A7.08',
            'Rt_MCA_AntTemporal': 'A7.09',
            'Lt_MCA_lat_OFA': 'A8.01',
            'Lt_MCA_PreRolandic': 'A8.02',
            'Lt_MCA_Rolandic': 'A8.03',
            'Lt_MCA_AntPerietal': 'A8.04',
            'Lt_MCA_PostParietal': 'A8.05',
            'Lt_MCA_Angular': 'A8.06',
            'Lt_MCA_PostTemporal': 'A8.07',
            'Lt_MCA_MidTemporal': 'A8.08',
            'Lt_MCA_AntTemporal': 'A8.09',
            'Rt_ACA_med_OFA': 'A9.01',
            'Rt_A2_Frontopolar': 'A9.02',
            'Rt_ACA_Callosamarginal': 'A9.03',
            'Rt_ACA_Pericallosal': 'A9.04',
            'Lt_ACA_med_OFA': 'A10.01',
            'Lt_A2_Frontopolar': 'A10.02',
            'Lt_ACA_Callosamarginal': 'A10.03',
            'Lt_ACA_Pericallosal': 'A10.04',
            'Rt_VA': 'P1.01',
            'Lt_VA': 'P2.01',
            'Rt_P1,P2,P3': 'P3.01',
            'Lt_P1,P2,P3': 'P4.01',
            'Rt_PCoA': 'P5.01',
            'Rt_Hippocampal_artery': 'P5.02',
            'Rt_PCA_AnteariorTemporal': 'P5.03',
            'Rt_PCA_PosteriorTemporal': 'P5.04',
            'Rt_Parieto_occipital': 'P5.05',
            'Rt_Calcarine': 'P5.06',
            'Lt_PCoA': 'P6.01',
            'Lt_Hippocampal_artery': 'P6.02',
            'Lt_PCA_AnteriorTemporal': 'P6.03',
            'Lt_PCA_PosteriorTemporal': 'P6.04',
            'Lt_Parieto_occipital': 'P6.05',
            'Lt_Calcarine': 'P6.06',
            'Rt_PICA': 'P7.01',
            'Rt_AICA': 'P7.02',
            'Rt_SCA': 'P7.03',
            'Lt_PICA': 'P8.01',
            'Lt_AICA': 'P8.02',
            'Lt_SCA': 'P8.03',
            'BA': 'P0.01'
        }
        df_all_modeling = feature_vector
        df_all_modeling.replace(REPLACEMENTS, inplace=True)

        df_all_modeling_group1 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A1.01') |
            (df_all_modeling.VesselName == 'A1.02')
        ]
        df_all_modeling_group1_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A1.01') |
            (df_all_modeling.VesselName == 'A1.02')
        ]
        df_all_modeling_group1['VesselName'] = 'A1'

        df_all_modeling_group2 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A2.01') |
            (df_all_modeling.VesselName == 'A2.02')
        ]
        df_all_modeling_group2_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A2.01') |
            (df_all_modeling.VesselName == 'A2.02')
        ]
        df_all_modeling_group2['VesselName'] = 'A2'

        df_all_modeling_group3 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A3.01') |
            (df_all_modeling.VesselName == 'A3.02') |
            (df_all_modeling.VesselName == 'A3.03')
        ]
        df_all_modeling_group3_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A3.01') |
            (df_all_modeling.VesselName == 'A3.02') |
            (df_all_modeling.VesselName == 'A3.03')
        ]
        df_all_modeling_group3['VesselName'] = 'A3'

        df_all_modeling_group4 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A4.01') |
            (df_all_modeling.VesselName == 'A4.02') |
            (df_all_modeling.VesselName == 'A4.03')
        ]
        df_all_modeling_group4_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A4.01') |
            (df_all_modeling.VesselName == 'A4.02') |
            (df_all_modeling.VesselName == 'A4.03')
        ]
        df_all_modeling_group4['VesselName'] = 'A4'

        df_all_modeling_group5 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A7.01') |
            (df_all_modeling.VesselName == 'A7.02') |
            (df_all_modeling.VesselName == 'A7.03') |
            (df_all_modeling.VesselName == 'A7.04') |
            (df_all_modeling.VesselName == 'A7.05') |
            (df_all_modeling.VesselName == 'A7.06') |
            (df_all_modeling.VesselName == 'A7.07') |
            (df_all_modeling.VesselName == 'A7.08') |
            (df_all_modeling.VesselName == 'A7.09')
        ]
        df_all_modeling_group5_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A7.01') |
            (df_all_modeling.VesselName == 'A7.02') |
            (df_all_modeling.VesselName == 'A7.03') |
            (df_all_modeling.VesselName == 'A7.04') |
            (df_all_modeling.VesselName == 'A7.05') |
            (df_all_modeling.VesselName == 'A7.06') |
            (df_all_modeling.VesselName == 'A7.07') |
            (df_all_modeling.VesselName == 'A7.08') |
            (df_all_modeling.VesselName == 'A7.09')
        ]
        df_all_modeling_group5['VesselName'] = 'A7'

        df_all_modeling_group6 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A8.01') |
            (df_all_modeling.VesselName == 'A8.02') |
            (df_all_modeling.VesselName == 'A8.03') |
            (df_all_modeling.VesselName == 'A8.04') |
            (df_all_modeling.VesselName == 'A8.05') |
            (df_all_modeling.VesselName == 'A8.06') |
            (df_all_modeling.VesselName == 'A8.07') |
            (df_all_modeling.VesselName == 'A8.08') |
            (df_all_modeling.VesselName == 'A8.09')
        ]
        df_all_modeling_group6_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A8.01') |
            (df_all_modeling.VesselName == 'A8.02') |
            (df_all_modeling.VesselName == 'A8.03') |
            (df_all_modeling.VesselName == 'A8.04') |
            (df_all_modeling.VesselName == 'A8.05') |
            (df_all_modeling.VesselName == 'A8.06') |
            (df_all_modeling.VesselName == 'A8.07') |
            (df_all_modeling.VesselName == 'A8.08') |
            (df_all_modeling.VesselName == 'A8.09')
        ]
        df_all_modeling_group6['VesselName'] = 'A8'

        df_all_modeling_group7 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A5.01')
        ]
        df_all_modeling_group7_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A5.01')
        ]
        df_all_modeling_group7['VesselName'] = 'A5'

        df_all_modeling_group8 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A6.01')
        ]
        df_all_modeling_group8_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A6.01')
        ]
        df_all_modeling_group8['VesselName'] = 'A6'

        df_all_modeling_group9 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A9.01') |
            (df_all_modeling.VesselName == 'A9.02') |
            (df_all_modeling.VesselName == 'A9.03') |
            (df_all_modeling.VesselName == 'A9.04')
        ]
        df_all_modeling_group9_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A9.01') |
            (df_all_modeling.VesselName == 'A9.02') |
            (df_all_modeling.VesselName == 'A9.03') |
            (df_all_modeling.VesselName == 'A9.04')
        ]
        df_all_modeling_group9['VesselName'] = 'A9'

        df_all_modeling_group10 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A10.01') |
            (df_all_modeling.VesselName == 'A10.02') |
            (df_all_modeling.VesselName == 'A10.03') |
            (df_all_modeling.VesselName == 'A10.04')
        ]
        df_all_modeling_group10_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A10.01') |
            (df_all_modeling.VesselName == 'A10.02') |
            (df_all_modeling.VesselName == 'A10.03') |
            (df_all_modeling.VesselName == 'A10.04')
        ]
        df_all_modeling_group10['VesselName'] = 'A10'

        df_all_modeling_group11 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P1.01')
        ]
        df_all_modeling_group11_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P1.01')
        ]
        df_all_modeling_group11['VesselName'] = 'P1'

        df_all_modeling_group12 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P2.01')
        ]
        df_all_modeling_group12_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P2.01')
        ]
        df_all_modeling_group12['VesselName'] = 'P2'

        df_all_modeling_group13 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P3.01')
        ]
        df_all_modeling_group13_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P3.01')
        ]
        df_all_modeling_group13['VesselName'] = 'P3'

        df_all_modeling_group14 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P4.01')
        ]
        df_all_modeling_group14_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P4.01')
        ]
        df_all_modeling_group14['VesselName'] = 'P4'

        df_all_modeling_group15 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P5.01') |
            (df_all_modeling.VesselName == 'P5.02') |
            (df_all_modeling.VesselName == 'P5.03') |
            (df_all_modeling.VesselName == 'P5.04') |
            (df_all_modeling.VesselName == 'P5.05') |
            (df_all_modeling.VesselName == 'P5.06')
        ]
        df_all_modeling_group15_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P5.01') |
            (df_all_modeling.VesselName == 'P5.02') |
            (df_all_modeling.VesselName == 'P5.03') |
            (df_all_modeling.VesselName == 'P5.04') |
            (df_all_modeling.VesselName == 'P5.05') |
            (df_all_modeling.VesselName == 'P5.06')
        ]
        df_all_modeling_group15['VesselName'] = 'P5'

        df_all_modeling_group16 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P6.01') |
            (df_all_modeling.VesselName == 'P6.02') |
            (df_all_modeling.VesselName == 'P6.03') |
            (df_all_modeling.VesselName == 'P6.04') |
            (df_all_modeling.VesselName == 'P6.05') |
            (df_all_modeling.VesselName == 'P6.06')
        ]
        df_all_modeling_group16_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P6.01') |
            (df_all_modeling.VesselName == 'P6.02') |
            (df_all_modeling.VesselName == 'P6.03') |
            (df_all_modeling.VesselName == 'P6.04') |
            (df_all_modeling.VesselName == 'P6.05') |
            (df_all_modeling.VesselName == 'P6.06')
        ]
        df_all_modeling_group16['VesselName'] = 'P6'

        df_all_modeling_group17 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P7.01') |
            (df_all_modeling.VesselName == 'P7.02') |
            (df_all_modeling.VesselName == 'P7.03')
        ]
        df_all_modeling_group17_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P7.01') |
            (df_all_modeling.VesselName == 'P7.02') |
            (df_all_modeling.VesselName == 'P7.03')
        ]
        df_all_modeling_group17['VesselName'] = 'P7'

        df_all_modeling_group18 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P8.01') |
            (df_all_modeling.VesselName == 'P8.02') |
            (df_all_modeling.VesselName == 'P8.03')
        ]
        df_all_modeling_group18_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P8.01') |
            (df_all_modeling.VesselName == 'P8.02') |
            (df_all_modeling.VesselName == 'P8.03')
        ]
        df_all_modeling_group18['VesselName'] = 'P8'

        df_all_modeling_group19 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P0.01')
        ]
        df_all_modeling_group19_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'P0.01')
        ]
        df_all_modeling_group19['VesselName'] = 'P0'

        df_all_modeling_group20 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A0.01')
        ]
        df_all_modeling_group20_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'A0.01')
        ]
        df_all_modeling_group20['VesselName'] = 'A0'

        df_all_modeling_group = pd.concat(
            [
                df_all_modeling_group1,
                df_all_modeling_group2,
                df_all_modeling_group3,
                df_all_modeling_group4,
                df_all_modeling_group5,
                df_all_modeling_group6,
                df_all_modeling_group7,
                df_all_modeling_group8,
                df_all_modeling_group9,
                df_all_modeling_group10,
                df_all_modeling_group11,
                df_all_modeling_group12,
                df_all_modeling_group13,
                df_all_modeling_group14,
                df_all_modeling_group15,
                df_all_modeling_group16,
                df_all_modeling_group17,
                df_all_modeling_group18,
                df_all_modeling_group19,
                df_all_modeling_group20
            ], ignore_index=True, axis=0
        )
        
        df_all_modeling_group_r = pd.concat(
            [
                df_all_modeling_group1_r,
                df_all_modeling_group2_r,
                df_all_modeling_group3_r,
                df_all_modeling_group4_r,
                df_all_modeling_group5_r,
                df_all_modeling_group6_r,
                df_all_modeling_group7_r,
                df_all_modeling_group8_r,
                df_all_modeling_group9_r,
                df_all_modeling_group10_r,
                df_all_modeling_group11_r,
                df_all_modeling_group12_r,
                df_all_modeling_group13_r,
                df_all_modeling_group14_r,
                df_all_modeling_group15_r,
                df_all_modeling_group16_r,
                df_all_modeling_group17_r,
                df_all_modeling_group18_r,
                df_all_modeling_group19_r,
                df_all_modeling_group20_r
            ], ignore_index=True, axis=0
        )

        '''
        classes = pd.DataFrame(df_all_modeling_group.VesselName.value_counts()).index.tolist()
        classes = sorted(classes, key=lambda x: int(x.split('-')[0][1:]), reverse=False)
        classes_r = sorted(classes, key=lambda x: int(x.split('-')[0][1:]), reverse=True)
        classes_mirror = sortSymmetry(classes)
        classes_r_mirror = sortSymmetry(classes_r)
        '''
        X = df_all_modeling_group
        X_r = df_all_modeling_group_r
    
    if n_components == '67':
        df_all_modeling = feature_vector
        
    return (
        X, X_r,
        df_all_modeling_group1, df_all_modeling_group1_r,
        df_all_modeling_group2, df_all_modeling_group2_r,
        df_all_modeling_group3, df_all_modeling_group3_r,
        df_all_modeling_group4, df_all_modeling_group4_r,
        df_all_modeling_group5, df_all_modeling_group5_r,
        df_all_modeling_group6, df_all_modeling_group6_r,
        df_all_modeling_group7, df_all_modeling_group7_r,
        df_all_modeling_group8, df_all_modeling_group8_r,
        df_all_modeling_group9, df_all_modeling_group9_r,
        df_all_modeling_group10, df_all_modeling_group10_r,
        df_all_modeling_group11, df_all_modeling_group11_r,
        df_all_modeling_group12, df_all_modeling_group12_r,
        df_all_modeling_group13, df_all_modeling_group13_r,
        df_all_modeling_group14, df_all_modeling_group14_r,
        df_all_modeling_group15, df_all_modeling_group15_r,
        df_all_modeling_group16, df_all_modeling_group16_r,
        df_all_modeling_group17, df_all_modeling_group17_r,
        df_all_modeling_group18, df_all_modeling_group18_r,
        df_all_modeling_group19, df_all_modeling_group19_r,
        df_all_modeling_group20, df_all_modeling_group20_r
    )


def splitVesselGroup(feature_vector_group, n_components):
    if n_components == 20:
        df_all_modeling = feature_vector_group
        
        df_all_modeling_group1 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G01-Rt_ICA')]
        df_all_modeling_group2 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G02-Lt_ICA')]
        df_all_modeling_group3 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G03-Rt_Ant-MCA-Basal')]
        df_all_modeling_group4 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G04-Lt_Ant-MCA-Basal')]
        df_all_modeling_group5 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G05-Rt_Ant-MCA-Pial')]
        df_all_modeling_group6 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G06-Lt_Ant-MCA-Pial')]
        df_all_modeling_group7 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G07-Rt_Ant-ACA-Basal')]
        df_all_modeling_group8 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G08-Lt_Ant-ACA-Basal')]
        df_all_modeling_group9 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G09-Rt_Ant-ACA-Pial')]
        df_all_modeling_group10 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G10-Lt_Ant-ACA-Pial')]
        df_all_modeling_group11 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G11-Rt_Post-VA')]
        df_all_modeling_group12 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G12-Lt_Post-VA')]
        df_all_modeling_group13 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G13-Rt_Post-PCA-Basal')]
        df_all_modeling_group14 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G14-Lt_Post-PCA-Basal')]
        df_all_modeling_group15 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G15-Rt_Post-PCA-Pial')]
        df_all_modeling_group16 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G16-Lt_Post-PCA-Pial')]
        df_all_modeling_group17 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G17-Rt_SCA,AICA,PICA')]
        df_all_modeling_group18 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G18-Lt_SCA,AICA,PICA')]
        df_all_modeling_group19 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G19-BA')]
        df_all_modeling_group20 = df_all_modeling.loc[(df_all_modeling.VesselName == 'G20-ACoA')]
        
    return (
        df_all_modeling_group1,
        df_all_modeling_group2,
        df_all_modeling_group3,
        df_all_modeling_group4,
        df_all_modeling_group5,
        df_all_modeling_group6,
        df_all_modeling_group7,
        df_all_modeling_group8,
        df_all_modeling_group9,
        df_all_modeling_group10,
        df_all_modeling_group11,
        df_all_modeling_group12,
        df_all_modeling_group13,
        df_all_modeling_group14,
        df_all_modeling_group15,
        df_all_modeling_group16,
        df_all_modeling_group17,
        df_all_modeling_group18,
        df_all_modeling_group19,
        df_all_modeling_group20
    )


def saveModel(model, tag_model):
    tag_model = tag_model + '.pkl'
    with open(tag_model, 'wb') as file:  
        pickle.dump(model, file)


def loadModel(tag_model):
    tag_model = tag_model + '.pkl'
    with open(tag_model, 'rb') as file:
        model = pickle.load(file)
    return model


def processFiles(subpath, filetype, index):
    if index == 0:
        PATH = os.getcwd() + subpath
        FILES = os.listdir(PATH)
        FILES = glob.glob(PATH + '/*.' + filetype)
    if index == 1:
        PATH = os.getcwd() + subpath
        FILES = os.listdir(PATH)
        FILES = glob.glob(PATH + '/**/**.' + filetype)
    return natsorted(FILES)


def applyCerebrovascularClassifier(
    FILES, path_model_macrovessel, path_model_microvessel, scaler_g20_all,
    scaler_g20_1, scaler_g20_2, scaler_g20_3, scaler_g20_4, scaler_g20_5,
    scaler_g20_6, scaler_g20_7, scaler_g20_8, scaler_g20_9, scaler_g20_10,
    scaler_g20_11, scaler_g20_12, scaler_g20_13, scaler_g20_14, scaler_g20_15,
    scaler_g20_16, scaler_g20_17, scaler_g20_18, scaler_g20_19, scaler_g20_20
):
    '''
    FILES = src.processFiles('/data_NoMatch', 'xlsx', 1)
    model_MLP_g20_all = src.loadModel('data_model/clf_MLP_HL2000_g20_all')
    for idx_model in range(1, 21):
        locals()['model_MLP_g20_' + str(idx_model)] = src.loadModel('data_model/clf_MLP_HL2000_g20_' + str(idx_model))
    y_pred_voted_all = pd.DataFrame()
    for files in FILES:
        file = pd.read_excel(files)
        tag_file = files.split('/')[-1].split('.')[0]
        file = file.loc[:, ~file.columns.str.contains('^Unnamed')]
        if file.shape[1] == 16:
            df_all_modeling = file.iloc[:, np.r_[0:3, 6:len(file.columns)]].apply(scipy.stats.zscore)
        elif file.shape[1] == 19:
            df_all_modeling = file.iloc[:, np.r_[0:3, 9:len(file.columns)]].apply(scipy.stats.zscore)
        df_all_spot = file.iloc[:, np.r_[0:4, 5]]
        X_new = pd.DataFrame(np.nan_to_num(df_all_modeling))
        y_pred_new = scaler_g20_all.inverse_transform(model_MLP_g20_all.predict(X_new))
        y_pred_new = pd.DataFrame(y_pred_new.reshape(-1, 1))
        y_pred_new_all = pd.concat(
            [df_all_modeling, df_all_spot, y_pred_new],
            ignore_index=True, axis=1
        )
        y_pred_new_all.columns = df_all_modeling.columns.tolist() + ['X', 'Y', 'Z', 'PointID', 'NewGroupID', 'VesselName']
        tag_center = y_pred_new_all.loc[(y_pred_new_all.VesselName == 'G19-BA')].iloc[:, 1].mean()
        y_pred_new_all_voted = pd.DataFrame()
        for idx_NewGroupID in range(1, np.max(y_pred_new_all.NewGroupID)+1):
            y_pred_new_all_NewGroupID = y_pred_new_all.loc[(y_pred_new_all.NewGroupID == idx_NewGroupID)]
            for idx_NewGroupID_y in range(y_pred_new_all_NewGroupID.shape[0]):
                data_y = y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, 1]
                if data_y < tag_center:
                    if y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'G02-Lt_ICA':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'G01-Rt_ICA'
                    elif y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'G04-Lt_Ant-MCA-Basal':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'G03-Rt_Ant-MCA-Basal'
                    elif y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'G06-Lt_Ant-MCA-Pial':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'G05-Rt_Ant-MCA-Pial'
                elif data_y > tag_center:
                    if y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'G01-Rt_ICA':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'G02-Lt_ICA'
                    elif y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'G03-Rt_Ant-MCA-Basal':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'G04-Lt_Ant-MCA-Basal'
                    elif y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'G05-Rt_Ant-MCA-Pial':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'G06-Lt_Ant-MCA-Pial'
            voting_macrovessel = src.sortMostCommon(y_pred_new_all_NewGroupID.iloc[:, -1])
            y_pred_new_all_NewGroupID.VesselName = voting_macrovessel
            tag_scaler_g20 = str(int(voting_macrovessel.split('-')[0][1:]))
            y_pred_new_all_NewGroupID['MicroVesselName'] = eval('scaler_g20_' + tag_scaler_g20).inverse_transform(
                eval('model_MLP_g20_' + tag_scaler_g20).predict(
                    y_pred_new_all_NewGroupID.iloc[:, :-6]
                )
            )
            for idx_NewGroupID_y in range(y_pred_new_all_NewGroupID.shape[0]):
                data_y = y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, 1]
                if data_y < tag_center:
                    if y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'Lt_P2':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'Rt_P2'
                elif data_y > tag_center:
                    if y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'Rt_P2':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'Lt_P2'
            voting_microvessel = src.sortMostCommon(y_pred_new_all_NewGroupID.iloc[:, -1])
            y_pred_new_all_NewGroupID.MicroVesselName = voting_microvessel
            y_pred_new_all_voted = pd.concat(
                [y_pred_new_all_voted, y_pred_new_all_NewGroupID],
                ignore_index=True, axis=0
            )
        y_pred_new_all_voted.iloc[:, -7:].to_csv('data_pred/MLP/y_pred_voted_' + tag_file + '.csv')
        y_pred_voted_all = pd.concat(
            [y_pred_voted_all, y_pred_new_all_voted],
            ignore_index=True, axis=0
        )
    (
        y_pred_voted_all_group1, y_pred_voted_all_group2,
        y_pred_voted_all_group3, y_pred_voted_all_group4,
        y_pred_voted_all_group5, y_pred_voted_all_group6,
        y_pred_voted_all_group7, y_pred_voted_all_group8,
        y_pred_voted_all_group9, y_pred_voted_all_group10,
        y_pred_voted_all_group11, y_pred_voted_all_group12,
        y_pred_voted_all_group13, y_pred_voted_all_group14,
        y_pred_voted_all_group15, y_pred_voted_all_group16,
        y_pred_voted_all_group17, y_pred_voted_all_group18,
        y_pred_voted_all_group19, y_pred_voted_all_group20
    ) = src.splitVesselGroup(y_pred_voted_all, 20)
    '''
    model_g20_all = loadModel(path_model_macrovessel)
    for idx_model in range(1, 21):
        locals()['model_g20_' + str(idx_model)] = loadModel(path_model_microvessel + str(idx_model))
    
    y_pred_voted_all = pd.DataFrame()
    for files in FILES:
        file = pd.read_excel(files)
        tag_file = files.split('/')[-1].split('.')[0]
        file = file.loc[:, ~file.columns.str.contains('^Unnamed')]
        if file.shape[1] == 16:
            df_all_modeling = file.iloc[:, np.r_[0:3, 6:len(file.columns)]].apply(scipy.stats.zscore)
        elif file.shape[1] == 19:
            df_all_modeling = file.iloc[:, np.r_[0:3, 9:len(file.columns)]].apply(scipy.stats.zscore)
            
            
        df_all_spot = file.iloc[:, 5]
        X_new = pd.DataFrame(np.nan_to_num(df_all_modeling))
        
        
        y_pred_new = scaler_g20_all.inverse_transform(model_g20_all.predict(X_new))
        y_pred_new = pd.DataFrame(y_pred_new.reshape(-1, 1))
        y_pred_new_all = pd.concat(
            [df_all_modeling, df_all_spot, y_pred_new],
            ignore_index=True, axis=1
        )
        y_pred_new_all.columns = df_all_modeling.columns.tolist() + ['NewGroupID', 'VesselName']
        tag_center = y_pred_new_all.loc[(y_pred_new_all.VesselName == 'G19-BA')].iloc[:, 1].mean()
        y_pred_new_all_voted = pd.DataFrame()
        for idx_NewGroupID in range(1, np.max(y_pred_new_all.NewGroupID)+1):
            y_pred_new_all_NewGroupID = y_pred_new_all.loc[(y_pred_new_all.NewGroupID == idx_NewGroupID)]
            for idx_NewGroupID_y in range(y_pred_new_all_NewGroupID.shape[0]):
                data_y = y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, 1]
                if data_y < tag_center:
                    if y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'G02-Lt_ICA':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'G01-Rt_ICA'
                    elif y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'G04-Lt_Ant-MCA-Basal':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'G03-Rt_Ant-MCA-Basal'
                    elif y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'G06-Lt_Ant-MCA-Pial':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'G05-Rt_Ant-MCA-Pial'
                elif data_y > tag_center:
                    if y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'G01-Rt_ICA':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'G02-Lt_ICA'
                    elif y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'G03-Rt_Ant-MCA-Basal':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'G04-Lt_Ant-MCA-Basal'
                    elif y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'G05-Rt_Ant-MCA-Pial':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'G06-Lt_Ant-MCA-Pial'
            voting_macrovessel = sortMostCommon(y_pred_new_all_NewGroupID.iloc[:, -1])
            y_pred_new_all_NewGroupID.VesselName = voting_macrovessel
            tag_scaler_g20 = str(int(voting_macrovessel.split('-')[0][1:]))
            y_pred_new_all_NewGroupID['MicroVesselName'] = eval('scaler_g20_' + tag_scaler_g20).inverse_transform(
                eval('model_g20_' + tag_scaler_g20).predict(
                    y_pred_new_all_NewGroupID.iloc[:, :-2]
                )
            )
            for idx_NewGroupID_y in range(y_pred_new_all_NewGroupID.shape[0]):
                data_y = y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, 1]
                if data_y < tag_center:
                    if y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'Lt_P2':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'Rt_P2'
                elif data_y > tag_center:
                    if y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] == 'Rt_P2':
                        y_pred_new_all_NewGroupID.iloc[idx_NewGroupID_y, -1] = 'Lt_P2'
            voting_microvessel = sortMostCommon(y_pred_new_all_NewGroupID.iloc[:, -1])
            y_pred_new_all_NewGroupID.MicroVesselName = voting_microvessel
            y_pred_new_all_voted = pd.concat(
                [y_pred_new_all_voted, y_pred_new_all_NewGroupID],
                ignore_index=True, axis=0
            )
        # y_pred_new_all_voted.iloc[:, -3:].drop_duplicates().to_csv('data_pred/MLP/y_pred_voted_' + tag_file + '.csv')
        # y_pred_new_all_voted.iloc[:, -3:].drop_duplicates().to_excel('data_pred/MLP/y_pred_voted_' + tag_file + '.xlsx')
        # y_pred_new_all_voted.iloc[:, -3:].to_excel('data_nomatch/control_pred/y_pred_voted_' + tag_file + '.xlsx')
        y_pred_new_all_voted.to_excel('data_nomatch/control_pred/y_pred_voted_' + tag_file + '.xlsx')
        y_pred_voted_all = pd.concat(
            [y_pred_voted_all, y_pred_new_all_voted],
            ignore_index=True, axis=0
        )
    (
        y_pred_voted_all_group1, y_pred_voted_all_group2,
        y_pred_voted_all_group3, y_pred_voted_all_group4,
        y_pred_voted_all_group5, y_pred_voted_all_group6,
        y_pred_voted_all_group7, y_pred_voted_all_group8,
        y_pred_voted_all_group9, y_pred_voted_all_group10,
        y_pred_voted_all_group11, y_pred_voted_all_group12,
        y_pred_voted_all_group13, y_pred_voted_all_group14,
        y_pred_voted_all_group15, y_pred_voted_all_group16,
        y_pred_voted_all_group17, y_pred_voted_all_group18,
        y_pred_voted_all_group19, y_pred_voted_all_group20
    ) = splitVesselGroup(y_pred_voted_all, 20)


def processHeaderInformation(path, data):
    '''
    - AcquisitionDate
    - StudyDate
    - Manufacturer
    - InstitutionName
    - StudyDescription
    - SeriesDescription
    - SequenceName
    - ProtocolName
    - MagneticFieldStrength
    - PixelSpacing
    - Rows
    - Columns
    - SliceThickness
    - SpacingBetweenSlices
    - RepetitionTime
    - EchoTime
    etc.
    '''
    
    PATH = os.getcwd() + path
    FILES = os.listdir(PATH)
    FILES = glob.glob(PATH + data)

    (
        files_all, plane_all, scale_all_x, scale_all_y, scale_all,
        study_date_all, study_description_all,
        series_description_all, magnetic_field_strength_all,
        rows_all, columns_all, slice_thickness_all, spacing_between_slices_all,
        repetition_time_all, echo_time_all
    ) = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for files in FILES:
        files_all.append(files.split('/')[-1].split('_')[1])
        plane = findPlaneOrientation(files)
        plane_all.append(plane)
        scale_all_x.append(pydicom.read_file(files).PixelSpacing[0])
        scale_all_y.append(pydicom.read_file(files).PixelSpacing[1])
        study_date_all.append(pydicom.read_file(files).StudyDate)
        study_description_all.append(pydicom.read_file(files).StudyDescription)
        series_description_all.append(pydicom.read_file(files).SeriesDescription)
        magnetic_field_strength_all.append(pydicom.read_file(files).MagneticFieldStrength)
        rows_all.append(pydicom.read_file(files).Rows)
        columns_all.append(pydicom.read_file(files).Columns)
        slice_thickness_all.append(pydicom.read_file(files).SliceThickness)
        spacing_between_slices_all.append(pydicom.read_file(files).SpacingBetweenSlices)
        repetition_time_all.append(pydicom.read_file(files).RepetitionTime)
        echo_time_all.append(pydicom.read_file(files).EchoTime)
        scale_all.append(pydicom.read_file(files).PixelSpacing[0])
        scale_all.append(pydicom.read_file(files).PixelSpacing[1])
    
    files_profile = pd.concat(
        [
            pd.DataFrame(files_all), pd.DataFrame(plane_all),
            pd.DataFrame(scale_all_x), pd.DataFrame(scale_all_y),
            pd.DataFrame(study_date_all),
            pd.DataFrame(study_description_all), pd.DataFrame(series_description_all),
            pd.DataFrame(magnetic_field_strength_all), pd.DataFrame(rows_all), pd.DataFrame(columns_all),
            pd.DataFrame(slice_thickness_all), pd.DataFrame(spacing_between_slices_all),
            pd.DataFrame(repetition_time_all), pd.DataFrame(echo_time_all)
        ],
        ignore_index=True, axis=1
    )
    
    files_profile.columns = [
        'ID', 'PlaneOrientation', 'PixelSpacingX', 'PixelSpacingY',
        'StudyDate', 'StudyDescription', 'SeriesDescription', 'MagneticFieldStrength',
        'Rows', 'Columns', 'SliceThickness', 'SpacingBetweenSlices', 'RepetitionTime', 'EchoTime'
    ]
    
    files_profile_anomaly = files_profile.loc[
        (round(files_profile.PixelSpacingX, 2) != 0.28) |
        (round(files_profile.PixelSpacingY, 2) != 0.28)
    ]
    
    return files_profile, files_profile_anomaly

# Selective algorithm for DICOM files

'''
PATH = '/Volumes/MyBookDuo/MASTER/Control/Control2000'
# FILES = glob.glob(PATH + '/**/**/**_TOF/*000.DCM')
FILES = glob.glob(PATH + '/**/**_TOF/*000.DCM')
for files in FILES:
    source = files
    destination = os.getcwd() + '/data_raw/control_' + files.split('/')[-2] + '_' + files.split('/')[-1][:-4] + '.dcm'
    copyfile(source, destination)

PATH = '/Volumes/MyBookDuo/MASTER/Patient/Mysteri_2019/DeID_Mysteri_2019/Gene_Patient_new'
# PATH = '/Volumes/My Passport_red/SMC_RED/SHN/S_deID'
# PATH = '/Volumes/My Passport_red/SMC_RED/A/02_deID_DICOM'
FILES = glob.glob(PATH + '/**/TOF/*0001.dcm')
for files in FILES:
    source = files
    destination = os.getcwd() + '/data_raw/stroke_' + files.split('/')[-3].split('_')[-1] + '_TOF_' + files.split('/')[-1][:-4] + '.dcm'
    copyfile(source, destination)
'''

# Copying specific types of files into another destination

'''
CONTROL455 = os.getcwd() + '/data_clinical/control455_20210803.xlsx'
tag_control455 = pd.read_excel(CONTROL455).iloc[:, 0].to_list()
PATH = os.getcwd() + '/data_NoMatch/control'
FILES = os.listdir(PATH)
FILES = glob.glob(PATH + '/*dataclearing_X02841.xlsx')
os.mkdir(os.getcwd() + '/control455')

for files in FILES:
    switch_control455 = files.split('/')[-1].split('_')[1]
    if int(switch_control455) in tag_control455:
        source = files
        destination = os.getcwd() + '/control455/' + files.split('/')[-1]
        copyfile(source, destination)
'''