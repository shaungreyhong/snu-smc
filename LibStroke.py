import pandas as pd


const_replacement = {
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
    'Rt_Hippocampal artery': 'P5.02',
    'Rt_Hippocampal_artery': 'P5.02',
    'Rt_PCA_AnteriorTemporal': 'P5.03',
    'Rt_PCA_PosteriorTemporal': 'P5.04',
    'Rt_Parieto_occipital': 'P5.05',
    'Rt_Calcarine': 'P5.06',
    'Lt_PCoA': 'P6.01',
    'Lt_Hippocampal artery': 'P6.02',
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

const_replacement_r = {
    'A0': 'ACoA', # Anterior communicating artery
    'A1': 'RtICA',
    'A2': 'LtICA',
    'A3': 'RtBasalMCA',
    'A4': 'LtBasalMCA',
    'A5': 'RtBasalACA',
    'A6': 'LtBasalACA',
    'A7': 'RtPialMCA',
    'A8': 'LtPialMCA',
    'A9': 'RtPialACA',
    'A10': 'LtPialACA',
    'P0': 'BA', # Basilar artery
    'P1': 'RtVA',
    'P2': 'LtVA',
    'P3': 'RtBasalPCA',
    'P4': 'LtBasalPCA',
    'P5': 'RtPialPCA',
    'P6': 'LtPialPCA',
    'P7': 'RtCbll',
    'P8': 'LtCbll',
    'A0.01': 'ACoA',
    'A1.01': 'Rt_ICA',
    'A1.02': 'Rt_Ophthalmic',
    'A2.01': 'Lt_ICA',
    'A2.02': 'Lt_Ophthalmic',
    'A3.01': 'Rt_M1',
    'A3.02': 'Rt_MCA_Superior',
    'A3.03': 'Rt_MCA_Inferior',
    'A4.01': 'Lt_M1',
    'A4.02': 'Lt_MCA_Superior',
    'A4.03': 'Lt_MCA_Inferior',
    'A5.01': 'Rt_A1,A2',
    'A6.01': 'Lt_A1,A2',
    'A7.01': 'Rt_MCA_lat_OFA',
    'A7.02': 'Rt_MCA_PreRolandic',
    'A7.03': 'Rt_MCA_Rolandic',
    'A7.04': 'Rt_MCA_AntPerietal',
    'A7.05': 'Rt_MCA_PostParietal',
    'A7.06': 'Rt_MCA_Angular',
    'A7.07': 'Rt_MCA_PostTemporal',
    'A7.08': 'Rt_MCA_MidTemporal',
    'A7.09': 'Rt_MCA_AntTemporal',
    'A8.01': 'Lt_MCA_lat_OFA',
    'A8.02': 'Lt_MCA_PreRolandic',
    'A8.03': 'Lt_MCA_Rolandic',
    'A8.04': 'Lt_MCA_AntPerietal',
    'A8.05': 'Lt_MCA_PostParietal',
    'A8.06': 'Lt_MCA_Angular',
    'A8.07': 'Lt_MCA_PostTemporal',
    'A8.08': 'Lt_MCA_MidTemporal',
    'A8.09': 'Lt_MCA_AntTemporal',
    'A9.01': 'Rt_ACA_med_OFA',
    'A9.02': 'Rt_A2_Frontopolar',
    'A9.03': 'Rt_ACA_Callosamarginal',
    'A9.04': 'Rt_ACA_Pericallosal',
    'A10.01': 'Lt_ACA_med_OFA',
    'A10.02': 'Lt_A2_Frontopolar',
    'A10.03': 'Lt_ACA_Callosamarginal',
    'A10.04': 'Lt_ACA_Pericallosal',
    'P0.01': 'BA',
    'P1.01': 'Rt_VA',
    'P2.01': 'Lt_VA',
    'P3.01': 'Rt_P1,P2,P3',
    'P4.01': 'Lt_P1,P2,P3',
    'P5.01': 'Rt_PCoA',
    'P5.02': 'Rt_Hippocampal_artery',
    'P5.03': 'Rt_PCA_AnteriorTemporal',
    'P5.04': 'Rt_PCA_PosteriorTemporal',
    'P5.05': 'Rt_Parieto_occipital',
    'P5.06': 'Rt_Calcarine',
    'P6.01': 'Lt_PCoA',
    'P6.02': 'Lt_Hippocampal_artery',
    'P6.03': 'Lt_PCA_AnteriorTemporal',
    'P6.04': 'Lt_PCA_PosteriorTemporal',
    'P6.05': 'Lt_Parieto_occipital',
    'P6.06': 'Lt_Calcarine',
    'P7.01': 'Rt_PICA',
    'P7.02': 'Rt_AICA',
    'P7.03': 'Rt_SCA',
    'P8.01': 'Lt_PICA',
    'P8.02': 'Lt_AICA',
    'P8.03': 'Lt_SCA'
}


def reconstruct_cerebrovascular_acronym(feature_vector, switch):
    if switch == 'A':
        feature_vector.iloc[:, -1] = feature_vector.iloc[:, -1].replace(
            [
                'Rt_ICA',
                'Rt_Ophthalmic',
                'Rt_AChA',
                'Lt_ICA',
                'Lt_Ophthalmic',
                'Lt_AChA',
                'Rt_M1',
                'Rt_MCA_Superior',
                'Rt_MCA_Inferior',
                'Lt_M1', 'Lt_MCA_Superior', 'Lt_MCA_Inferior',
                'Rt_MCA_lat_OFA', 'Rt_MCA_PreRolandic', 'Rt_MCA_Rolandic',
                'Rt_MCA_AntPerietal', 'Rt_MCA_PostParietal', 'Rt_MCA_Angular',
                'Rt_MCA_PostTemporal', 'Rt_MCA_MidTemporal',
                'Rt_MCA_AntTemporal',
                'Rt_MCA_Prefrontal',
                'Lt_MCA_lat_OFA', 'Lt_MCA_PreRolandic', 'Lt_MCA_Rolandic',
                'Lt_MCA_AntPerietal', 'Lt_MCA_PostParietal',
                'Lt_MCA_Angular',
                'Lt_MCA_PostTemporal', 'Lt_MCA_MidTemporal',
                'Lt_MCA_AntTemporal',
                'Lt_MCA_Prefrontal',
                'Rt_A1', 'Rt_A2', 'Rt_A1+A2',
                'Lt_A1', 'Lt_A2', 'Lt_A1+A2',
                'Rt_ACA_med_OFA', 'Rt_A2_Frontopolar',
                'Rt_ACA_Callosamarginal',
                'Rt_ACA_Pericallosal', 'Lt_ACA_med_OFA', 'Lt_A2_Frontopolar',
                'Lt_ACA_Callosamarginal', 'Lt_ACA_Pericallosal', 'Rt_VA',
                'Lt_VA', 'Rt_P1', 'Rt_P2',
                'Rt_P1+P2', 'Rt_P3,P4', 'Lt_P1',
                'Lt_P2', 'Lt_P1+P2', 'Lt_P3,P4',
                'Rt_PPA', 'Rt_Hippocampal artery', 'Rt_PCA_AnteriorTemporal',
                'Rt_PCA_PosteriorTemporal', 'Rt_PCA_lat_Pca', 'Rt_PCoA',
                'Lt_PPA',
                'Lt_Hippocampal artery', 'Lt_PCA_AnteriorTemporal',
                'Lt_PCA_PosteriorTemporal',
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
                'Rt_MCA_PostTemporal', 'Rt_MCA_MidTemporal',
                'Rt_MCA_AntTemporal',
                'Rt_MCA_Prefrontal',
                'Lt_MCA_lat_OFA', 'Lt_MCA_PreRolandic', 'Lt_MCA_Rolandic',
                'Lt_MCA_AntPerietal', 'Lt_MCA_PostParietal', 'Lt_MCA_Angular',
                'Lt_MCA_PostTemporal', 'Lt_MCA_MidTemporal',
                'Lt_MCA_AntTemporal',
                'Lt_MCA_Prefrontal',
                'Rt_A1', 'Rt_A2', 'Rt_A1+A2',
                'Lt_A1', 'Lt_A2', 'Lt_A1+A2',
                'Rt_ACA_med_OFA', 'Rt_A2_Frontopolar',
                'Rt_ACA_Callosamarginal',
                'Rt_ACA_Pericallosal', 'Lt_ACA_med_OFA', 'Lt_A2_Frontopolar',
                'Lt_ACA_Callosamarginal', 'Lt_ACA_Pericallosal', 'Rt_VA',
                'Lt_VA', 'Rt_P1', 'Rt_P2',
                'Rt_P1+P2', 'Rt_P3,P4', 'Lt_P1',
                'Lt_P2', 'Lt_P1+P2', 'Lt_P3,P4',
                'Rt_PPA', 'Rt_Hippocampal artery', 'Rt_PCA_AnteriorTemporal',
                'Rt_PCA_PosteriorTemporal', 'Rt_PCA_lat_Pca', 'Rt_PCoA',
                'Lt_PPA',
                'Lt_Hippocampal artery', 'Lt_PCA_AnteriorTemporal',
                'Lt_PCA_PosteriorTemporal',
                'Lt_PCA_lat_Pca', 'Lt_PCoA', 'Rt_PICA', 'Rt_AICA',
                'Rt_IAA', 'Rt_SCA', 'Lt_PICA',
                'Lt_AICA', 'Lt_IAA', 'Lt_SCA',
                'BA', 'ACoA'
            ]
        )
    elif switch == 'B':
        feature_vector.iloc[:, -1] = feature_vector.iloc[:, -1].replace(
            [
                'Rt_ICA', 'Rt_Ophthalmic', 'Rt_AChA',
                'Lt_ICA', 'Lt_Ophthalmic', 'Lt_AChA',
                'Rt_M1', 'Rt_MCA_Superior', 'Rt_MCA_Inferior',
                'Lt_M1', 'Lt_MCA_Superior', 'Lt_MCA_Inferior',
                'Rt_MCA_lat_OFA', 'Rt_MCA_PreRolandic', 'Rt_MCA_Rolandic',
                'Rt_MCA_AntPerietal', 'Rt_MCA_PostParietal', 'Rt_MCA_Angular',
                'Rt_MCA_PostTemporal', 'Rt_MCA_MidTemporal',
                'Rt_MCA_AntTemporal',
                'Rt_MCA_Prefrontal',
                'Lt_MCA_lat_OFA', 'Lt_MCA_PreRolandic', 'Lt_MCA_Rolandic',
                'Lt_MCA_AntPerietal', 'Lt_MCA_PostParietal', 'Lt_MCA_Angular',
                'Lt_MCA_PostTemporal', 'Lt_MCA_MidTemporal',
                'Lt_MCA_AntTemporal',
                'Lt_MCA_Prefrontal',
                'Rt_A1', 'Rt_A2', 'Rt_A1+A2',
                'Lt_A1', 'Lt_A2', 'Lt_A1+A2',
                'Rt_ACA_med_OFA', 'Rt_A2_Frontopolar',
                'Rt_ACA_Callosamarginal',
                'Rt_ACA_Pericallosal', 'Lt_ACA_med_OFA', 'Lt_A2_Frontopolar',
                'Lt_ACA_Callosamarginal', 'Lt_ACA_Pericallosal', 'Rt_VA',
                'Lt_VA', 'Rt_P1', 'Rt_P2',
                'Rt_P1+P2', 'Rt_P3,P4', 'Lt_P1',
                'Lt_P2', 'Lt_P1+P2', 'Lt_P3,P4',
                'Rt_PPA', 'Rt_Hippocampal artery', 'Rt_PCA_AnteriorTemporal',
                'Rt_PCA_PosteriorTemporal', 'Rt_PCA_lat_Pca', 'Rt_PCoA',
                'Lt_PPA',
                'Lt_Hippocampal artery', 'Lt_PCA_AnteriorTemporal',
                'Lt_PCA_PosteriorTemporal',
                'Lt_PCA_lat_Pca', 'Lt_PCoA', 'Rt_PICA', 'Rt_AICA',
                'Rt_IAA', 'Rt_SCA', 'Lt_PICA',
                'Lt_AICA', 'Lt_IAA', 'Lt_SCA',
                'BA', 'ACoA'
            ], [
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
            ]
        )

    return feature_vector


def reconstruct_cerebrovascular_unit_old(feature_vector, switch):
    if switch == '20A':
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
        df_all_modeling_group1_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group1_r, 'A'
        )

        df_all_modeling_group2 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ICA') |
            (df_all_modeling.VesselName == 'Lt_Ophthalmic')
        ]
        df_all_modeling_group2_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ICA') |
            (df_all_modeling.VesselName == 'Lt_Ophthalmic')
        ]
        df_all_modeling_group2['VesselName'] = 'G02-Lt_ICA'
        df_all_modeling_group2_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group2_r, 'A'
        )

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
        df_all_modeling_group3_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group3_r, 'A'
        )

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
        df_all_modeling_group4_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group4_r, 'A'
        )

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
        df_all_modeling_group5_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group5_r, 'A'
        )

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
        df_all_modeling_group6_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group6_r, 'A'
        )

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
        df_all_modeling_group7_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group7_r, 'A'
        )

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
        df_all_modeling_group8_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group8_r, 'A'
        )

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
        df_all_modeling_group9_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group9_r, 'A'
        )

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
        df_all_modeling_group10_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group10_r, 'A'
        )

        df_all_modeling_group11 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11['VesselName'] = 'G11-Rt_Post-VA'
        df_all_modeling_group11_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group11_r, 'A'
        )

        df_all_modeling_group12 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12['VesselName'] = 'G12-Lt_Post-VA'
        df_all_modeling_group12_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group12_r, 'A'
        )

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
        df_all_modeling_group13_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group13_r, 'A'
        )

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
        df_all_modeling_group14_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group14_r, 'A'
        )

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
        df_all_modeling_group15_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group15_r, 'A'
        )

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
        df_all_modeling_group16_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group16_r, 'A'
        )

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
        df_all_modeling_group17_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group17_r, 'A'
        )

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
        df_all_modeling_group18_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group18_r, 'A'
        )

        df_all_modeling_group19 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19['VesselName'] = 'G19-BA'
        df_all_modeling_group19_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group19_r, 'A'
        )

        df_all_modeling_group20 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20['VesselName'] = 'G20-ACoA'
        df_all_modeling_group20_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group20_r, 'A'
        )

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

    if switch == '20B':
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
        df_all_modeling_group1_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group1_r, 'A'
        )

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
        df_all_modeling_group2_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group2_r, 'A'
        )

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
        df_all_modeling_group3_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group3_r, 'A'
        )

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
        df_all_modeling_group4_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group4_r, 'A'
        )

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
        df_all_modeling_group5_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group5_r, 'A'
        )

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
        df_all_modeling_group6_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group6_r, 'A'
        )

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
        df_all_modeling_group7_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group7_r, 'A'
        )

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
        df_all_modeling_group8_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group8_r, 'A'
        )

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
        df_all_modeling_group9_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group9_r, 'A'
        )

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
        df_all_modeling_group10_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group10_r, 'A'
        )

        df_all_modeling_group11 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11['VesselName'] = 'G11-Rt_Post-VA'
        df_all_modeling_group11_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group11_r, 'A'
        )

        df_all_modeling_group12 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12['VesselName'] = 'G12-Lt_Post-VA'
        df_all_modeling_group12_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group12_r, 'A'
        )

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
        df_all_modeling_group13_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group13_r, 'A'
        )

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
        df_all_modeling_group14_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group14_r, 'A'
        )

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
        df_all_modeling_group15_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group15_r, 'A'
        )

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
        df_all_modeling_group16_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group16_r, 'A'
        )

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
        df_all_modeling_group17_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group17_r, 'A'
        )

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
        df_all_modeling_group18_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group18_r, 'A'
        )

        df_all_modeling_group19 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19['VesselName'] = 'G19-BA'
        df_all_modeling_group19_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group19_r, 'A'
        )

        df_all_modeling_group20 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20['VesselName'] = 'G20-ACoA'
        df_all_modeling_group20_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group20_r, 'A'
        )

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


def reconstruct_cerebrovascular_unit(feature_vector, switch):
    if switch == '4A':
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

        X = df_all_modeling_group

    if switch == '9A':
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

        X = df_all_modeling_group

    if switch == '20A':
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
        df_all_modeling_group1_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group1_r, 'A'
        )

        df_all_modeling_group2 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ICA') |
            (df_all_modeling.VesselName == 'Lt_Ophthalmic')
        ]
        df_all_modeling_group2_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_ICA') |
            (df_all_modeling.VesselName == 'Lt_Ophthalmic')
        ]
        df_all_modeling_group2['VesselName'] = 'G02-Lt_ICA'
        df_all_modeling_group2_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group2_r, 'A'
        )

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
        df_all_modeling_group3_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group3_r, 'A'
        )

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
        df_all_modeling_group4_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group4_r, 'A'
        )

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
        df_all_modeling_group5_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group5_r, 'A'
        )

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
        df_all_modeling_group6_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group6_r, 'A'
        )

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
        df_all_modeling_group7_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group7_r, 'A'
        )

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
        df_all_modeling_group8_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group8_r, 'A'
        )

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
        df_all_modeling_group9_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group9_r, 'A'
        )

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
        df_all_modeling_group10_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group10_r, 'A'
        )

        df_all_modeling_group11 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11['VesselName'] = 'G11-Rt_Post-VA'
        df_all_modeling_group11_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group11_r, 'A'
        )

        df_all_modeling_group12 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12['VesselName'] = 'G12-Lt_Post-VA'
        df_all_modeling_group12_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group12_r, 'A'
        )

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
        df_all_modeling_group13_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group13_r, 'A'
        )

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
        df_all_modeling_group14_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group14_r, 'A'
        )

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
        df_all_modeling_group15_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group15_r, 'A'
        )

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
        df_all_modeling_group16_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group16_r, 'A'
        )

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
        df_all_modeling_group17_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group17_r, 'A'
        )

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
        df_all_modeling_group18_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group18_r, 'A'
        )

        df_all_modeling_group19 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19['VesselName'] = 'G19-BA'
        df_all_modeling_group19_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group19_r, 'A'
        )

        df_all_modeling_group20 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20['VesselName'] = 'G20-ACoA'
        df_all_modeling_group20_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group20_r, 'A'
        )

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

        X = df_all_modeling_group
        X_r = df_all_modeling_group_r

    if switch == '20B':
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
        df_all_modeling_group1_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group1_r, 'A'
        )

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
        df_all_modeling_group2_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group2_r, 'A'
        )

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
        df_all_modeling_group3_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group3_r, 'A'
        )

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
        df_all_modeling_group4_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group4_r, 'A'
        )

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
        df_all_modeling_group5_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group5_r, 'A'
        )

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
        df_all_modeling_group6_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group6_r, 'A'
        )

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
        df_all_modeling_group7_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group7_r, 'A'
        )

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
        df_all_modeling_group8_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group8_r, 'A'
        )

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
        df_all_modeling_group9_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group9_r, 'A'
        )

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
        df_all_modeling_group10_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group10_r, 'A'
        )

        df_all_modeling_group11 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Rt_VA')
        ]
        df_all_modeling_group11['VesselName'] = 'G11-Rt_Post-VA'
        df_all_modeling_group11_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group11_r, 'A'
        )

        df_all_modeling_group12 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'Lt_VA')
        ]
        df_all_modeling_group12['VesselName'] = 'G12-Lt_Post-VA'
        df_all_modeling_group12_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group12_r, 'A'
        )

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
        df_all_modeling_group13_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group13_r, 'A'
        )

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
        df_all_modeling_group14_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group14_r, 'A'
        )

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
        df_all_modeling_group15_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group15_r, 'A'
        )

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
        df_all_modeling_group16_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group16_r, 'A'
        )

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
        df_all_modeling_group17_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group17_r, 'A'
        )

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
        df_all_modeling_group18_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group18_r, 'A'
        )

        df_all_modeling_group19 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'BA')
        ]
        df_all_modeling_group19['VesselName'] = 'G19-BA'
        df_all_modeling_group19_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group19_r, 'A'
        )

        df_all_modeling_group20 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20_r = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'ACoA')
        ]
        df_all_modeling_group20['VesselName'] = 'G20-ACoA'
        df_all_modeling_group20_r = reconstruct_cerebrovascular_acronym(
            df_all_modeling_group20_r, 'A'
        )

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

        X = df_all_modeling_group
        X_r = df_all_modeling_group_r

    if switch == '20C':
        df_all_modeling = feature_vector
        df_all_modeling.replace(const_replacement, inplace=True)

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

        X = df_all_modeling_group
        X_r = df_all_modeling_group_r

    if switch == '67':
        df_all_modeling = feature_vector

    return (
        X, X_r,
        df_all_modeling_group1,
        df_all_modeling_group1_r,
        df_all_modeling_group2,
        df_all_modeling_group2_r,
        df_all_modeling_group3,
        df_all_modeling_group3_r,
        df_all_modeling_group4,
        df_all_modeling_group4_r,
        df_all_modeling_group5,
        df_all_modeling_group5_r,
        df_all_modeling_group6,
        df_all_modeling_group6_r,
        df_all_modeling_group7,
        df_all_modeling_group7_r,
        df_all_modeling_group8,
        df_all_modeling_group8_r,
        df_all_modeling_group9,
        df_all_modeling_group9_r,
        df_all_modeling_group10,
        df_all_modeling_group10_r,
        df_all_modeling_group11,
        df_all_modeling_group11_r,
        df_all_modeling_group12,
        df_all_modeling_group12_r,
        df_all_modeling_group13,
        df_all_modeling_group13_r,
        df_all_modeling_group14,
        df_all_modeling_group14_r,
        df_all_modeling_group15,
        df_all_modeling_group15_r,
        df_all_modeling_group16,
        df_all_modeling_group16_r,
        df_all_modeling_group17,
        df_all_modeling_group17_r,
        df_all_modeling_group18,
        df_all_modeling_group18_r,
        df_all_modeling_group19,
        df_all_modeling_group19_r,
        df_all_modeling_group20,
        df_all_modeling_group20_r
    )


def split_cerebrovascular_chunk(feature_vector, switch):
    if switch == 20:
        df_all_modeling = feature_vector

        df_all_modeling_group1 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G01-Rt_ICA')
        ]
        df_all_modeling_group2 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G02-Lt_ICA')
        ]
        df_all_modeling_group3 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G03-Rt_Ant-MCA-Basal')
        ]
        df_all_modeling_group4 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G04-Lt_Ant-MCA-Basal')
        ]
        df_all_modeling_group5 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G05-Rt_Ant-MCA-Pial')
        ]
        df_all_modeling_group6 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G06-Lt_Ant-MCA-Pial')
        ]
        df_all_modeling_group7 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G07-Rt_Ant-ACA-Basal')
        ]
        df_all_modeling_group8 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G08-Lt_Ant-ACA-Basal')
        ]
        df_all_modeling_group9 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G09-Rt_Ant-ACA-Pial')
        ]
        df_all_modeling_group10 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G10-Lt_Ant-ACA-Pial')
        ]
        df_all_modeling_group11 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G11-Rt_Post-VA')
        ]
        df_all_modeling_group12 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G12-Lt_Post-VA')
        ]
        df_all_modeling_group13 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G13-Rt_Post-PCA-Basal')
        ]
        df_all_modeling_group14 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G14-Lt_Post-PCA-Basal')
        ]
        df_all_modeling_group15 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G15-Rt_Post-PCA-Pial')
        ]
        df_all_modeling_group16 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G16-Lt_Post-PCA-Pial')
        ]
        df_all_modeling_group17 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G17-Rt_SCA,AICA,PICA')
        ]
        df_all_modeling_group18 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G18-Lt_SCA,AICA,PICA')
        ]
        df_all_modeling_group19 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G19-BA')
        ]
        df_all_modeling_group20 = df_all_modeling.loc[
            (df_all_modeling.VesselName == 'G20-ACoA')
        ]

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
