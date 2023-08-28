import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
from roi_isrsa_funcs import icbrain_beh_corr_permutation
from isrsa_funcs import behaviour_distance_calculator
from nltools.mask import expand_mask
from nltools.data import Brain_Data
from nltools.mask import roi_to_brain
from nilearn.input_data import NiftiMasker
from nilearn.image import load_img
import nibabel as nib
import pickle

brain_dir = '/data/yma/500days/brain'
result_dir = '/data/yma/500days/result/searchlight/ica/wellbeing7d_ic_perm_50cs'
sub_list = ["sub01","sub02","sub03","sub04","sub05","sub06","sub07","sub08","sub09","sub10","sub11","sub12","sub13","sub14","sub15","sub16","sub17","sub18","sub19","sub20"]
control = True
tr_num = 5470

behaviour_file_wellbeing = '/home/yma/data/Behaviour/500days/500days_wellbeing_7items.csv'
behaviour_file_age = '/home/yma/data/Behaviour/500days/500days_age.csv'
behaviour_file_gender = '/home/yma/data/Behaviour/500days/500days_gender.csv'

beh_df1, wellbeing_distance = behaviour_distance_calculator(behaviour_file_wellbeing, 'correlation')
beh_df2, age_distance = behaviour_distance_calculator(behaviour_file_age, 'euclidean')
beh_df3, gender_distance = behaviour_distance_calculator(behaviour_file_gender, 'euclidean')
stacked_arr = np.vstack((wellbeing_distance, age_distance,gender_distance)).T
assert all(beh_df1.index == beh_df2.index)
assert all(beh_df1.index == beh_df3.index)
beh_all_distances = pd.DataFrame(stacked_arr, columns=['wellbeing','age','gender'])

f = open(os.path.join('/data/yma/500days/roi_ts', "wellbeing7d_93ic_normz1_cs50_rois_ts_dict.pickle"), "rb")
#wellbeing7d_91ic_normz2_cs50_rois_ts_dict.pickle
#wellbeing7d_90ic_normz3_cs50_rois_ts_dict.pickle
#wellbeing7d_90ic_normz4_cs50_rois_ts_dict.pickle
#wellbeing7d_90ic_normz5_cs50_rois_ts_dict.pickle
#wellbeing7d_83ic_normz6_cs50_rois_ts_dict.pickle
#wellbeing7d_48ic_normz7_cs50_rois_ts_dict.pickle
#wellbeing7d_7ic_normz8_cs50_rois_ts_dict.pickle
#norm9: none surviving cluster size > 50
all_roi_dict = pickle.load(f)
wellbeing_isrsa = pd.read_csv('/data/yma/500days/result/searchlight/ica/wellbeing7d_ic_ts_control_5470secs.csv',header=None).T


#permutation
n_permute = 10000
tail = 1
n_jobs = 12
return_perms = False


all_roi_rs, all_roi_ps, _ = icbrain_beh_corr_permutation(wellbeing_isrsa, all_roi_dict, sub_list, beh_all_distances, beh_df1, control, n_permute,tail,n_jobs,return_perms)
all_roi_rs = pd.Series(all_roi_rs).sort_index()
all_roi_ps = pd.Series(all_roi_ps).sort_index()
all_roi_ps[all_roi_rs < 0] = 1 - all_roi_ps[all_roi_rs < 0]
all_roi_rs.to_csv(os.path.join(result_dir, f'500days_searchlight_93ics_normz1_cs50_rois_isrsa_permutation{str(n_permute)}_rs_wellbeing7dcontrol_1tail_mantel.csv'))
all_roi_ps.to_csv(os.path.join(result_dir, f'500days_searchlight_93ics_normz1_cs50_rois_isrsa_permutation{str(n_permute)}_ps_wellbeing7dcontrol_1tail_mantel.csv'))






