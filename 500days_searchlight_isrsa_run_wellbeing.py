import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
from nilearn.image import load_img
import nibabel as nib
import os
import glob
from isrsa_funcs import behaviour_distance_calculator
from isrsa_funcs import apply_mask
from isrsa_funcs import round_assign
from isrsa_funcs import ISRSA_Calculator

brain_dir = '/data/yma/500days/brain' #need to be in the format: brain_dir/sub01/functional_file.nii.gz
result_dir = '/data/yma/500days/result/searchlight' #result directory; need a a folder called CSVs in it
mask_file = '/home/yma/data/Brain/500days/mask/500days_union_mask.nii' #group mask

loaded_niimg = load_img('/data/yma/500days/brain/sub02/derivatives_sub-2_func_sub-2_task-500daysofsummer_bold_no_blur_no_censor_ica.nii.gz')
#func file of any subject

sub_list = ["sub01","sub02","sub03","sub04","sub05","sub06","sub07","sub08","sub09","sub10","sub11","sub12","sub13","sub14","sub15","sub16","sub17","sub18","sub19","sub20"] #should be in right sequence

all_proc_sub_list = [['sub01','sub02','sub03','sub04'],['sub05','sub06','sub07','sub08'],
['sub09','sub10','sub11','sub12'],
['sub13','sub14','sub15','sub16'],
['sub17','sub18','sub19','sub20']]
#separate subjects into groups to extract their timeseries,
#number of groups=number of jobs/cores used

radius = 6 #radius of searchlight in mm
tr_num = 5470 #number of TRs
n_process = 15 #number of jobs/cores used
total_voxel_num = 70482 #number of voxels in the group mask
voxel_num_per_round = 5000 #depend on avaiable RAM

"""
#not controlling for age and gender
control = False
behaviour_file_wellbeing = '/home/yma/data/Behaviour/500days/500days_wellbeing_7items.csv'# need to be shape(subs*traits)
behaviour_df, wellbeing_distance = behaviour_distance_calculator(behaviour_file_wellbeing, 'correlation')

all_rounds = round_assign(total_voxel_num, voxel_num_per_round)

if __name__=='__main__':
    for Ax_start_id,Ax_end_id,A_round in tqdm(all_rounds):

        calculator_object = ISRSA_Calculator(wellbeing_distance, mask_file, radius, loaded_niimg, brain_dir, all_proc_sub_list, sub_list, behaviour_df, tr_num, n_process, Ax_start_id, Ax_end_id,control)
        result = calculator_object.isrsa_wrapper()
        result.T.to_csv(os.path.join(result_dir, f'500days_searchlight_isrsa_wellbeing_A{A_round}.csv'))
"""     
#controlling for age and gender

control = True
behaviour_file_wellbeing = '/home/yma/data/Behaviour/500days/500days_wellbeing_7items.csv'# need to be shape(subs*traits)
behaviour_file_age = '/home/yma/data/Behaviour/500days/500days_age.csv'
behaviour_file_gender = '/home/yma/data/Behaviour/500days/500days_gender.csv'
beh_df1, wellbeing_distance = behaviour_distance_calculator(behaviour_file_wellbeing, 'correlation')
beh_df2, age_distance = behaviour_distance_calculator(behaviour_file_age, 'euclidean')
beh_df3, gender_distance = behaviour_distance_calculator(behaviour_file_gender, 'euclidean')
stacked_arr = np.vstack((wellbeing_distance, age_distance,gender_distance)).T
assert all(beh_df1.index == beh_df2.index)
assert all(beh_df1.index == beh_df3.index)
beh_all_distances = pd.DataFrame(stacked_arr, columns=['wellbeing','age','gender'])

all_rounds = round_assign(total_voxel_num, voxel_num_per_round)

if __name__=='__main__':
    for Ax_start_id,Ax_end_id,A_round in tqdm(all_rounds):

        calculator_object = ISRSA_Calculator(beh_all_distances, mask_file, radius, loaded_niimg, brain_dir, all_proc_sub_list, sub_list, beh_df1, tr_num, n_process, Ax_start_id, Ax_end_id, control)
        result = calculator_object.isrsa_wrapper()
        result.T.to_csv(os.path.join(result_dir, 'CSVs', f'500days_searchlight_isrsa_wellbeing7d_control_A{A_round}.csv'))

        
#save nifti       
result_files = glob.glob(os.path.join(result_dir,'CSVs', '*.csv'))
result_df_list = []
for f in result_files:
    result_df_list.append(pd.read_csv(f, header = 0, index_col = 0))
    
total_result = pd.concat(result_df_list, axis=0)
total_result = total_result.sort_index(axis=0)


from isrsa_funcs import apply_mask
mask_file = '/home/yma/data/Brain/500days/mask/500days_union_mask.nii' 
process_mask, process_mask_affine, _ = apply_mask(mask_file)
img_sequence = []
for i, t in enumerate(total_result):
    searchlight_isrsa_3D = np.zeros(process_mask.shape)
    searchlight_isrsa_3D[process_mask] = np.array(total_result[t])
    ni_img = nib.Nifti1Image(searchlight_isrsa_3D, process_mask_affine)
    img_sequence.append(ni_img)

img_4D = nib.funcs.concat_images(img_sequence, check_affines=True, axis=None)
nib.save(img_4D, os.path.join(result_dir, '500days_searchlight_isrsa_wellbegin7d_control_4D.nii.gz'))

#save csv
total_result = total_result.T
total_result[np.isnan(total_result)] = 0
total_result.to_csv(os.path.join(result_dir, '500days_searchlight_isrsa_wellbeing7d_control.csv'))
