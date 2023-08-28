import pandas as pd
import numpy as np
import glob
import os
from roi_isrsa_funcs import roi_process_assign
from roi_isrsa_funcs import extract_icroi_ts_wrapper
import pickle

brain_dir = '/data/yma/500days/brain'
result_dir = '/data/yma/500days/roi_ts'
sub_list = ["sub01","sub02","sub03","sub04","sub05","sub06","sub07","sub08","sub09","sub10","sub11","sub12","sub13","sub14","sub15","sub16","sub17","sub18","sub19","sub20"]


n_process = 10
roi_num = 93
thre=0.1 #0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
mask_dir = f'/home/yma/data/Brain/wellbeing7d_control_ic_normz{thre}_cs50_masks' 
proc_roi_list = roi_process_assign(roi_num, n_process)


if __name__=='__main__':
    all_roi_dict = extract_icroi_ts_wrapper(proc_ic_list,mask_dir,brain_dir,thre)

print(time.time()-time_s)
with open(os.path.join(result_dir, "wellbeing7d_93ic_normz1_cs50_rois_ts_dict_part4.pickle"), 'wb') as f:
    pickle.dump(all_roi_dict, f, pickle.HIGHEST_PROTOCOL)

