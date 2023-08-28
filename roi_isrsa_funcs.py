import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import re
import os
import glob
import time
from scipy.spatial.distance import pdist
from nilearn.input_data import NiftiMasker
import multiprocessing
from multiprocessing import Process, Manager
from scipy import stats
import pingouin
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import manhattan_distances
from joblib import Parallel, delayed
from sklearn.utils import check_random_state


def roi_process_assign(roi_num, n_round):
    if roi_num % n_round == 0:
        n_roi = int(roi_num/n_round)
        proc_roi_list = []
        for p in range(n_round):
            one_proc_roi_list = []
            for r in range(n_roi):
                one_proc_roi_list.append(str(n_roi*p+r+1))
            proc_roi_list.append(one_proc_roi_list)
    else:
        n_roi = roi_num//(n_round-1) 
        proc_roi_list = []
        for p in range(n_round):
            if p<= n_round-2:
                one_proc_roi_list = []
                for r in range(n_roi):
                    one_proc_roi_list.append(str(n_roi*p+r+1))
                proc_roi_list.append(one_proc_roi_list)
            else:
                final_proc_roi_list = []
                for r in range(roi_num-n_roi*(n_round-1)):
                    final_proc_roi_list.append(str(n_roi*p+r+1))
                proc_roi_list.append(final_proc_roi_list)
        
    return proc_roi_list

def extract_icroi_ts(all_ic_dict,proc_ics,mask_dir,file_list,brain_dir,thre):
    for ic in proc_ics:
        mask_file = os.path.join(mask_dir, f'{ic}_normz{thre}_cs50_mask.nii.gz')
        masker = NiftiMasker(mask_img=mask_file)
        one_ic_dict = {}
        for f in file_list:
            sub = os.path.basename(os.path.dirname(f))
            func_data = masker.fit_transform(f)
            one_ic_dict[sub] = func_data
        all_ic_dict[ic] = one_ic_dict
        
        
def extract_icroi_ts_wrapper(proc_ic_list,mask_dir,brain_dir,thre):
    file_list = glob.glob(os.path.join(brain_dir,'sub*','*.nii.gz'))
    manager = multiprocessing.Manager()
    all_ic_dict = manager.dict()
    p_list = []       
    for proc_ics in proc_ic_list:
        p = Process(target=extract_icroi_ts, args=(all_ic_dict, proc_ics,mask_dir,file_list,brain_dir,thre))
        p_list.append(p)
        p.start()

    for proc in p_list:
        proc.join()    

    all_ic_dict = all_ic_dict._getvalue()
        
    return all_ic_dict


def _calc_pvalue(all_p, stat, tail):
    """Calculates p value based on distribution of correlations
    This function is called by the permutation functions
        all_p: list of correlation values from permutation
        stat: actual value being tested, i.e., rp['correlation'] or rp['mean']
        tail: (int) either 2 or 1 for two-tailed p-value or one-tailed
    """

    denom = float(len(all_p)) + 1
    if tail == 1:
        numer = np.sum(all_p >= stat) + 1 if stat >= 0 else np.sum(all_p <= stat) + 1
    elif tail == 2:
        numer = np.sum(np.abs(all_p) >= np.abs(stat)) + 1
    else:
        raise ValueError("tail must be either 1 or 2")
    return numer / denom

def _permute_func(brain_distance_square, behaviour_distance, control, random_state=None):
    random_state = check_random_state(random_state)
    data_row_id = range(brain_distance_square.shape[0])
    permuted_ix1 = random_state.choice(data_row_id, size=len(data_row_id), replace=False)
    #permuted_ix2 = random_state.choice(data_row_id, size=len(data_row_id), replace=False)
    new_brain_dist = brain_distance_square.iloc[permuted_ix1, permuted_ix1].values
    new_brain_dist = new_brain_dist[np.triu_indices(new_brain_dist.shape[0], k=1)]
    if control == False:
        new_r = stats.spearmanr(new_brain_dist, behaviour_distance)[0]
    if control == True:
        total_distances = pd.concat([behaviour_distance, pd.DataFrame(new_brain_dist, columns=["brain"])], axis=1)
        new_r = np.float64(pingouin.partial_corr(total_distances, x='wellbeing',y='brain',covar=['age','gender'],method='spearman').r.item())

    return new_r

def matrix_permutation(
    brain_distance_square,
    behaviour_distance,
    n_permute,
    control, 
    tail,
    n_jobs,
    return_perms,
    random_state,
):
    """
    Returns:
        rp: (dict) dictionary of permutation results ['correlation','p']
    """
    MAX_INT = np.iinfo(np.int32).max
    random_state = check_random_state(random_state)
    seeds = random_state.randint(MAX_INT, size=n_permute)
    rp = {}
    if control == False:
        brain_distance = brain_distance_square[np.triu_indices(brain_distance_square.shape[0], k=1)]
        r = stats.spearmanr(brain_distance, behaviour_distance)[0]
        if np.isnan(r) == False:
            rp["correlation"] = r
            all_p = Parallel(n_jobs=n_jobs)(delayed(_permute_func)(pd.DataFrame(brain_distance_square), behaviour_distance, control, random_state=seeds[i]) for i in range(n_permute))
            rp["p"] = _calc_pvalue(all_p, rp["correlation"], tail)
            if return_perms:
                rp["perm_dist"] = all_p
        if np.isnan(r) == True:
            rp["correlation"] = r
            rp["p"] = float('nan')
            if return_perms:
                rp["perm_dist"] = float('nan')
    
    if control == True:
        brain_distance = brain_distance_square[np.triu_indices(brain_distance_square.shape[0], k=1)]
        total_distances = pd.concat([behaviour_distance, pd.DataFrame(brain_distance, columns=["brain"])], axis=1)
        try:
            r = np.float64(pingouin.partial_corr(total_distances, x='wellbeing',y='brain',covar=['age','gender'],method='spearman').r.item())
        except:
            r = np.float64('nan')
        if np.isnan(r) == False:
            rp["correlation"] = r
            all_p = Parallel(n_jobs=n_jobs)(delayed(_permute_func)(pd.DataFrame(brain_distance_square), behaviour_distance, control, random_state=seeds[i]) for i in range(n_permute))
            rp["p"] = _calc_pvalue(all_p, rp["correlation"], tail)
            if return_perms:
                rp["perm_dist"] = all_p
        if np.isnan(r) == True:
            rp["correlation"] = r
            rp["p"] = np.float64('nan')
            if return_perms:
                rp["perm_dist"] = float('nan')
            
    return rp


def icbrain_beh_corr_permutation(wellbeing_isrsa,all_roi_dict, sub_list, behaviour_distance, behaviour_df, control,n_permute,tail,n_jobs,return_perms):
    all_roi_rs = {}
    all_roi_ps = {}
    all_roi_perm_dist = {}
    for roi in all_roi_dict:
        t_to_permutate= np.argmax(wellbeing_isrsa.iloc[int(re.search(r'\d+', roi).group())-1,:])
        one_roi_dict = {}
        for sub in sub_list:
            one_roi_dict[sub] = all_roi_dict[roi][sub][t_to_permutate, :]
        one_roi_df = pd.DataFrame(one_roi_dict).T
        assert all(one_roi_df.index == behaviour_df.index), "Subject IDs are not in the correct order"
        brain_distance = pdist(one_roi_df, metric='correlation')
        brain_distance_square = squareform(brain_distance)
        rp = matrix_permutation(brain_distance_square,behaviour_distance,n_permute,control,tail,n_jobs,return_perms,random_state=None)
        all_roi_rs[roi+'_t'+str(t_to_permutate+1)] = rp['correlation']
        all_roi_ps[roi+'_t'+str(t_to_permutate+1)] = rp['p']
        if return_perms == True:
            all_roi_perm_dist[roi+'_t'+str(t_to_permutate+1)] = rp['perm_dist']
        if return_perms == False:
            all_roi_perm_dist[roi+'_t'+str(t_to_permutate+1)] = float('nan')
        
    return all_roi_rs, all_roi_ps, all_roi_perm_dist

