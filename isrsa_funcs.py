import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os
import glob
from nilearn.maskers.nifti_spheres_masker import _apply_mask_and_get_affinity
from nilearn.masking import _load_mask_img
from nilearn.image.resampling import coord_transform
from nilearn.image import load_img
from scipy import sparse
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy import stats
import pingouin
import seaborn as sns
import time
import multiprocessing
from multiprocessing import Process, Manager
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from tqdm import tqdm


def behaviour_distance_calculator(behaviour_file, metric):
    ## Load behavioural score file 
    behaviour_dat = pd.read_csv(behaviour_file, header = 0, index_col = 0)

    ## Pairwise correlation distance
    behaviour_distance = pdist(behaviour_dat, metric=metric)#vectorised
    
    return behaviour_dat, behaviour_distance 



def apply_mask(mask_file):

    ## Get coordinates in the mask
    # process_mask:3d boolean(coordinates inside the mask are True)；
    # process_mask_affine: affine matrix transforming coordinates in process_mask to real brain coordinates 
    process_mask, process_mask_affine = _load_mask_img(mask_file)
    process_mask_coords = np.where(process_mask != 0)
    process_mask_coords = coord_transform(process_mask_coords[0], process_mask_coords[1], process_mask_coords[2], process_mask_affine)
    process_mask_coords = np.asarray(process_mask_coords).T #this is real brain coordinates inside the mask(shape: voxels*coordinates)
    
    return process_mask, process_mask_affine, process_mask_coords


def searchlight_id_calculator(process_mask_coords, loaded_niimg, radius, mask_file, Ax_start_id, Ax_end_id):
    '''
    A : scipy.sparse.lil_matrix
        Contains the boolean indices for each sphere.
        shape: (number of seeds, number of voxels)
    Ax: contains seeds in one round
    '''
    A = _apply_mask_and_get_affinity(process_mask_coords, loaded_niimg, radius, True, mask_img= mask_file)[1]
    Ax = A[Ax_start_id:Ax_end_id, :]
   
    return Ax

def extract_ts(all_sub_ts, brain_dir, process_mask_coords, radius, mask_file, one_proc_sub_list):
    '''
    all_sub_ts : dictionary {"sub75":2D_array, "sub76": 2D_array, ...}
    for each sub, time-series are saved in a 2D numpy.ndarray; shape: (number of scans, number of voxels)
    '''
    for sub in one_proc_sub_list:
        f = glob.glob(os.path.join(brain_dir,sub,'*.nii.gz'))[0]
        sub_niimg = load_img(f) 
        all_sub_ts[sub] = _apply_mask_and_get_affinity(process_mask_coords, sub_niimg, radius, True, mask_img= mask_file)[0]
    
    
def extract_ts_wrapper(brain_dir, process_mask_coords, radius, mask_file, all_proc_sub_list):
    manager = multiprocessing.Manager()
    all_sub_ts = manager.dict()
    p_list = []       
    for one_proc_sub_list in all_proc_sub_list:
        p = Process(target=extract_ts, args=(all_sub_ts, brain_dir, process_mask_coords, radius, mask_file, one_proc_sub_list))
        p_list.append(p)
        p.start()

    for proc in p_list:
        proc.join()    

    all_sub_ts = all_sub_ts._getvalue()
    
    return all_sub_ts


def extract_searchlight_ts(Ax, all_sub_ts, sub_list, start_id):
    '''
    Ax_voxels_dict={"v1":{"sub1":TRs*voxels，“sub2":TR*voxels}, "v2":{...}...}
    v1, v2.. are searchlight centres
    '''
    Ax_voxels_dict = {}
    for v in enumerate(Ax):
        all_sub_dict = {}
        for sub in sub_list:
            all_sub_dict[sub] = all_sub_ts[sub][:,Ax.rows[v[0]]]
        Ax_voxels_dict["v"+str(v[0]+1+start_id)] = all_sub_dict
        
    return Ax_voxels_dict


def brain_beh_corr(Ax_v_isrsa, behaviour_df, Ax, tr_num, start_id, Ax_voxels_dict, behaviour_distance, control):
    for v in enumerate(Ax):
        one_v_isrsa = {}
        for t in range(tr_num):
        # tr_num is the number of TR
            oneTR_dict = {}
            for sub in Ax_voxels_dict["v"+str(v[0]+1+start_id)]:
                oneTR_dict[sub] =  Ax_voxels_dict["v"+str(v[0]+1+start_id)][sub][t, :]
            oneTR_df = pd.DataFrame(oneTR_dict).T
            assert all(oneTR_df.index == behaviour_df.index), "Subject IDs are not in the correct order"
            brain_distance = pdist(oneTR_df, metric='correlation')
            if control == True:
                total_distances = pd.concat([behaviour_distance, pd.DataFrame(brain_distance, columns=["brain"])], axis=1)
                try:
                    r = pingouin.partial_corr(total_distances, x='wellbeing',y='brain',covar=['age','gender'],method='spearman').r.item()
                except:
                    r = float('nan')
            if control == False: 
                r = stats.spearmanr(behaviour_distance, brain_distance)[0]
            one_v_isrsa["t"+str(t+1).zfill(4)] = r
        Ax_v_isrsa["v"+str(v[0]+1+start_id).zfill(5)] = one_v_isrsa


        
def round_assign(total_voxel_num, voxel_num_per_round):
    if total_voxel_num%voxel_num_per_round == 0:
        rounds = int(total_voxel_num/voxel_num_per_round)
        all_rounds = []
        for i in range(rounds):
            Ax_start_id = i*voxel_num_per_round
            Ax_end_id = (i+1)*voxel_num_per_round
            A_round = i+1
            all_rounds.append((Ax_start_id,Ax_end_id,A_round))
            
    else:
        rounds = total_voxel_num//voxel_num_per_round + 1
        all_rounds = []
        for i in range(rounds):
            if i <= rounds-2:
                Ax_start_id = i*voxel_num_per_round
                Ax_end_id = (i+1)*voxel_num_per_round
                A_round = i+1
                all_rounds.append((Ax_start_id,Ax_end_id,A_round))
            else:
                Ax_start_id = i*voxel_num_per_round
                Ax_end_id = total_voxel_num
                A_round = i+1
                all_rounds.append((Ax_start_id,Ax_end_id,A_round))
                
    return all_rounds
    
        
def process_assign(Ax, n_process):
    if Ax.shape[0] % n_process == 0:
        n_voxel = int(Ax.shape[0]/n_process)
        proc_Ax_list = []
        for i in range(n_process):
            proc_Ax_list.append(Ax[i*n_voxel:(i+1)*n_voxel, :])
    else:
        n_voxel = Ax.shape[0]//(n_process-1) 
        #n_process needs to be << Ax.shape[0] (n_process**2~=Ax.shape[0])
        proc_Ax_list = []
        for i in range(n_process):
            if i<= n_process-2:
                proc_Ax_list.append(Ax[i*n_voxel:(i+1)*n_voxel, :])
            else:
                proc_Ax_list.append(Ax[i*n_voxel:Ax.shape[0], :])
        
    return n_voxel, proc_Ax_list



class ISRSA_Calculator(object):
    def __init__(self, behaviour_distance, mask_file, radius, loaded_niimg, brain_dir, all_proc_sub_list, sub_list, behaviour_df, tr_num, n_process, Ax_start_id, Ax_end_id, control):
        self.behaviour_distance = behaviour_distance
        self.mask_file = mask_file
        self.radius = radius
        self.loaded_niimg = loaded_niimg
        self.brain_dir = brain_dir 
        #self.brain_file_urls = brain_file_urls
        self.all_proc_sub_list = all_proc_sub_list
        self.sub_list = sub_list
        self.behaviour_df = behaviour_df
        self.tr_num = tr_num
        self.n_process = n_process
        self.Ax_start_id = Ax_start_id
        self.Ax_end_id = Ax_end_id
        self.control = control
        
    def isrsa_wrapper(self):
        process_mask_coords = apply_mask(self.mask_file)[2]
        all_sub_ts = extract_ts_wrapper(self.brain_dir, process_mask_coords, self.radius, self.mask_file, self.all_proc_sub_list)
        Ax = searchlight_id_calculator(process_mask_coords, self.loaded_niimg, self.radius, self.mask_file, self.Ax_start_id, self.Ax_end_id)
        searchlight_ts_x = extract_searchlight_ts(Ax, all_sub_ts, self.sub_list, self.Ax_start_id)

        del process_mask_coords
        del all_sub_ts

        n_voxel, proc_Ax_list = process_assign(Ax, self.n_process)
        print("Start multiprocessing")
        #if __name__=='__main__':
            #multiprocessing.set_start_method("fork")
        manager = multiprocessing.Manager()
        isrsa_result_x = manager.dict()
        p_list = []

        for i, proc_Ax in enumerate(proc_Ax_list):
            proc_start_id = self.Ax_start_id + i*n_voxel
            p = Process(target=brain_beh_corr, args=(isrsa_result_x, self.behaviour_df, proc_Ax, self.tr_num,proc_start_id,searchlight_ts_x,self.behaviour_distance, self.control))
            p_list.append(p)
            p.start()

        for proc in p_list:
            proc.join()    
                
        result_x_pd = pd.DataFrame(isrsa_result_x._getvalue())
        sorted_result_x_pd = result_x_pd.sort_index(axis=1)
        #print(sorted_result_pd)

        return sorted_result_x_pd

    









  
