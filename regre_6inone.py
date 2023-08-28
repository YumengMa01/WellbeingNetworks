import pandas as pd 
import numpy as np
from scipy.io import loadmat
import pandas as pd
import numpy as np


amplitude_isik = loadmat('/home/yma/data/Behaviour/500days/annotations/isik/amplitude.mat')['amplitude'][:,0]
social_nonsocial_isik = loadmat('/home/yma/data/Behaviour/500days/annotations/isik/social_nonsocial.mat')['social_nonsocial'][:,0]
pixel_isik = loadmat('/home/yma/data/Behaviour/500days/annotations/isik/pixel.mat')['pixel'][:,0]
valence_isik = loadmat('/home/yma/data/Behaviour/500days/annotations/isik/valence.mat')['valence'][:,0]
arousal_isik = loadmat('/home/yma/data/Behaviour/500days/annotations/isik/arousal.mat')['arousal'][:,0]
speaking_isik = loadmat('/home/yma/data/Behaviour/500days/annotations/isik/speaking.mat')['speaking'][:,0]

upsample_social=np.empty((5166,))
for i,s in enumerate(social_nonsocial_isik):
    upsample_social[(3*i):(3*i+3)]=s
upsample_valence=np.empty((5166,))
for i,s in enumerate(valence_isik):
    upsample_valence[(3*i):(3*i+3)]=s
upsample_arousal=np.empty((5166,))
for i,s in enumerate(arousal_isik):
    upsample_arousal[(3*i):(3*i+3)]=s
upsample_pixel=np.empty((5166,))
for i,s in enumerate(pixel_isik):
    upsample_pixel[(3*i):(3*i+3)]=s
upsample_amplitude=np.empty((5166,))
for i,s in enumerate(amplitude_isik):
    upsample_amplitude[(3*i):(3*i+3)]=s
upsample_speaking=np.empty((5166,))
for i,s in enumerate(speaking_isik):
    upsample_speaking[(3*i):(3*i+3)]=s


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
z_valence = scaler.fit_transform(upsample_valence.reshape(-1, 1))
z_arousal = scaler.fit_transform(upsample_arousal.reshape(-1, 1))
z_social = scaler.fit_transform(upsample_social.reshape(-1, 1))
z_speaking = scaler.fit_transform(upsample_speaking.reshape(-1, 1))
z_pixel = scaler.fit_transform(upsample_pixel.reshape(-1, 1))
z_amplitude = scaler.fit_transform(upsample_amplitude.reshape(-1, 1))

import pandas as pd
wellbeing_isrsa = pd.read_csv('/data/yma/500days/result/searchlight/ica/wellbeing7d_ic_ts_control_5470secs.csv',header=None).T
wellbeing_isrsa =wellbeing_isrsa.iloc[:,43:5205]

#valence--------------
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import fdrcorrection
def one_permute_regre(wellbeing_isrsa,roi,z_valence,z_arousal,z_social,z_speaking, z_pixel,z_amplitude):
    isrsa = np.array(wellbeing_isrsa.iloc[roi,:])
    raw_new_valence=np.random.permutation(valence_isik)
    upsample_new_valence=np.empty((5166,))
    for i,s in enumerate(raw_new_valence):
        upsample_new_valence[(3*i):(3*i+3)]=s
    z_new_valence=scaler.fit_transform(upsample_new_valence.reshape(-1, 1))
    X = pd.concat([pd.DataFrame(z_new_valence[0:5162], columns=["valence"]),
                   pd.DataFrame(z_arousal[0:5162], columns=["arousal"]),
                   pd.DataFrame(z_social[0:5162], columns=["soci"]),
                   pd.DataFrame(z_speaking[0:5162], columns=["speaking"]),
                   pd.DataFrame(z_pixel[0:5162], columns=["pixel"]),
                   pd.DataFrame(z_amplitude[0:5162], columns=["amplitude"])], axis=1)
    y = isrsa
    reg = LinearRegression().fit(X, y)
    return reg.coef_[0]

import warnings
warnings.filterwarnings('ignore')
import pingouin
from tqdm import tqdm
from joblib import Parallel, delayed
p_list = []
beta_list = []
perm_distribution_dict = {}
for roi in tqdm(range(100)):
    r_perm=Parallel(n_jobs=10)(delayed(one_permute_regre)(wellbeing_isrsa,roi,z_valence,z_arousal,z_social,z_speaking, z_pixel,z_amplitude) for per in range(10000))

    isrsa = np.array(wellbeing_isrsa.iloc[roi,:])
    X = pd.concat([pd.DataFrame(z_valence[0:5162], columns=["valence"]),
                   pd.DataFrame(z_arousal[0:5162], columns=["arousal"]),
                   pd.DataFrame(z_social[0:5162], columns=["soci"]),
                   pd.DataFrame(z_speaking[0:5162], columns=["speaking"]),
                   pd.DataFrame(z_pixel[0:5162], columns=["pixel"]),
                   pd.DataFrame(z_amplitude[0:5162], columns=["amplitude"])], axis=1)
    y = isrsa
    reg = LinearRegression().fit(X, y)
    stat = reg.coef_[0]
    beta_list.append(stat)
    denom = float(len(r_perm)) + 1
    numer = np.sum(r_perm >= stat) + 1 if stat >= 0 else np.sum(r_perm <= stat) + 1
    p_list.append(numer / denom)
    perm_distribution_dict[roi] = r_perm

pd.DataFrame(fdrcorrection(p_list, alpha=0.01)[1]).to_csv('/data/yma/500days/result/searchlight/ica/wellbeing7d_feature/regre_6inone/500days_searchlight100ica_wellbeing_isrsa_control_zvalence_regre_6inonemodel_shift4s_permby3secs_fdrps.csv')
pd.DataFrame(p_list).to_csv('/data/yma/500days/result/searchlight/ica/wellbeing7d_feature/regre_6inone/500days_searchlight100ica_wellbeing_isrsa_control_zvalence_regre_6inonemodel_shift4s_permby3secs_uncorrps.csv')
pd.DataFrame(beta_list).to_csv('/data/yma/500days/result/searchlight/ica/wellbeing7d_feature/regre_6inone/500days_searchlight100ica_wellbeing_isrsa_control_zvalence_regre_6inonemodel_shift4s_permby3secs_betas.csv')


#arousal----------------
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import fdrcorrection
def one_permute_regre(wellbeing_isrsa,roi,z_valence,z_arousal,z_social,z_speaking, z_pixel,z_amplitude):
    isrsa = np.array(wellbeing_isrsa.iloc[roi,:])
    raw_new_arousal=np.random.permutation(arousal_isik)
    upsample_new_arousal=np.empty((5166,))
    for i,s in enumerate(raw_new_arousal):
        upsample_new_arousal[(3*i):(3*i+3)]=s
    z_new_arousal=scaler.fit_transform(upsample_new_arousal.reshape(-1, 1))
    X = pd.concat([pd.DataFrame(z_valence[0:5162], columns=["valence"]),
                   pd.DataFrame(z_new_arousal[0:5162], columns=["arousal"]),
                   pd.DataFrame(z_social[0:5162], columns=["soci"]),
                   pd.DataFrame(z_speaking[0:5162], columns=["speaking"]),
                   pd.DataFrame(z_pixel[0:5162], columns=["pixel"]),
                   pd.DataFrame(z_amplitude[0:5162], columns=["amplitude"])], axis=1)
    y = isrsa
    reg = LinearRegression().fit(X, y)
    return reg.coef_[1]

import warnings
warnings.filterwarnings('ignore')
import pingouin
from tqdm import tqdm
from joblib import Parallel, delayed
p_list = []
beta_list = []
perm_distribution_dict = {}
for roi in tqdm(range(100)):
    r_perm=Parallel(n_jobs=10)(delayed(one_permute_regre)(wellbeing_isrsa,roi,z_valence,z_arousal,z_social,z_speaking, z_pixel,z_amplitude) for per in range(10000))

    isrsa = np.array(wellbeing_isrsa.iloc[roi,:])
    X = pd.concat([pd.DataFrame(z_valence[0:5162], columns=["valence"]),
                   pd.DataFrame(z_arousal[0:5162], columns=["arousal"]),
                   pd.DataFrame(z_social[0:5162], columns=["soci"]),
                   pd.DataFrame(z_speaking[0:5162], columns=["speaking"]),
                   pd.DataFrame(z_pixel[0:5162], columns=["pixel"]),
                   pd.DataFrame(z_amplitude[0:5162], columns=["amplitude"])], axis=1)
    y = isrsa
    reg = LinearRegression().fit(X, y)
    stat = reg.coef_[1]
    beta_list.append(stat)
    denom = float(len(r_perm)) + 1
    numer = np.sum(r_perm >= stat) + 1 if stat >= 0 else np.sum(r_perm <= stat) + 1
    p_list.append(numer / denom)
    perm_distribution_dict[roi] = r_perm

pd.DataFrame(fdrcorrection(p_list, alpha=0.01)[1]).to_csv('/data/yma/500days/result/searchlight/ica/wellbeing7d_feature/separate_regre_6inone/500days_searchlight100ica_wellbeing_isrsa_control_zarousal_regre_6inonemodel_shift4s_permby3secs_fdrps.csv')
pd.DataFrame(p_list).to_csv('/data/yma/500days/result/searchlight/ica/wellbeing7d_feature/regre_6inone/500days_searchlight100ica_wellbeing_isrsa_control_zarousal_regre_6inonemodel_shift4s_permby3secs_uncorrps.csv')
pd.DataFrame(beta_list).to_csv('/data/yma/500days/result/searchlight/ica/wellbeing7d_feature/regre_6inone/500days_searchlight100ica_wellbeing_isrsa_control_zarousal_regre_6inonemodel_shift4s_permby3secs_betas.csv')


#social----------
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import fdrcorrection
def one_permute_regre(wellbeing_isrsa,roi,z_valence,z_arousal,z_social,z_speaking, z_pixel,z_amplitude):
    isrsa = np.array(wellbeing_isrsa.iloc[roi,:])
    raw_new_social=np.random.permutation(social_nonsocial_isik)
    upsample_new_social=np.empty((5166,))
    for i,s in enumerate(raw_new_social):
        upsample_new_social[(3*i):(3*i+3)]=s
    z_new_social=scaler.fit_transform(upsample_new_social.reshape(-1, 1))
    X = pd.concat([pd.DataFrame(z_valence[0:5162], columns=["valence"]),
                   pd.DataFrame(z_arousal[0:5162], columns=["arousal"]),
                   pd.DataFrame(z_new_social[0:5162], columns=["soci"]),
                   pd.DataFrame(z_speaking[0:5162], columns=["speaking"]),
                   pd.DataFrame(z_pixel[0:5162], columns=["pixel"]),
                   pd.DataFrame(z_amplitude[0:5162], columns=["amplitude"])], axis=1)
    y = isrsa
    reg = LinearRegression().fit(X, y)
    return reg.coef_[2]

import warnings
warnings.filterwarnings('ignore')
import pingouin
from tqdm import tqdm
from joblib import Parallel, delayed
p_list = []
beta_list = []
perm_distribution_dict = {}
for roi in tqdm(range(100)):
    r_perm=Parallel(n_jobs=10)(delayed(one_permute_regre)(wellbeing_isrsa,roi,z_valence,z_arousal,z_social,z_speaking, z_pixel,z_amplitude) for per in range(10000))

    isrsa = np.array(wellbeing_isrsa.iloc[roi,:])
    X = pd.concat([pd.DataFrame(z_valence[0:5162], columns=["valence"]),
                   pd.DataFrame(z_arousal[0:5162], columns=["arousal"]),
                   pd.DataFrame(z_social[0:5162], columns=["soci"]),
                   pd.DataFrame(z_speaking[0:5162], columns=["speaking"]),
                   pd.DataFrame(z_pixel[0:5162], columns=["pixel"]),
                   pd.DataFrame(z_amplitude[0:5162], columns=["amplitude"])], axis=1)
    y = isrsa
    reg = LinearRegression().fit(X, y)
    stat = reg.coef_[2]
    beta_list.append(stat)
    denom = float(len(r_perm)) + 1
    numer = np.sum(r_perm >= stat) + 1 if stat >= 0 else np.sum(r_perm <= stat) + 1
    p_list.append(numer / denom)
    perm_distribution_dict[roi] = r_perm

pd.DataFrame(fdrcorrection(p_list, alpha=0.01)[1]).to_csv('/data/yma/500days/result/searchlight/ica/wellbeing7d_feature/regre_6inone/500days_searchlight100ica_wellbeing_isrsa_control_zsocial_regre_6inonemodel_shift4s_permby3secs_fdrps.csv')
pd.DataFrame(p_list).to_csv('/data/yma/500days/result/searchlight/ica/wellbeing7d_feature/regre_6inone/500days_searchlight100ica_wellbeing_isrsa_control_zsocial_regre_6inonemodel_shift4s_permby3secs_uncorrps.csv')
pd.DataFrame(beta_list).to_csv('/data/yma/500days/result/searchlight/ica/wellbeing7d_feature/regre_6inone/500days_searchlight100ica_wellbeing_isrsa_control_zsocial_regre_6inonemodel_shift4s_permby3secs_betas.csv')


#speaking----------
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import fdrcorrection
def one_permute_regre(wellbeing_isrsa,roi,z_valence,z_arousal,z_social,z_speaking, z_pixel,z_amplitude):
    isrsa = np.array(wellbeing_isrsa.iloc[roi,:])
    raw_new_speaking=np.random.permutation(speaking_isik)
    upsample_new_speaking=np.empty((5166,))
    for i,s in enumerate(raw_new_speaking):
        upsample_new_speaking[(3*i):(3*i+3)]=s
    z_new_speaking=scaler.fit_transform(upsample_new_speaking.reshape(-1, 1))
    X = pd.concat([pd.DataFrame(z_valence[0:5162], columns=["valence"]),
                   pd.DataFrame(z_arousal[0:5162], columns=["arousal"]),
                   pd.DataFrame(z_social[0:5162], columns=["soci"]),
                   pd.DataFrame(z_new_speaking[0:5162], columns=["speaking"]),
                   pd.DataFrame(z_pixel[0:5162], columns=["pixel"]),
                   pd.DataFrame(z_amplitude[0:5162], columns=["amplitude"])], axis=1)
    y = isrsa
    reg = LinearRegression().fit(X, y)
    return reg.coef_[3]

import warnings
warnings.filterwarnings('ignore')
import pingouin
from tqdm import tqdm
from joblib import Parallel, delayed
p_list = []
beta_list = []
perm_distribution_dict = {}
for roi in tqdm(range(100)):
    r_perm=Parallel(n_jobs=10)(delayed(one_permute_regre)(wellbeing_isrsa,roi,z_valence,z_arousal,z_social,z_speaking, z_pixel,z_amplitude) for per in range(10000))

    isrsa = np.array(wellbeing_isrsa.iloc[roi,:])
    X = pd.concat([pd.DataFrame(z_valence[0:5162], columns=["valence"]),
                   pd.DataFrame(z_arousal[0:5162], columns=["arousal"]),
                   pd.DataFrame(z_social[0:5162], columns=["soci"]),
                   pd.DataFrame(z_speaking[0:5162], columns=["speaking"]),
                   pd.DataFrame(z_pixel[0:5162], columns=["pixel"]),
                   pd.DataFrame(z_amplitude[0:5162], columns=["amplitude"])], axis=1)
    y = isrsa
    reg = LinearRegression().fit(X, y)
    stat = reg.coef_[3]
    beta_list.append(stat)
    denom = float(len(r_perm)) + 1
    numer = np.sum(r_perm >= stat) + 1 if stat >= 0 else np.sum(r_perm <= stat) + 1
    p_list.append(numer / denom)
    perm_distribution_dict[roi] = r_perm

pd.DataFrame(fdrcorrection(p_list, alpha=0.01)[1]).to_csv('/data/yma/500days/result/searchlight/ica/wellbeing7d_feature/regre_6inone/500days_searchlight100ica_wellbeing_isrsa_control_zspeaking_regre_6inonemodel_shift4s_permby3secs_fdrps.csv')
pd.DataFrame(p_list).to_csv('/data/yma/500days/result/searchlight/ica/wellbeing7d_feature/regre_6inone/500days_searchlight100ica_wellbeing_isrsa_control_zspeaking_regre_6inonemodel_shift4s_permby3secs_uncorrps.csv')
pd.DataFrame(beta_list).to_csv('/data/yma/500days/result/searchlight/ica/wellbeing7d_feature/regre_6inone/500days_searchlight100ica_wellbeing_isrsa_control_zspeaking_regre_6inonemodel_shift4s_permby3secs_betas.csv')

