import os
import nimare
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nimare.decode import continuous

out_dir = os.path.abspath("./nimare_ns_data/")
files = nimare.extract.fetch_neurosynth(path=out_dir, version="7", overwrite=False, source="abstract", vocab="terms",)
neurosynth_db = files[0]
neurosynth_dset = nimare.io.convert_neurosynth_to_dataset(coordinates_file=neurosynth_db["coordinates"], metadata_file=neurosynth_db["metadata"], annotations_files=neurosynth_db["features"])
neurosynth_dset.save(os.path.join(out_dir, "neurosynth_dataset.pkl.gz"))


neurosynth_dset = nimare.dataset.Dataset.load(os.path.join(out_dir, "neurosynth_dataset.pkl.gz"))
neurosynth_dset.update_path(out_dir)
term_list = ["id", "study_id", "contrast_id",
'terms_abstract_tfidf__memory retrieval',
'terms_abstract_tfidf__language',
'terms_abstract_tfidf__autobiographical',
'terms_abstract_tfidf__self referential',
'terms_abstract_tfidf__interoceptive',             
'terms_abstract_tfidf__emotion',
'terms_abstract_tfidf__autonomic',
'terms_abstract_tfidf__affective',
'terms_abstract_tfidf__attention',
'terms_abstract_tfidf__visual',
'terms_abstract_tfidf__speech production',
'terms_abstract_tfidf__speech perception']

neurosynth_dset.annotations = neurosynth_dset.annotations[term_list]
decoder = continuous.CorrelationDecoder(feature_group=None, features=None)
decoder.fit(neurosynth_dset)

import pickle
with open(os.path.join('/data/yma/500days/result/searchlight/ica/wellbeing7d_7_cls_thre_ic_maps', "12items_14371papers_decoder.pickle"), 'wb') as f:
    pickle.dump(decoder, f, pickle.HIGHEST_PROTOCOL)
    
    
features = ['mem retriev','lang','autobio','self','interocep','emo','autonomic','affective','attention','visual','spch produc','spch percep']
features = [*features, features[0]]
cl=7 #1 2 3 4 5 6
cm = 1/2.54 
#fig, (ax1, ax2) = plt.subplots(1, 2)
r_pos = decoder.transform(f"/data/yma/500days/result/searchlight/ica/wellbeing7d_7_cls_thre_ic_maps/49sig_merge_clades/clade{cl}_sigics_merge_prob0.5_c50.nii.gz")['r']
r_pos = [*r_pos, r_pos[0]]
#r_neg = pd.read_csv('/data/yma/500days/result/searchlight/ica/neurosynth/newnew_12term_r_ic39.csv',index_col=0)['r']
#r_neg = [*r_neg, r_neg[0]]
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(r_pos))

plt.figure(figsize=(5*cm, 5*cm))
plt.figure(dpi=300)
plt.subplot(polar=True)

plt.plot(label_loc, r_pos,color='#D62728')
#fec615----yellow original
#BCBD22----yellow clade1
#7F7F7F----grey clade2
#9467BD----purple clade3
#2CA02C----green clade4
#8C564B----brown clade5
#e377c2----pink clade6
#D62728----red clade7

#plt.plot(label_loc, r_pos, label='More Pleasant (+\u03B2)')

#fontname="Times New Roman", size=28,fontweight="bold"
#plt.title('Cluster 1', size=10, y=1.09)

lines, labels = plt.thetagrids(np.degrees(label_loc), labels=features,fontname="Times New Roman", fontsize = 18, color = 'w')
plt.fill(label_loc,r_pos,facecolor='#FF7377',alpha=0.25)
#f9e076----yellow clade1
#d3d3d3----grey clade2
#cb99c7----purple clade3
#90EE90----green clade4
#C4A484----brown clade5
#FFB6C1----yellow clade6
#FF7377----red clade7

plt.savefig(f'/data/yma/500days/result/searchlight/ica/wellbeing7d_7_cls_thre_ic_maps/polarplots_mergesigs/clade{cl}_12terms_r_18pt_300dpi_5cm5cm_whitetext.jpg',facecolor='w',dpi=300)
#plt.legend(loc=(0.95, 0.95)