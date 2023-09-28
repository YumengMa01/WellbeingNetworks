# WellbeingNetworks
**Ma, Y., & Skipper, J.I. (2023). Individual differences in wellbeing are supported by separable sets of co-active self- and visual-attention-related brain networks. bioRxiv.**

Preprocessed fMRI data of _500 Days of Summer_ movie watching are availbe on [https://openneuro.org/datasets/ds002837/versions/2.0.0]. \
Result images are available on [https://neurovault.org/collections/15197/].

**1. Searchlight spatial inter-participant representational similarity analysis (SSIP-RSA)**
* 500days_searchlight_isrsa_run_wellbeing.py: run SSIP-RSA  
* isrsa_funcs.py: functions required for 500days_searchlight_isrsa_run_wellbeing.py

**2. ICA**
* ica.sh: run ICA in the resulting ρ time series from SSIP-RSA

**3. Hierachical clustering**
* allother_analyses.ipynb, Hierachical clustering: cluster IC timecourses 

**4. Create masks for Mantel test**
* normalisez.sh: divide each voxel's loading by the maximum loading value for each IC map
* thre_normz_maps.sh: apply different thresholds (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, and 0.9) to create masks used in Mantel test

**5. Mantel test**
* roi_isrsa_funcs.py: functions required for roi_ts_extract.py and Mantel_roi_isrsa_run.py
* roi_ts_extract.py: extract BOLD time series from masks
* Mantel_roi_isrsa_run.py: run Mantel test for each mask
* allother_analyses.ipynb, Mantel correction: multiple comparison correction using FDR

**6. Decoding**
* mergesigics_clades.sh: merge maps of ICs that are statistically significant in Mantel test for each clade
* nimare_Decoding.py: correlate merged IC maps with term-based meta-analyses maps

**7. Regression**
* regre_6inone.py: regress IC timecourses with movie annotations
* allother_analyses.ipynb, Regression correction: multiple comparison correction using FDR








