#!/bin/bash
tr=1
dims=100

fname="/data/yma/500days/result/searchlight/500days_searchlight_isrsa_wellbegin7d_control_4D.nii.gz"
mname="/home/yma/data/Brain/MNI152_2009_template_SSW"


fsl5.0-melodic \
  --tr="$tr" --nobet --report --Oall --dim="$dims" \
  --outdir=500days_searchlight_isrsa_wellbeing_control_4D_5470secs_dim"$dims"_ica \
  --bgimage="$mname"_mask.nii.gz \
  --mask="$mname"_mask.nii.gz \
  --in="$fname".nii.gz
