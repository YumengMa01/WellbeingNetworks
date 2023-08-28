#!/bin/bash
for ic in {1..100}
do
max=$(fslstats /Volumes/YM2/ID/Results/Nifti/500days/searchlight/wellbeing7d/500days_searchlight_isrsa_wellbeing_control_4D_5470secs_dim100_ica/z_maps/ic_${ic}.nii.gz -R | awk '{print $2}')
fslmaths /Volumes/YM2/ID/Results/Nifti/500days/searchlight/wellbeing7d/500days_searchlight_isrsa_wellbeing_control_4D_5470secs_dim100_ica/z_maps/ic_${ic}.nii.gz -div ${max} ic_${ic}_normalisedz.nii.gz
done
