#!/bin/bash
clust=50
thresh=0.1 #0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
mkdir ./wellbeing7d_control_ic_normz"$thresh"_cs"$clust"_masks
for ic in {1..100}
do
3dmerge \
  -1tindex 0 -1dindex 0 -1thresh "$thresh" -gcount \
  -dxyz=1 -1clust 1 "$clust" \
  -prefix ic${ic}_normz"$thresh"_cs"$clust".nii.gz \
  /Volumes/YM2/ID/Results/Nifti/500days/searchlight/wellbeing7d/500days_searchlight_isrsa_wellbeing_control_4D_5470secs_dim100_ica/z_maps/nor_z_maps/ic_${ic}_normalisedz.nii.gz
done

for ic in {1..100}
do
3dcalc -a ic${ic}_normz"$thresh"_cs"$clust".nii.gz -expr 'ispositive(a)' -prefix ./wellbeing7d_control_ic_normz"$thresh"_cs"$clust"_masks/ic${ic}_normz"$thresh"_cs"$clust"_mask.nii.gz
done 

for ic in {1..100}
do
max_num=`3dBrickStat -max ./wellbeing7d_control_ic_normz"$thresh"_cs"$clust"_masks/ic${ic}_normz"$thresh"_cs"$clust"_mask.nii.gz`
if [ $max_num == 0 ]
then 
  rm ./wellbeing7d_control_ic_normz"$thresh"_cs"$clust"_masks/ic${ic}_normz"$thresh"_cs"$clust"_mask.nii.gz
fi
done