#!/bin/bash
#clade1--------------------
rm clade1
mkdir clade1
clust=50

for ic in 29 31 32 43 50 51 54 60 64 68 78 80 86 87 88
do 
  cp /Users/mac/Desktop/wbid_tem/drop83/neurovault_niftis/500days_searchlight_isrsa_wellbeing_control_4D_5470secs_dim100_ica_stats/thresh_zstat${ic}.nii.gz ./clade1
done

3dmerge \
  -nozero -gnzmean \
  -dxyz=1 -1clust 1 "$clust" \
  -prefix clade1_sigics_merge_prob0.5_c"$clust".nii.gz \
  ./clade1/thresh_zstat*.nii.gz 
  
3dresample -dxyz 1 1 1 -rmode Linear -prefix clade1_sigics_merge_prob0.5_c"$clust"_resample.nii.gz -input clade1_sigics_merge_prob0.5_c"$clust".nii.gz

#clade2--------------------
rm clade2
mkdir clade2
clust=50

for ic in 42 55 57 63 67 72 96
do 
  cp /Users/mac/Desktop/wbid_tem/drop83/neurovault_niftis/500days_searchlight_isrsa_wellbeing_control_4D_5470secs_dim100_ica_stats/thresh_zstat${ic}.nii.gz ./clade2
done

3dmerge \
  -nozero -gnzmean \
  -dxyz=1 -1clust 1 "$clust" \
  -prefix clade2_sigics_merge_prob0.5_c"$clust".nii.gz \
  ./clade2/thresh_zstat*.nii.gz 
  
3dresample -dxyz 1 1 1 -rmode Linear -prefix clade2_sigics_merge_prob0.5_c"$clust"_resample.nii.gz -input clade2_sigics_merge_prob0.5_c"$clust".nii.gz

#clade3--------------------
rm clade3
mkdir clade3
clust=50

for ic in 7 8 12 14
do 
  cp /Users/mac/Desktop/wbid_tem/drop83/neurovault_niftis/500days_searchlight_isrsa_wellbeing_control_4D_5470secs_dim100_ica_stats/thresh_zstat${ic}.nii.gz ./clade3
done

3dmerge \
  -nozero -gnzmean \
  -dxyz=1 -1clust 1 "$clust" \
  -prefix clade3_sigics_merge_prob0.5_c"$clust".nii.gz \
  ./clade3/thresh_zstat*.nii.gz 

3dresample -dxyz 1 1 1 -rmode Linear -prefix clade3_sigics_merge_prob0.5_c"$clust"_resample.nii.gz -input clade3_sigics_merge_prob0.5_c"$clust".nii.gz

#clade4--------------------
rm clade4
mkdir clade4
clust=50

for ic in 18
do 
  cp /Users/mac/Desktop/wbid_tem/drop83/neurovault_niftis/500days_searchlight_isrsa_wellbeing_control_4D_5470secs_dim100_ica_stats/thresh_zstat${ic}.nii.gz ./clade4
done

3dmerge \
  -nozero -gnzmean \
  -dxyz=1 -1clust 1 "$clust" \
  -prefix clade4_sigics_merge_prob0.5_c"$clust".nii.gz \
  ./clade4/thresh_zstat*.nii.gz 
    
3dresample -dxyz 1 1 1 -rmode Linear -prefix clade4_sigics_merge_prob0.5_c"$clust"_resample.nii.gz -input clade4_sigics_merge_prob0.5_c"$clust".nii.gz

#clade5--------------------
rm clade5
mkdir clade5
clust=50

for ic in 15 40
do 
  cp /Users/mac/Desktop/wbid_tem/drop83/neurovault_niftis/500days_searchlight_isrsa_wellbeing_control_4D_5470secs_dim100_ica_stats/thresh_zstat${ic}.nii.gz ./clade5
done

3dmerge \
  -nozero -gnzmean \
  -dxyz=1 -1clust 1 "$clust" \
  -prefix clade5_sigics_merge_prob0.5_c"$clust".nii.gz \
  ./clade5/thresh_zstat*.nii.gz 
  
3dresample -dxyz 1 1 1 -rmode Linear -prefix clade5_sigics_merge_prob0.5_c"$clust"_resample.nii.gz -input clade5_sigics_merge_prob0.5_c"$clust".nii.gz

#clade6--------------------
rm clade6
mkdir clade6
clust=50

for ic in 22 58 59 62 76 85 99
do 
  cp /Users/mac/Desktop/wbid_tem/drop83/neurovault_niftis/500days_searchlight_isrsa_wellbeing_control_4D_5470secs_dim100_ica_stats/thresh_zstat${ic}.nii.gz ./clade6
done

3dmerge \
  -nozero -gnzmean \
  -dxyz=1 -1clust 1 "$clust" \
  -prefix clade6_sigics_merge_prob0.5_c"$clust".nii.gz \
  ./clade6/thresh_zstat*.nii.gz 
  
3dresample -dxyz 1 1 1 -rmode Linear -prefix clade6_sigics_merge_prob0.5_c"$clust"_resample.nii.gz -input clade6_sigics_merge_prob0.5_c"$clust".nii.gz

#clade7--------------------
rm clade7
mkdir clade7
clust=50

for ic in 9 10 16 19 21 26 28 35 44 47 66 70 90
do 
  cp /Users/mac/Desktop/wbid_tem/drop83/neurovault_niftis/500days_searchlight_isrsa_wellbeing_control_4D_5470secs_dim100_ica_stats/thresh_zstat${ic}.nii.gz ./clade7
done

3dmerge \
  -nozero -gnzmean \
  -dxyz=1 -1clust 1 "$clust" \
  -prefix clade7_sigics_merge_prob0.5_c"$clust".nii.gz \
  ./clade7/thresh_zstat*.nii.gz 
    
3dresample -dxyz 1 1 1 -rmode Linear -prefix clade7_sigics_merge_prob0.5_c"$clust"_resample.nii.gz -input clade7_sigics_merge_prob0.5_c"$clust".nii.gz

