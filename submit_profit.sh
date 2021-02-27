#!/bin/bash
turbine_array=( 1 )
setback_array=( 1.1 )
ppa_array=( 1.01 1.05 1.1 1.2 )

for i in "${turbine_array[@]}"
do
   : 
   for j in "${setback_array[@]}"
   do
      : 
      for k in "${ppa_array[@]}"
      do
         :
         sbatch runscript_remove.sh $i $j profit $k
      done
   done
done