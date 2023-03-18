#!/bin/bash

if [ $# -ne 14 ]; then
    echo '3dclass.sh jobno twist rise iter folder_name'
fi
script_dir=$(cd $(dirname $0);pwd)
jobno=$1
helical_twist_initial=$2
helical_rise_initial=$3
helical_twist_min=$2
helical_twist_max=$2
helical_rise_min=$3
helical_rise_max=$3
angpix=$4
particle_diameter=$5
helical_outer_diameter=$6
helical_inner_diameter=$7
helical_z_percentage=$8
helical_nr_asu=$9
sym=${10}
tau2_fudge=${11}
K=${12}
iter=${13}
relion_gpu=${14}
relion_thread=${15}
folder_name=${16}
inputfile=${17}
referencefile=${18}
ini_high=${19}
output_dir=${script_dir}/${folder_name}
if [ ! -d ${output_dir} ]; then
    mkdir ${output_dir}
fi



job_dir=${output_dir}/job${jobno}

today=`date '+%Y%m%d'`
# init_model=init_ref_featureless.mrc
# K=1

# angpix=3.0
# particle_diameter=270.0
# helical_outer_diameter=240.0
# helical_inner_diameter=10.0

if [ ! -d ${job_dir} ]; then
    mkdir ${job_dir}
fi

echo 'jobno='${jobno}
echo 'output_dir='${output_dir}
echo 'script_dir='${script_dir}
echo 'inputfile='${inputfile}
echo 'referencefile='${referencefile}
echo 'folder_name='${folder_name}
echo 'helical_twist_initial='${helical_twist_initial}
echo 'helical_rise_initial='${helical_rise_initial}
echo 'angpix='${angpix}
echo 'particle_diameter='${particle_diameter}
echo 'helical_outer_diameter='${helical_outer_diameter}
echo 'helical_inner_diameter='${helical_inner_diameter}
echo 'helical_z_percentage='${helical_z_percentage}
echo 'helical_nr_asu='${helical_nr_asu}
echo 'sym='${sym}
echo 'tau2_fudge='${tau2_fudge}
echo 'K='${K}
echo 'iter='${iter}
echo 'relion_gpu='${relion_gpu}
echo 'relion_thread='${relion_thread}
echo 'ini_high='${ini_high}

if [ ${angpix} = "0" ]; then
    command="relion_refine --o ${job_dir}/run --i ${inputfile} --ref ${referencefile} --firstiter_cc --ini_high ${ini_high} --dont_combine_weights_via_disc --pool 1 --ctf --iter ${iter} --tau2_fudge ${tau2_fudge} --particle_diameter ${particle_diameter} --K ${K} --flatten_solvent --zero_mask --oversampling 1 --healpix_order 3 --offset_range 5 --offset_step 2 --sym ${sym} --norm --scale  --helix --helical_inner_diameter ${helical_inner_diameter} --helical_outer_diameter ${helical_outer_diameter} --helical_nr_asu ${helical_nr_asu} --helical_twist_initial ${helical_twist_initial} --helical_rise_initial ${helical_rise_initial} --helical_z_percentage ${helical_z_percentage} --helical_symmetry_search --helical_twist_min ${helical_twist_min} --helical_twist_max ${helical_twist_max} --helical_rise_min ${helical_rise_min} --helical_rise_max ${helical_rise_max} --sigma_tilt 5 --sigma_psi 3.33333 --j ${relion_thread} --gpu ${relion_gpu} 2> ${job_dir}/err.log | tee ${job_dir}/class3d_${today}_job${jobno}.log"
else
    command="relion_refine --o ${job_dir}/run --i ${inputfile} --ref ${referencefile} --firstiter_cc --ini_high ${ini_high} --dont_combine_weights_via_disc --pool 1 --ctf --iter ${iter} --tau2_fudge ${tau2_fudge} --particle_diameter ${particle_diameter} --K ${K} --flatten_solvent --zero_mask --oversampling 1 --healpix_order 3 --offset_range 5 --offset_step 2 --sym ${sym} --norm --scale  --helix --helical_inner_diameter ${helical_inner_diameter} --helical_outer_diameter ${helical_outer_diameter} --helical_nr_asu ${helical_nr_asu} --helical_twist_initial ${helical_twist_initial} --helical_rise_initial ${helical_rise_initial} --helical_z_percentage ${helical_z_percentage} --helical_symmetry_search --helical_twist_min ${helical_twist_min} --helical_twist_max ${helical_twist_max} --helical_rise_min ${helical_rise_min} --helical_rise_max ${helical_rise_max} --sigma_tilt 5 --sigma_psi 3.33333 --j ${relion_thread} --gpu ${relion_gpu} --angpix ${angpix} 2> ${job_dir}/err.log | tee ${job_dir}/class3d_${today}_job${jobno}.log"
fi

echo $command

#cp ${script_dir}/preprocessed.mrcs ${job_dir}/
echo $command > ${job_dir}/note.txt
date >> ${job_dir}/note.txt
eval $command
date >> ${job_dir}/note.txt
cp ${0} ${job_dir}/
