#!/bin/bash

# Test configuration parameters
attack_type="fgsm"                # Attack type to test against
use_randomized_smoothing=True    # Whether to use randomized smoothing
n=500                           # Number of samples for smoothing
alpha=0.001                      # Smoothing parameter
sigma=0.5                      # Noise level for smoothing
denoiser_arch="cifar_dncnn"           # Architecture of the denoiser
classifier_path="/home/dgjer/master/Master-Rad/base_training_checkpoints/sweep_2025-07-06_01-56-46/2025-07-06_01-56-46/checkpoints/model_normal.pth"  # Path to classifier model
denoiser_path="/home/dgjer/master/Master-Rad/denoised_smoothing_results/stability_obj/cifar10_dncnn_stability_0_500/best.pth.tar"  # Path to trained denoiser
save_root_path="../denoised_smoothing_results/tests/stability_obj"  # Root path for saving results

# Create timestamped directory
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
save_root_path="${save_root_path}/${timestamp}"

node="n16"

original_code_dir=".."

# Create code directory in save_root_path
mkdir -p "${save_root_path}/code"

# Copy all files and directories recursively, preserving directory structure
cp -r "${original_code_dir}/denoised_smoothing" "${save_root_path}/code/"

# Copy all top-level files
find "$original_code_dir" -maxdepth 1 -type f -exec cp {} "${save_root_path}/code/" \;

# Update config_test.py in code dir
config_file="${save_root_path}/code/config_test.py"
sed -i "s|^attack_type *= *.*|attack_type = \"${attack_type}\"|" "$config_file"
sed -i "s|^use_randomized_smoothing *= *.*|use_randomized_smoothing = ${use_randomized_smoothing}|" "$config_file"
sed -i "s|^n *= *.*|n = ${n}|" "$config_file"
sed -i "s|^alpha *= *.*|alpha = ${alpha}|" "$config_file"
sed -i "s|^denoiser_arch *= *.*|denoiser_arch = \"${denoiser_arch}\"|" "$config_file"
sed -i "s|^denoiser_path *= *.*|denoiser_path = \"${denoiser_path}\"|" "$config_file"
sed -i "s|^save_root_path *= *.*|save_root_path = \"../\"|" "$config_file"
sed -i "s|^sigma *= *.*|sigma = ${sigma}|" "$config_file"

# Update test_model_paths to contain only the classifier path
# First, clean up the existing test_model_paths array
sed -i '/^test_model_paths/,/]/c\test_model_paths = []' "$config_file"
# Then add the new classifier path
sed -i "s|^test_model_paths = \[\]|test_model_paths = ['${classifier_path}']|" "$config_file"

# Path to test_denoiser.py inside code dir
test_py_path="${save_root_path}/code/test_denoiser.py"

# Submit job with test_denoiser.py path as argument
sbatch --nodelist="$node" --output="${save_root_path}/slurm.%j.out" --error="${save_root_path}/slurm.%j.err" run_base_training.sh "$test_py_path"

echo -e "Submitted test job with: \nattack_type: ${attack_type} \nuse_randomized_smoothing: ${use_randomized_smoothing} \nn: ${n} \nalpha: ${alpha} \nsigma: ${sigma} \ndenoiser_arch: ${denoiser_arch} \ndenoiser_path: ${denoiser_path} \nclassifier_path: ${classifier_path} \nsave_root_path: ${save_root_path} \nnode: ${node}"
echo "----------------------------------------------------------------------------------------------------------"