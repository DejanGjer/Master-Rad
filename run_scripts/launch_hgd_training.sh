#!/bin/bash

# Hyperparameters to sweep
learning_rates=(0.001 0.003 0.01)
bilinear_values=(False True)
learn_noise_values=(False True)
loss_types=("pgd" "lgd")

# Single-value parameters
dataset_name="mnist"
attack_type="fgsm"

nodes=("n16" "n19", "n20")

original_code_dir=".."

# Create a sweep directory with timestamp
sweep_timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
sweep_dir="hgd_training_results/sweep_${sweep_timestamp}"
mkdir -p "../${sweep_dir}"

for lr in "${learning_rates[@]}"; do
    for bilinear in "${bilinear_values[@]}"; do
        for learn_noise in "${learn_noise_values[@]}"; do
            for loss in "${loss_types[@]}"; do
                # Create a unique log directory for each job inside sweep_dir
                timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
                log_dir="${sweep_dir}/${timestamp}"
                mkdir -p "../${log_dir}/code"

                # Copy all files (not directories) from original_code_dir to code dir
                find "$original_code_dir" -maxdepth 1 -type f -exec cp {} "../${log_dir}/code/" \;

                # Update config.py in code dir
                config_file="../${log_dir}/code/config.py"
                sed -i "s|^dataset_name *= *.*|dataset_name = \"${dataset_name}\"|" "$config_file"
                sed -i "s|^attack_type *= *.*|attack_type = \"${attack_type}\"|" "$config_file"
                sed -i "s|^learning_rate *= *.*|learning_rate = ${lr}|" "$config_file"
                sed -i "s|^bilinear *= *.*|bilinear = ${bilinear}|" "$config_file"
                sed -i "s|^learn_noise *= *.*|learn_noise = ${learn_noise}|" "$config_file"
                sed -i "s|^loss *= *.*|loss = \"${loss}\"|" "$config_file"

                # Select node
                node_index=$(( (RANDOM) % ${#nodes[@]} ))
                selected_node="${nodes[$node_index]}"

                # Path to main.py inside code dir
                main_py_path="../${log_dir}/code/main.py"

                # Submit job with main.py path as argument
                sbatch --nodelist="$selected_node" --output="../${log_dir}/slurm.%j.out" --error="../${log_dir}/slurm.%j.err" run_base_training.sh "$main_py_path"

                echo -e "Submitted job with \ndataset_name: ${dataset_name} \nattack_type: ${attack_type} \nlearning_rate: ${lr} \nbilinear: ${bilinear} \nlearn_noise: ${learn_noise} \nloss: ${loss} \nnode: ${selected_node} \nlog directory: ${log_dir}"
                echo "----------------------------------------------------------------------------------------------------------"

                sleep 1
            done
        done
    done
done