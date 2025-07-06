#!/bin/bash

lrs=(0.03 0.1 0.3)
weight_decays=(0.001 0.005 0.01)
batch_sizes=(128 768)
nodes=("n17" "n17" "n19" "n20")

original_code_dir=".."

# Create a sweep directory with timestamp
sweep_timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
sweep_dir="base_training_checkpoints/sweep_${sweep_timestamp}"
mkdir -p "../${sweep_dir}"

for i in "${!lrs[@]}"; do
    lr="${lrs[$i]}"
    for j in "${!weight_decays[@]}"; do
        weight_decay="${weight_decays[$j]}"
        for k in "${!batch_sizes[@]}"; do
            batch_size="${batch_sizes[$k]}"
            
            # Create a unique log directory for each job inside sweep_dir
            timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
            log_dir="${sweep_dir}/${timestamp}"
            mkdir -p "../${log_dir}/code"
            
            # Copy all files (not directories) from original_code_dir to code dir
            find "$original_code_dir" -maxdepth 1 -type f -exec cp {} "../${log_dir}/code/" \;
            
            # Update config_train.py in code dir
            config_file="../${log_dir}/code/config_train.py"
            sed -i "s|^learning_rate *= *.*|learning_rate = ${lr}|" "$config_file"
            sed -i "s|^decay *= *.*|decay = ${weight_decay}|" "$config_file"
            sed -i "s|^batch_size *= *.*|batch_size = ${batch_size}|" "$config_file"
            sed -i "s|^base_save_dir *= *.*|base_save_dir = '..'|" "$config_file"
            sed -i "s|^create_new_saving_dir *= *.*|create_new_saving_dir = False|" "$config_file"
            sed -i "s|^save_config_file *= *.*|save_config_file = False|" "$config_file"
            
            # Select node
            node_index=$(( ((i * ${#weight_decays[@]} * ${#batch_sizes[@]}) + j * ${#batch_sizes[@]} + k ) % ${#nodes[@]} ))
            selected_node="${nodes[$node_index]}"
            
            # Path to train.py inside code dir
            train_py_path="../${log_dir}/code/train.py"
            
            # Submit job with train.py path as argument
            sbatch --nodelist="$selected_node" --output="../${log_dir}/slurm.%j.out" --error="../${log_dir}/slurm.%j.err" run_base_training.sh "$train_py_path"
            
            echo -e "Submitted job with \nlearning rate: ${lr} \nweight decay: ${weight_decay}\n batch size: ${batch_size}\n node ${selected_node}\n log directory: ${log_dir}"
            echo "----------------------------------------------------------------"
            
            sleep 1
        done
    done
done
