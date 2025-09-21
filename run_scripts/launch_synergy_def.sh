#!/bin/bash

# Single-value parameters
dataset_name="cifar100"
attack_type="fgsm"

# Define nodes
nodes=("n16" "n20")

# Define model types (these will be used to construct paths)
model_types=("normal" "negative" "hybrid_nor" "hybrid_neg" "synergy_nor" "synergy_neg" "synergy_all" "tr_synergy_all")

# Define checkpoint directories for each dataset
declare -A checkpoint_dirs
checkpoint_dirs["cifar10"]="/home/dgjer/master/Master-Rad/base_training_checkpoints/sweep_2025-07-06_01-56-46/2025-07-06_01-56-46/checkpoints"
checkpoint_dirs["mnist"]="/home/dgjer/master/Master-Rad/base_training_checkpoints/sweep_2025-07-06_19-40-43/2025-07-06_19-40-48/checkpoints"
checkpoint_dirs["cifar100"]="/home/dgjer/master/Master-Rad/base_training_checkpoints/sweep_2025-07-08_00-24-53/2025-07-08_00-24-53/checkpoints"

original_code_dir=".."

# Create a sweep directory with timestamp
sweep_timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
sweep_dir="synergy_defenses_results/sweep_${sweep_timestamp}"
mkdir -p "../${sweep_dir}"

# Get the checkpoint directory based on dataset_name
checkpoint_dir=${checkpoint_dirs[$dataset_name]}

# Iterate through model types
for model_type in "${model_types[@]}"; do
    # Create model path
    model_path="${checkpoint_dir}/model_${model_type}.pth"
    
    # Create a unique log directory for each job inside sweep_dir
    timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    log_dir="${sweep_dir}/${timestamp}"
    mkdir -p "../${log_dir}/code"

    # Copy all files (not directories) from original_code_dir to code dir
    find "$original_code_dir" -maxdepth 1 -type f -exec cp {} "../${log_dir}/code/" \;

    # Update config.py in code dir
    config_file="../${log_dir}/code/config.py"
    
    # Update dataset_name and attack_type
    sed -i "s|^dataset_name *= *.*|dataset_name = '${dataset_name}'|" "$config_file"
    sed -i "s|^attack_type *= *.*|attack_type = '${attack_type}'|" "$config_file"
    
    # Update train_model_paths
    # First, remove all existing paths
    sed -i "/^train_model_paths = \[/,/\]/c\train_model_paths = [\n    '${model_path}'\n]" "$config_file"

    # Select node
    node_index=$(( (RANDOM) % ${#nodes[@]} ))
    selected_node="${nodes[$node_index]}"

    # Path to main.py inside code dir
    main_py_path="../${log_dir}/code/main.py"

    # Submit job with main.py path as argument
    sbatch --nodelist="$selected_node" --output="../${log_dir}/slurm.%j.out" --error="../${log_dir}/slurm.%j.err" run_base_training.sh "$main_py_path"

    echo -e "Submitted job with \ndataset_name: ${dataset_name} \nattack_type: ${attack_type} \nmodel_type: ${model_type} \nmodel_path: ${model_path} \nnode: ${selected_node} \nlog directory: ${log_dir}"
    echo "----------------------------------------------------------------------------------------------------------"

    sleep 1
done