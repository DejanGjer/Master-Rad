#!/bin/bash

# Define training parameters
noise=0.8
epochs=50
objective="denoising"

# Define computational resources
node="n16"  # Specify the node to run on

# Format noise value for directory name (e.g., 0.3 -> 0_30)
noise_str=$(printf "%.2f" $noise | sed 's/\./_/')0

# Define the output directory with both noise and objective in the name
outdir="/home/dgjer/master/Master-Rad/denoised_smoothing_results/stability_obj/cifar10_dncnn_${objective}_${noise_str}"

# Create the output directory if it doesn't exist
mkdir -p "$outdir"

# Save parameters to a file
params_file="$outdir/training_parameters.txt"
{
    echo "Training parameters:"
    echo "==================="
    echo "Output directory: $outdir"
    echo "Noise level: $noise"
    echo "Number of epochs: $epochs"
    echo "Objective: $objective"
    echo "Node: $node"
    echo "Timestamp: $(date)"
    echo "==================="
} > "$params_file"

echo "Parameters saved to: $params_file"

# Submit the job
sbatch \
    --nodelist="$node" \
    --output="$outdir/%x-%j.out" \
    --error="$outdir/%x-%j.err" \
    run_denoised_smoothing_training.sh \
    --outdir="$outdir" \
    --noise="$noise" \
    --epochs="$epochs" \
    --objective="$objective"