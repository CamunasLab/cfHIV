#!/bin/bash
#SBATCH -A naiss2025-22-186
#SBATCH --job-name=decon_array
#SBATCH --output=decon_array_%A_%a.out
#SBATCH --error=decon_array_%A_%a.err
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --partition=shared
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=1000:00
#SBATCH --array=0-99   # adjust upper bound to number_of_samples-1

# Move to your project dir
cd ~/camunaslab/pipelines/hiv_deconv

# Get sample name for this array task
SAMPLE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" sample_names.txt)

echo "Running deconvolution for sample: $SAMPLE"

python3 -c "import deconvolve as deconv; \
deconv.main(1, 'gene_names_x_samples.csv', ['$SAMPLE'], 'NNLS', '$SAMPLE', jackknife=False)"
