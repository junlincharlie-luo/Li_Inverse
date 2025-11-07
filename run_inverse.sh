#!/bin/bash
#SBATCH --job-name=li_inverse
#SBATCH --output=li_inverse_%j.out
#SBATCH --error=li_inverse_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@stanford.edu

# Load required modules
module load python/3.9
module load py-numpy/1.26.0
module load py-scipy/1.11.2

# Activate virtual environment (if you have one)
# source /path/to/your/venv/bin/activate

# Or load conda environment
# module load miniconda3
# source activate your_env_name

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Run the Python script
python Generated_by_GPT.py

# Print completion time
echo "Job finished at: $(date)"
