#!/bin/bash
#$ -cwd                        # Run from current directory
#$ -N apptainer_run            # Job name
#$ -pe smp 4                   # Request 4 CPU cores
#$ -l h_rt=00:30:00            # Runtime limit
#$ -l mem_free=2G              # 2GB per core (8GB total)
#$ -j y                        # Merge stdout and stderr
#$ -o run_output.log           # Combined output log

echo "=== Run job started on $(hostname) at $(date) ==="

# Set number of cores inside the container if needed
export APPTAINERENV_NSLOTS=$NSLOTS

# Run the container
apptainer run planetwars.sif

echo "=== Run job finished at $(date) ==="
