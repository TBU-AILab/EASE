#!/bin/bash
#$ -cwd                        # Run job from current working directory
#$ -N apptainer_build          # Job name
#$ -pe smp 8                   # Request 8 CPU cores
#$ -l h_rt=01:00:00            # Runtime limit: 1 hour
#$ -l mem_free=4G              # Request 4GB RAM per core (total: 32GB)
#$ -j y                        # Merge stdout and stderr into one output file
#$ -o build_combined.log       # Combined log file

echo "=== Build job started on $(hostname) at $(date) ==="

# Export NSLOTS to pass into container environment
export APPTAINERENV_NSLOTS=$NSLOTS

# Build the container image
apptainer build planetwars.sif planetwars.def

echo "=== Build job finished at $(date) ==="
