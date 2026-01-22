#!/bin/bash
#$ -cwd
#$ -N apptainer_build
#$ -l h_rt=1:00:00
#$ -o build_output.log
#$ -e build_error.log

echo "Build job started on $(hostname) at $(date)"
apptainer build planetwars.sif planetwars.def
echo "Build job finished at $(date)"
