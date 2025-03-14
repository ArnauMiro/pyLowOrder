#!/bin/bash
#
# Automatic generation of cases for scalability tests running on GPU.
# Script launch is thought for a HPC cluster with SLURM for queue management.
# Queue parameters and modules are tuned to work for the accelerated partition of MareNostrum 5, adjust accordingly for your cluster.
# Adjust script path accordingly

# Matrix sizes to run scalability. They are inputs to the python scripts. A folder will be created for each combination of M and N
Ms=(2e6 3e6 4e6)
Ns=(2000)

# Number of nodes to test (NPROCS=NNODES*NPROCSXNODE)
NNODES=(1)
NPROCNODE=4 #MN5 has 4 GPU per node

# Function to test the scalability of
FUNC='SVD'
TEST='weakM'

# Script to run
SCRIPT="../example_${FUNC}_GPU_${TEST}.py"

################
# AUTOMATED PART
################
for M in ${Ms[@]}; do
	for N in ${Ns[@]}; do
		mkdir "${FUNC}_${TEST}_M${M}_N${N}"
		cd "${FUNC}_${TEST}_M${M}_N${N}"
		for n in ${NNODES[@]}; do
			# Compute number of tasks required
			p=$(( $n * $NPROCNODE ))
			# File name
			file="submit_${p}.sh"
			echo "\
#!/bin/bash
#SBATCH --job-name=${FUNC}_${n}
#SBATCH --output=sca${p}.out
#SBATCH --error=sca${p}.err
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --nodes=${n}
#SBATCH --time=0:20:00
#SBATCH --qos=acc_debug
#SBATCH --account=bsc21
#SBATCH --exclusive
module purge
module load mkl/2023.2.0 nvidia-hpc-sdk/24.3 hdf5/1.14.1-2-nvidia-nvhpcx python/3.12.1-gcc cuda/12.2 && source ~/gpumpi/bin/activate && unset PYTHONPATH
export PYTHONPATH=/home/bsc/bsc021893/pyLowOrder:$PYTHONPATH
export SLURM_CPU_BIND=none
date
mpirun -np ${p} python ${SCRIPT} ${M} ${N}
date
			" > $file
			sbatch $file
		done
	cd ..
	done
done


