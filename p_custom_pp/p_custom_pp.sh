#! /bin/bash

# first job - no dependencies
# export jid1=$(sbatch --cpus-per-task=4 ft_mod1.sh | sed -n 's/.*job //p')
# export jid1=$(sbatch -x idc-beta-batch-pvc-node-[03,20] ft_mod1.sh | sed -n 's/.*job //p')
# export jid1=$(sbatch -x idc-beta-batch-pvc-node-[03,20] --priority 0 --job-name pcs1 --dependency=afterany:22088 p_custom.sh | sed -n 's/.*job //p')
export jid1=$(sbatch -x idc-beta-batch-pvc-node-[05] --priority 0 --job-name pcs1 p_custom.sh | sed -n 's/.*job //p')


echo $jid1
# multiple jobs can depend on a single job
export jid2=$(sbatch  -w idc-beta-batch-pvc-node-[05] --priority 0 --job-name pcs2 --dependency=afterany:$jid1  p_custom.sh | sed -n 's/.*job //p')
export jid3=$(sbatch  -w idc-beta-batch-pvc-node-[05] --priority 0 --job-name pcs3 --dependency=afterany:$jid2  p_custom.sh | sed -n 's/.*job //p')
export jid4=$(sbatch  -w idc-beta-batch-pvc-node-[05] --priority 0 --job-name pcs4 --dependency=afterany:$jid3  p_custom.sh | sed -n 's/.*job //p')


# show dependencies in squeue output:
squeue -u $USER -o "%.8A %.4C %.10m %.20E"

# sbatch  --dependency=afterany:20833  ft_mod2.sh
# sbatch  --dependency=afterany:20834 ft_mod3.sh
# sbatch  --dependency=afterany:20835  ft_mod4.sh