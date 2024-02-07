# Use this script to run multiple Hydra sweeps in parallel with SLURM + Joblib.
# Because runs are very short, it's best to run them in parallel over multiple CPUs
# with Hydra Joblib plugin.
# To further parallelize everything, we recommend to submit multiple SLURM jobs
# with low compute requirements. Having low requirements ensures your job gets
# high priority.
# Each job will take care of a chunk of the seed range.
# Note that this works because runs are quick and need only CPUs.

# This script submits 10 jobs, each running the 'medium_det' sweep over 10 different seeds.
# Each job keeps launching 8 runs in parallel (because we request 8 CPUs).
# Note that sweeps on noisy environments take x10 time, so you need to request more time.

# To check the job's progress
# cat /scratch/USERNAME/slurm_out/JOB_ID_0_log.out

import os
import submitit
import numpy as np

n_chunks = 20
seeds_chunks = np.array_split(range(0, 100), n_chunks)

username = os.environ["USER"]

for seeds in seeds_chunks:
    executor = submitit.AutoExecutor(
        folder=f"/scratch/{username}/slurm_out"
    )  # you will find error logs here
    executor.update_parameters(
        slurm_account="def-mbowling",
        timeout_min=59,
        nodes=1,
        cpus_per_task=4,
        mem_gb=1,
    )
    cmd = (
        "python main.py "
        "-m hydra/launcher=joblib "
        "hydra/sweeper=medium_det "
        "hydra.launcher.verbose=1000 "
        "experiment.rng_seed=" + ",".join(map(str, seeds))
    )
    job = executor.submit(os.system, cmd)
    print(f"Submitted job: {job.job_id}")
