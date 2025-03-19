from datetime import time
import submitit
import argparse


gpu_table = {
    "A40": {"partition": "savio3_gpu", "qos": "a40_gpu3_normal"},
    "A5000": {"partition": "savio4_gpu", "qos": "a5k_gpu4_normal"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Submit job to Slurm")
    parser.add_argument("--job_name", type=str, default="omat24-eqv2")
    parser.add_argument("--account", type=str, default="ac_msemeng")
    parser.add_argument("--time", type=str, default="00:30:00")
    parser.add_argument("--gpu", type=str, default="A40")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus_per_task", type=int, default=8)
    return parser.parse_args()


def main_wrapper(args):
    print("Starting job")
    print(args)
    time.sleep(10)
    print("Job finished")


if __name__ == "__main__":
    args = parse_args()

    # Setup submitit executor
    executor = submitit.AutoExecutor(folder="slurm_logs")
    executor.update_parameters(
        mem_gb=32,
        gpus_per_node=args.gpus,
        tasks_per_node=1,
        cpus_per_task=args.cpus_per_task,
        nodes=1,
        timeout_min=int(args.time.split(":")[0]) * 60 + int(args.time.split(":")[1]),
        slurm_partition=gpu_table[args.gpu]["partition"],
        slurm_account=args.account,
        slurm_qos=gpu_table[args.gpu]["qos"],
        slurm_job_name=args.job_name,
    )

    # Create job
    job = executor.submit(main_wrapper, args)
    print(f"Submitted job ID: {job.job_id}")
