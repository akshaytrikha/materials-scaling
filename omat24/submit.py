import submitit

from train import main
from arg_parser import get_args


gpu_table = {
    "A40": {"partition": "savio3_gpu", "qos": "a40_gpu3_normal"},
    "A5000": {"partition": "savio4_gpu", "qos": "a5k_gpu4_normal"},
}


def main_wrapper(args):
    print("Starting job")
    main(args=args)


if __name__ == "__main__":
    args = get_args()

    # Setup submitit executor
    executor = submitit.AutoExecutor(folder="slurm_logs")
    executor.update_parameters(
        nodes=1,
        tasks_per_node=args.gpus,
        cpus_per_task=args.cpus_per_task,
        timeout_min=int(args.time.split(":")[0]) * 60 + int(args.time.split(":")[1]),
        slurm_partition=gpu_table[args.gpu]["partition"],
        slurm_account=args.account,
        slurm_qos=gpu_table[args.gpu]["qos"],
        slurm_job_name=args.job_name,
        slurm_gres=f"gpu:{args.gpu}:{args.gpus}",
    )

    # Create job
    print("Submitting job...")
    job = executor.submit(main_wrapper, args)
    print(f"Submitted job ID: {job.job_id}")
