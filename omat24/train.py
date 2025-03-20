# Increase file descriptor limit for using GH200 on Lambda Labs
# import patch_torch_scatter

# # Increase file descriptor limits
# import resource
# try:
#     soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
#     new_soft = min(65536, hard)  # Increase to maximum allowed or 65536
#     resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
#     print(f"Increased file descriptor limit from {soft} to {new_soft}")
# except Exception as e:
#     print(f"Warning: Could not increase file descriptor limit: {e}")

import warnings

warnings.filterwarnings(
    "ignore", message="You are using `torch.load` with `weights_only=False`"
)
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")

# External
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
import pprint
import json
import math
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import subprocess
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Internal
from data import get_dataloaders
from data_utils import download_dataset, VALID_DATASETS, DATASET_INFO
from arg_parser import get_args
from models.fcn import MetaFCNModels
from models.transformer_models import MetaTransformerModels
from models.schnet import MetaSchNetModels
from models.equiformer_v2 import MetaEquiformerV2Models
from train_utils import train

# Set seed & device
SEED = 1024
torch.manual_seed(SEED)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(True)
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def setup_ddp(rank, world_size):
    """Initialize DDP process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP process group."""
    dist.destroy_process_group()


def main(rank=None, world_size=None):
    is_main_process = rank == 0 if world_size is not None else True
    args = get_args()
    log = not args.no_log
    global DEVICE

    # DDP setup if world_size is provided
    if world_size is not None:
        setup_ddp(rank, world_size)
        DEVICE = torch.device(f"cuda:{rank}")
        dist.barrier()

    # Convinience for running all datasets
    if args.datasets[0] == "all":
        args.datasets = VALID_DATASETS

    # Download datasets if not present
    dataset_paths = []
    for dataset_name in args.datasets:
        dataset_path = Path(
            f"{args.datasets_base_path}/{args.split_name}/{dataset_name}"
        )
        if not dataset_path.exists():
            download_dataset(dataset_name, args.split_name, args.datasets_base_path)
        dataset_paths.append(dataset_path)

    # User Hyperparam Feedback
    params = vars(args) | {
        "dataset_split": args.split_name,
    }
    if is_main_process:
        pprint.pprint(params)
        print()

    batch_size = args.batch_size[0]
    lr = args.lr[0]
    num_epochs = args.epochs
    use_factorize = args.factorize
    graph = args.architecture in ["SchNet", "EquiformerV2"]

    # Initialize meta model class based on architecture choice
    if args.architecture == "FCN":
        meta_models = MetaFCNModels(
            vocab_size=args.n_elements, use_factorized=use_factorize
        )
    elif args.architecture == "Transformer":
        meta_models = MetaTransformerModels(
            vocab_size=args.n_elements,
            max_seq_len=DATASET_INFO[args.split_name]["all"]["max_n_atoms"],
            use_factorized=use_factorize,
        )
    elif args.architecture == "SchNet":
        if DEVICE == torch.device("mps"):
            print("MPS is not supported for SchNet. Switching to CPU.")
            DEVICE = torch.device("cpu")
        meta_models = MetaSchNetModels(device=DEVICE)
    elif args.architecture == "EquiformerV2":
        if DEVICE == torch.device("mps"):
            print("MPS is not supported for EquiformerV2. Switching to CPU.")
            DEVICE = torch.device("cpu")
        meta_models = MetaEquiformerV2Models(device=DEVICE)

    # Create results path and initialize file if logging is enabled
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logdir = os.path.join("runs", f"exp_{timestamp}")
    writer = SummaryWriter(log_dir=tb_logdir)
    if is_main_process:
        print(f"TensorBoard logs will be saved to: {tb_logdir}")

    results_path = Path("results") / f"experiments_{timestamp}.json"
    experiment_results = {}
    if log and is_main_process:
        Path("results").mkdir(exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({}, f)  # Initialize as empty JSON
        print(f"\nLogging enabled. Results will be saved to {results_path}")
    else:
        print("\nLogging disabled. No experiment log will be saved.")

    for data_fraction in args.data_fractions:
        train_loader, val_loader = get_dataloaders(
            dataset_paths,
            train_data_fraction=data_fraction,
            batch_size=batch_size,
            seed=SEED,
            architecture=args.architecture,
            batch_padded=False,
            val_data_fraction=args.val_data_fraction,
            train_workers=args.train_workers,
            val_workers=args.val_workers,
            graph=graph,
            factorize=use_factorize,
            distributed=world_size is not None,
            augment=args.augment,
        )
        dataset_size = len(train_loader.dataset)
        if is_main_process:
            print(
                f"\nTraining on dataset fraction {data_fraction} with {dataset_size} samples"
            )
            if args.augment:
                print("Data augmentation (random rotations) is enabled")

        for model_idx, model in enumerate(meta_models):
            if is_main_process:
                print(
                    f"\nModel {model_idx + 1}/{len(meta_models)} is on device {DEVICE} and has {model.num_params} parameters"
                )

            model.to(DEVICE)

            # Store original model attributes before DDP wrapping
            num_params = model.num_params if hasattr(model, "num_params") else None
            embedding_dim = getattr(model, "embedding_dim", None)
            depth = getattr(model, "depth", None)

            optimizer = optim.AdamW(model.parameters(), lr=lr)

            if world_size is not None:
                # Wrap model in DDP to use multiple GPUs
                model = DDP(
                    model,
                    device_ids=[rank],
                    find_unused_parameters=True,
                    broadcast_buffers=False,
                )

            # lambda_schedule = lambda epoch: 0.5 * (
            #     1 + math.cos(math.pi * epoch / num_epochs)
            # )
            # scheduler = LambdaLR(optimizer, lr_lambda=lambda_schedule)
            scheduler = None

            # Prepare run entry etc.
            model_name = f"model_ds{dataset_size}_p{int(num_params)}"
            checkpoint_path = f"checkpoints/{args.architecture}_ds{dataset_size}_p{int(num_params)}_{timestamp}.pth"
            run_entry = {
                "model_name": model_name,
                "config": {
                    "architecture": args.architecture,
                    "embedding_dim": embedding_dim,
                    "depth": depth,
                    "num_params": num_params,
                    "dataset_size": dataset_size,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "seed": SEED,
                },
                "losses": {},
                "checkpoint_path": checkpoint_path,
            }
            ds_key = str(dataset_size)
            if ds_key not in experiment_results:
                experiment_results[ds_key] = []
            experiment_results[ds_key].append(run_entry)
            if log:
                with open(results_path, "w") as f:
                    json.dump(experiment_results, f, indent=4)

            # Create progress bar only on main process
            if is_main_process:
                progress_bar = tqdm(range(num_epochs + 1))
            else:
                progress_bar = range(num_epochs + 1)

            training_args = {
                "model": model,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "pbar": progress_bar,
                "graph": graph,
                "device": DEVICE,
                "distributed": (world_size is not None),
                "rank": rank,
                "patience": 5,
                "factorize": use_factorize,
                "writer": writer if is_main_process else None,
                "tensorboard_prefix": model_name,
                "num_visualization_samples": args.num_visualization_samples,
                "gradient_clip": args.gradient_clip,
                "validate_every": args.val_every,
                "visualize_every": args.vis_every,
                "use_mixed_precision": args.mixed_precision,
            }

            if log and is_main_process:
                training_args.update(
                    {
                        "results_path": results_path,
                        "experiment_results": experiment_results,
                        "data_size_key": ds_key,
                        "run_entry": run_entry,
                    }
                )

            trained_model, losses = train(**training_args)

            # Save checkpoint
            if is_main_process:  # Only save on main process
                Path("checkpoints").mkdir(exist_ok=True)
                model_state = (
                    trained_model.module.state_dict()
                    if isinstance(trained_model, DDP)
                    else trained_model.state_dict()
                )
                
                # Extract model configuration
                if hasattr(trained_model, "module"):
                    model_instance = trained_model.module
                else:
                    model_instance = trained_model
                
                # Create a model config dictionary with all relevant parameters
                model_config = {
                    "d_model": getattr(model_instance, "embedding_dim", None),
                    "depth": getattr(model_instance, "depth", None),
                    "n_heads": getattr(model_instance, "n_heads", None),
                    "d_ff_mult": getattr(model_instance, "d_ff_mult", None),
                    "use_factorized": getattr(model_instance, "use_factorized", use_factorize),
                    "num_params": getattr(model_instance, "num_params", None),
                    "architecture": getattr(model_instance, "name", None),
                }
                
                torch.save(
                    {
                        "model_state_dict": model_state,
                        "losses": losses,
                        "model_config": model_config,
                        "batch_size": batch_size,
                        "lr": lr,
                    },
                    checkpoint_path,
                )

            if is_main_process:
                progress_bar.close()

    writer.close()
    print(
        f"\nTraining completed. {'Results continuously saved to ' + str(results_path) if log else 'No experiment log was written.'}"
    )

    # Add barrier before cleanup
    if world_size is not None:
        dist.barrier()
        cleanup_ddp()


if __name__ == "__main__":
    args = get_args()
    if args.distributed:
        # Need to use spawn method for CUDA runtime initialization
        mp.set_start_method("spawn")
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        main()
