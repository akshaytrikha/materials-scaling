# External
import torch
import torch.optim as optim
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
from data_utils import download_dataset, VALID_DATASETS
from arg_parser import get_args
from models.fcn import MetaFCNModels
from models.transformer_models import MetaTransformerModels
from models.schnet import MetaSchNetModels
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
    args = get_args()
    log = not args.no_log
    global DEVICE

    # DDP setup if world_size is provided
    if world_size is not None:
        setup_ddp(rank, world_size)
        DEVICE = torch.device(f"cuda:{rank}")

    # Convinience for running all datasets
    if args.datasets[0] == "all":
        args.datasets = VALID_DATASETS

    # Download datasets if not present
    dataset_paths = []
    for dataset_name in args.datasets:
        dataset_path = Path(f"datasets/{args.split_name}/{dataset_name}")
        if not dataset_path.exists():
            download_dataset(dataset_name, args.split_name)
        dataset_paths.append(dataset_path)

    # User Hyperparam Feedback
    params = vars(args) | {
        "dataset_split": args.split_name,
        "dataset_paths": dataset_paths,
    }
    pprint.pprint(params)
    print()

    batch_size = args.batch_size[0]
    lr = args.lr[0]
    num_epochs = args.epochs
    use_factorize = args.factorize
    graph = args.architecture == "SchNet"

    # Initialize meta model class based on architecture choice
    if args.architecture == "FCN":
        meta_models = MetaFCNModels(
            vocab_size=args.n_elements, use_factorized=use_factorize
        )
    elif args.architecture == "Transformer":
        if args.split_name == "train":
            max_n_atoms = 236
        elif args.split_name == "val":
            max_n_atoms = 168

        meta_models = MetaTransformerModels(
            vocab_size=args.n_elements,
            max_seq_len=max_n_atoms,
            concatenated=True,
            use_factorized=use_factorize,
        )
    elif args.architecture == "SchNet":
        if DEVICE == torch.device("mps"):
            print("MPS is not supported for SchNet. Switching to CPU.")
            DEVICE = torch.device("cpu")
        meta_models = MetaSchNetModels(device=DEVICE)

    # Create results path and initialize file if logging is enabled
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logdir = os.path.join("runs", f"exp_{timestamp}")
    writer = SummaryWriter(log_dir=tb_logdir)
    print(f"TensorBoard logs will be saved to: {tb_logdir}")

    results_path = Path("results") / f"experiments_{timestamp}.json"
    experiment_results = {}
    if log:
        Path("results").mkdir(exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({}, f)  # Initialize as empty JSON
        print(f"\nLogging enabled. Results will be saved to {results_path}")
    else:
        print("\nLogging disabled. No experiment log will be saved.")

    is_main_process = rank == 0 if world_size is not None else True
    for data_fraction in args.data_fractions:
        train_loader, val_loader = get_dataloaders(
            dataset_paths,
            train_data_fraction=data_fraction,
            batch_size=batch_size,
            seed=SEED,
            batch_padded=False,
            val_data_fraction=args.val_data_fraction,
            train_workers=args.train_workers,
            val_workers=args.val_workers,
            graph=graph,
            factorize=use_factorize,
        )
        dataset_size = len(train_loader.dataset)
        if is_main_process:
            print(
                f"\nTraining on dataset fraction {data_fraction} with {dataset_size} samples"
            )
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

            lambda_schedule = lambda epoch: 0.5 * (
                1 + math.cos(math.pi * epoch / num_epochs)
            )
            scheduler = LambdaLR(optimizer, lr_lambda=lambda_schedule)

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

            trained_model, losses = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                pbar=progress_bar,
                graph=graph,
                device=DEVICE,
                distributed=world_size is not None,
                rank=rank,
                patience=50,
                factorize=use_factorize,
                results_path=results_path if log and is_main_process else None,
                experiment_results=(
                    experiment_results if log and is_main_process else None
                ),
                data_size_key=ds_key if log and is_main_process else None,
                run_entry=run_entry if log and is_main_process else None,
                writer=writer if is_main_process else None,
                tensorboard_prefix=model_name,
                num_visualization_samples=args.num_visualization_samples,
                gradient_clip=args.gradient_clip,
                validate_every=args.val_every,
                visualize_every=args.vis_every,
            )

            # Save checkpoint
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(
                {
                    "model_state_dict": trained_model.state_dict(),
                    "losses": losses,
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

    if world_size is not None:
        cleanup_ddp()

    if log:
        # Generate inference GIFs at different training stages
        subprocess.run(
            [
                "python3",
                "model_prediction_evolution.py",
                str(results_path),
                "--split",
                "train",
            ]
        )


if __name__ == "__main__":
    import torch.multiprocessing as mp

    args = get_args()
    if args.distributed:
        # Need to use spawn method for CUDA runtime initialization
        mp.set_start_method("spawn")
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        main()
