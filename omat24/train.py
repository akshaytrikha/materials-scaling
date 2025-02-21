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
from data import OMat24Dataset, get_dataloaders
from data_utils import download_dataset
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

    # DDP setup if world_size is provided
    if world_size is not None:
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download datasets if not present
    dataset_paths = []
    for dataset_name in args.datasets:
        dataset_path = Path(f"datasets/{args.split_name}/{dataset_name}")
        if not dataset_path.exists():
            download_dataset(dataset_name, args.split_name)
        dataset_paths += [dataset_path]
    # Load dataset
    graph = args.architecture == "SchNet"
    dataset = OMat24Dataset(
        dataset_paths=dataset_paths, augment=args.augment, graph=graph
    )

    # User Hyperparam Feedback
    params = vars(args) | {
        "dataset_split": args.split_name,
        "max_n_atoms": dataset.max_n_atoms,
    }
    pprint.pprint(params)
    print()

    batch_size = args.batch_size[0]
    lr = args.lr[0]
    num_epochs = args.epochs
    use_factorize = args.factorize

    # Initialize meta model class based on architecture choice
    if args.architecture == "FCN":
        meta_models = MetaFCNModels(
            vocab_size=args.n_elements, use_factorized=use_factorize
        )
    elif args.architecture == "Transformer":
        meta_models = MetaTransformerModels(
            vocab_size=args.n_elements,
            max_seq_len=dataset.max_n_atoms,
            concatenated=True,
            use_factorized=use_factorize,
        )
    elif args.architecture == "SchNet":
        if DEVICE == torch.device("mps"):
            print("MPS is not supported for SchNet. Switching to CPU.")
            DEVICE = torch.device("cpu")

    # Create results path and initialize file if logging is enabled
    experiment_results = {}

    # For TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logdir = os.path.join("runs", f"exp_{timestamp}")
    writer = SummaryWriter(log_dir=tb_logdir)
    print(f"TensorBoard logs will be saved to: {tb_logdir}")

    results_path = Path("results") / f"experiments_{timestamp}.json"
    if log:
        Path("results").mkdir(exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({}, f)  # Initialize as empty JSON
        print(f"\nLogging enabled. Results will be saved to {results_path}")
    else:
        print("\nLogging disabled. No experiment log will be saved.")

    # Get dataloaders with DDP support if needed
    train_loader, val_loader = get_dataloaders(
        dataset=dataset,
        train_data_fraction=args.data_fractions[0],
        batch_size=batch_size,
        seed=SEED,
        batch_padded=False,
        val_data_fraction=args.val_data_fraction,
        train_workers=args.train_workers,
        val_workers=args.val_workers,
        graph=graph,
        distributed=(
            world_size is not None
        ),  # Enable DDP dataloading if using multiple GPUs
    )

    # Only show progress bar on main process
    is_main_process = rank == 0 if world_size is not None else True
    
    for data_fraction in args.data_fractions:
        dataset_size = len(train_loader.dataset)
        if is_main_process:
            print(f"\nTraining on dataset fraction {data_fraction} with {dataset_size} samples")
        
        for model_idx, model in enumerate(meta_models):
            if log and is_main_process:
                print(
                    f"\nModel {model_idx + 1}/{len(meta_models)} is on device {device} "
                    f"with batch size {args.batch_size[model_idx]} "
                    f"and learning rate {args.lr[model_idx]}"
                )

            # Move model to device before wrapping with DDP
            model = model.to(device)

            # Store original model attributes before DDP wrapping
            num_params = model.num_params if hasattr(model, "num_params") else None
            embedding_dim = getattr(model, "embedding_dim", None)
            depth = getattr(model, "depth", None)

            # Create optimizer before DDP wrapper
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr[model_idx])

            # Wrap model in DDP if using multiple GPUs
            if world_size is not None:
                model = DDP(
                    model,
                    device_ids=[rank],
                    find_unused_parameters=True,
                    broadcast_buffers=False,
                )

            # Create checkpoint path using stored num_params
            checkpoint_path = f"checkpoints/{args.architecture}_ds{dataset_size}_p{int(num_params)}_{timestamp}.pth"

            # Use the stored num_params for model_name
            model_name = f"model_ds{dataset_size}_p{int(num_params)}"

            # Prepare run entry using stored attributes
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
                progress_bar = range(num_epochs + 1)  # Just a range object for non-main processes
            
            # Train
            trained_model, losses = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=None,
                pbar=progress_bar,
                graph=graph,
                device=device,
                distributed=(world_size is not None),
                rank=rank,
                results_path=results_path if log else None,
                experiment_results=experiment_results,
                data_size_key=ds_key if log else None,
                run_entry=run_entry if log else None,
                writer=writer,
                tensorboard_prefix=model_name,
                num_visualization_samples=args.num_visualization_samples,
                gradient_clip=args.gradient_clip,
            )

            # --- Save checkpoint ---
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

    # Close the SummaryWriter
    writer.close()

    print(
        f"\nTraining completed. {'Results continuously saved to ' + str(results_path) if log else 'No experiment log was written.'}"
    )

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

    if world_size is not None:
        cleanup_ddp()


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
