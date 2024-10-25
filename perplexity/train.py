# External
import torch.multiprocessing as mp
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pprint
from tqdm.auto import tqdm
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_scheduler

# Internal
from data import setup_dataset, get_dataloaders
from model import *
from ddp import cleanup_ddp, setup_ddp
from train_utils import train_epoch, load_checkpoint
from arg_parser import get_args
from log_utils import log_training_metrics, update_plots
from torch.utils.data import DistributedSampler


def train(args, device, is_master_process=True, device_ids=None):
    # Setup Dataset
    if args.dataset_version == "small":
        dataset_name = "wikitext-2-raw-v1"
    elif args.dataset_version == "large":
        dataset_name = "wikitext-103-raw-v1"
    dataset, tokenizer = setup_dataset(dataset_name)

    # Models, Loss
    if args.architecture == "FCN":
        models = MetaFullyConnectedModels(vocab_size=len(tokenizer))
    elif args.architecture == "VanillaTransformer":
        models = MetaXTransformers(vocab_size=len(tokenizer))
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # User Hyperparam Feedback
    if is_master_process:
        pprint.pprint(vars(args))
        print()

    # Group runs
    timestamp = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
    if args.checkpoint_group_name is not None:
        group_name = args.checkpoint_group_name
    else:
        group_name = f"{dataset_name}_{args.architecture}_ts={timestamp}"  # for wandb, checkpoints

    for data_fraction in tqdm(
        args.data_fractions, desc="Data Iteration", disable=not is_master_process
    ):
        # Create a subset of the dataset
        train_loader, val_loader = get_dataloaders(
            dataset, data_fraction, args.batch_size, args.ddp
        )

        for model in models:
            model.to(device)
            if args.ddp:
                model = DDP(model, device_ids=device_ids)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"\nModel is on device {device} and has {num_params} parameters")
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            total_steps = args.num_epochs * len(train_loader)
            num_warmup_steps = int(0.1 * total_steps)
            scheduler = get_scheduler(
                name="cosine",
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
            )

            # Define model name
            if args.architecture == "FCN":
                model_name = f"{args.architecture}_dv={args.dataset_version}_df={data_fraction}_p={num_params}"
            else:
                model_name = f"{args.architecture}_dv={args.dataset_version}_df={data_fraction}_p={num_params}"

            # Define checkpoint path
            checkpoint_dir = f"saved_models/{group_name}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pt")

            start_epoch = 0
            best_val_loss = float("inf")

            # Load from checkpoint if exists
            if is_master_process and os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}")
                start_epoch, best_val_loss = load_checkpoint(
                    model, optimizer, scheduler, checkpoint_path, device
                )
                print(
                    f"Resuming training from epoch {start_epoch} with best_val_loss={best_val_loss}"
                )

            if is_master_process and args.wandb_log:
                wandb.init(
                    project="wikitext-scaling",
                    name=model_name,
                    group=group_name,
                    config={
                        "learning_rate": args.lr,
                        "num_epochs": args.num_epochs,
                        "batch_size": args.batch_size,
                        "data_fraction": f"{int(data_fraction*100)}%",
                    },
                    resume="allow" if os.path.exists(checkpoint_path) else False,
                )

            # Train model
            for epoch in tqdm(
                range(start_epoch, args.num_epochs + 1),
                desc="Epoch Progress",
                leave=True,
                disable=not is_master_process,
            ):
                train_loss, val_loss = train_epoch(
                    model,
                    train_loader,
                    val_loader,
                    optimizer,
                    scheduler,
                    loss_fn,
                    device,
                )

                if is_master_process and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "epoch": epoch,
                            "best_val_loss": best_val_loss,
                        },
                        checkpoint_path,
                    )

                if is_master_process and epoch % 10 == 0:
                    log_training_metrics(
                        filename=f"{checkpoint_dir}/log_metrics.json",
                        data_fraction=data_fraction,
                        model_params=num_params,
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        best_val_loss=best_val_loss,
                        model_name=args.architecture,
                        batch_size=args.batch_size,
                        learning_rate=args.lr,
                    )

                    # Update plots for current model
                    update_plots(
                        metrics_file=f"{checkpoint_dir}/log_metrics.json",
                        plots_dir=f"{checkpoint_dir}/plots",
                        current_df=int(data_fraction * 100),
                    )

                # Wandb Logging
                best_val_perplexity = torch.exp(torch.tensor(best_val_loss)).item()
                if is_master_process and args.wandb_log:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_perplexity": best_val_perplexity,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "num_params": num_params,
                        }
                    )

            # Evaluate Perplexity after training
            if is_master_process:
                best_val_perplexity = torch.exp(torch.tensor(best_val_loss)).item()
                print(
                    f"Dataset Size: {int(data_fraction*100)}%, Val Perplexity: {best_val_perplexity}\n"
                )

            if is_master_process and args.wandb_log:
                wandb.finish()

        # Update plots for current data fraction
        if is_master_process:
            update_plots(
                metrics_file=f"{checkpoint_dir}/log_metrics.json",
                plots_dir=f"{checkpoint_dir}/plots",
                current_df=int(data_fraction * 100),
            )


def train_ddp(rank, world_size, master_addr, master_port, args):
    is_master_process = rank == 0
    device = torch.device(
        f"cuda:{rank % torch.cuda.device_count()}"
        if torch.cuda.is_available()
        else "cpu"
        # mps is not supported in ddp
    )
    device_ids = (
        [rank % torch.cuda.device_count()] if torch.cuda.is_available() else None
    )

    print(f"Running DDP training on rank {rank} using device {device}.")
    setup_ddp(rank, world_size, master_addr, master_port)
    train(args, device, is_master_process, device_ids)
    cleanup_ddp()


if __name__ == "__main__":
    args = get_args()

    if args.ddp:
        num_processes = torch.cuda.device_count() if torch.cuda.is_available() else 2
        world_size = int(os.environ.get("WORLD_SIZE", num_processes))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "12357")
        mp.spawn(
            train_ddp,
            args=(world_size, master_addr, master_port, args),
            nprocs=num_processes,
            join=True,
        )
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        train(args, device)
