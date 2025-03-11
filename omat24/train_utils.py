# External
import copy
import torch
import torch.nn as nn
from typing import Union
from torch.utils.flop_counter import FlopCounterMode
from contextlib import nullcontext
import torch.distributed as dist
from torch.amp import autocast, GradScaler
import inspect
import math
from bisect import bisect
from fairchem.core.common.typing import assert_is_instance as aii

# Internal
from loss import compute_loss
from log_utils import partial_json_log, log_tb_metrics
from torch_geometric.data import Batch


def reduce_losses(tensor, average=True):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= dist.get_world_size()
    return rt


def forward_pass(
    model: nn.Module,
    batch: Union[dict, Batch],
    graph: bool,
    training: bool,
    device: torch.device,
    factorize: bool,
):
    """A common forward pass function for inference across different architectures & dataloaders.

    Args:
        model (nn.Module): PyTorch model to use.
        batch (Union[Dict, Batch]): The batch of data to forward pass.
        graph (bool): Whether the model is a graph-based model.
        training (bool): Whether the model is in training mode.
        device (torch.device): The device to run the model on.
        factorize (bool): Whether to factorize the distance matrix.
    """
    if training or graph:
        context_manager = torch.enable_grad()
    else:
        context_manager = torch.no_grad()

    # Get the model name, handling both regular models and DDP-wrapped models
    model_name = (
        model.name
        if hasattr(model, "name")
        else model.module.name if hasattr(model, "module") else None
    )

    with context_manager:
        if type(batch) == dict:

            atomic_numbers = batch["atomic_numbers"].to(device, non_blocking=True)
            positions = batch["positions"].to(device, non_blocking=True)
            true_forces = batch["forces"].to(device, non_blocking=True)
            true_energy = batch["energy"].to(device, non_blocking=True)
            true_stress = batch["stress"].to(device, non_blocking=True)
            mask = atomic_numbers != 0
            natoms = mask.sum(dim=1).to(device)

            if factorize:
                factorized_distances = batch["factorized_matrix"].to(
                    device, non_blocking=True
                )
                pred_forces, pred_energy, pred_stress = model(
                    atomic_numbers, positions, factorized_distances, mask
                )
            else:
                distance_matrix = batch["distance_matrix"].to(device, non_blocking=True)
                pred_forces, pred_energy, pred_stress = model(
                    atomic_numbers, positions, distance_matrix, mask
                )

        elif isinstance(batch, Batch):
            # PyG Batch
            atomic_numbers = batch.atomic_numbers.to(device, non_blocking=True)
            positions = batch.pos.to(device, non_blocking=True)
            true_forces = batch.forces.to(device, non_blocking=True)
            true_energy = batch.energy.to(device, non_blocking=True)
            true_stress = batch.stress.to(device, non_blocking=True)
            mask = None
            if hasattr(batch, "natoms"):
                natoms = (
                    batch.natoms.to(device)
                    if hasattr(batch.natoms, "to")
                    else torch.tensor(batch.natoms, device=device)
                )
            else:
                natoms = None

            if model_name == "SchNet":
                edge_index = batch.edge_index.to(device, non_blocking=True)
                structure_index = batch.batch.to(device, non_blocking=True)

                pred_forces, pred_energy, pred_stress = model(
                    atomic_numbers,
                    positions,
                    edge_index,
                    structure_index,
                )
            elif model_name == "EquiformerV2":
                # equiformer constructs graphs internally
                batch = batch.to(device)
                pred_forces, pred_energy, pred_stress = model(batch)

    return (
        pred_forces,
        pred_energy,
        pred_stress,
        true_forces,
        true_energy,
        true_stress,
        mask,
        natoms,
    )


def get_amp_context(use_mixed_precision: bool, device_type: str) -> tuple:
    """Creates appropriate autocast context and scaler for mixed precision training.
    
    Args:
        use_mixed_precision (bool): Whether to use mixed precision.
        device_type (str): Device type ('cuda', 'cpu', etc.)
        
    Returns:
        tuple: (autocast_fn, scaler) - context manager and scaler for mixed precision
    """
    if use_mixed_precision and device_type == 'cuda':
        return autocast(device_type=device_type), GradScaler()
    else:
        # Create dummy objects if not using mixed precision
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            
        class DummyScaler:
            def scale(self, loss): return loss
            def step(self, opt): opt.step()
            def update(self): pass
            def unscale_(self, opt): pass
            
        return DummyContext(), DummyScaler()


def collect_samples_helper(num_visualization_samples, dataset, model, graph, device):
    samples = []
    for i in range(min(num_visualization_samples, len(dataset))):
        if graph:
            batch = Batch.from_data_list([dataset[i]])
            positions = batch["pos"]
            atomic_numbers = batch["atomic_numbers"]
            sample_length = len(atomic_numbers)
            idx = batch["idx"].cpu().tolist()[0]
        else:
            batch = {
                k: torch.unsqueeze(v, 0) if isinstance(v, torch.Tensor) else v
                for k, v in dataset[i].items()
            }
            # Extract data from batch and sqeeuze batch dimension
            positions = batch["positions"].squeeze(0)
            atomic_numbers = batch["atomic_numbers"].squeeze(0)
            sample_length = (batch["atomic_numbers"] != 0).sum(dim=1)[0].item()
            idx = batch["idx"]

        symbols = batch["symbols"]

        (
            pred_forces,
            pred_energy,
            pred_stress,
            true_forces,
            true_energy,
            true_stress,
            _,
            _,
        ) = forward_pass(model, batch, graph, False, device, False)

        if not graph:
            pred_forces = pred_forces.squeeze(0)
            true_forces = true_forces.squeeze(0)
            pred_stress = pred_stress.squeeze(0)
            true_stress = true_stress.squeeze(0)

        samples.append(
            {
                "idx": idx,
                "symbols": symbols,
                "atomic_numbers": atomic_numbers[:sample_length].cpu().tolist(),
                "positions": positions[:sample_length].cpu().tolist(),
                "true": {
                    "forces": true_forces[:, :sample_length].cpu().tolist(),
                    "energy": true_energy.cpu().tolist()[0],
                    "stress": true_stress.cpu().tolist(),
                },
                "pred": {
                    "forces": pred_forces[:, :sample_length].cpu().tolist(),
                    "energy": pred_energy.cpu().tolist()[0],
                    "stress": pred_stress.cpu().tolist(),
                },
            }
        )
    return samples


def collect_samples_for_visualizing(
    model, graph, train_loader, val_loader, device, num_visualization_samples
):
    """Collect samples and predictions from both training and validation sets.
    Uses fixed indices for consistent visualization.

    Args:
        model (torch.nn.Module): Trained model.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        device (torch.device): Device to run model on.
        num_visualization_samples (int): Number of samples to visualize.

    Returns:
        dict: Dictionary containing samples and predictions for training and validation sets.
    """
    # Get the underlying dataset from the DataLoader
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    return {
        "train": collect_samples_helper(
            num_visualization_samples, train_dataset, model, graph, device
        ),
        "val": collect_samples_helper(
            num_visualization_samples, val_dataset, model, graph, device
        ),
    }


def run_validation(model, val_loader, graph, device, factorize=False, use_mixed_precision=False):
    """
    Run validation on the validation set and return the average validation loss.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): The validation data loader.
        graph (bool): Whether the model is a graph-based model.
        device (torch.device): The device to run validation on.
        factorize (bool, optional): Whether to factorize. Defaults to False.
        use_mixed_precision (bool, optional): Whether to use mixed precision. Defaults to False.

    Returns:
        tuple: The average validation loss components.
    """
    amp_context, _ = get_amp_context(use_mixed_precision, device.type)
    
    model.eval()
    val_loss_sum = 0.0
    energy_loss_sum = 0.0
    force_loss_sum = 0.0
    stress_iso_loss_sum = 0.0
    stress_aniso_loss_sum = 0.0
    n = len(val_loader)

    for batch in val_loader:
        with torch.no_grad(), amp_context:
            (
                pred_forces,
                pred_energy,
                pred_stress,
                true_forces,
                true_energy,
                true_stress,
                mask,
                natoms,
            ) = forward_pass(
                model=model,
                batch=batch,
                graph=graph,
                training=False,
                device=device,
                factorize=factorize,
            )

            # Mapping atoms to their respective structures (for graphs)
            structure_index = (
                batch.batch if graph and hasattr(batch, "batch") else []
            )
            val_loss_dict = compute_loss(
                pred_forces,
                pred_energy,
                pred_stress,
                true_forces,
                true_energy,
                true_stress,
                mask,
                device,
                natoms,
                graph,
                structure_index,
            )
            
            val_loss_sum += val_loss_dict["total_loss"].item()
            energy_loss_sum += val_loss_dict["energy_loss"].item()
            force_loss_sum += val_loss_dict["force_loss"].item()
            stress_iso_loss_sum += val_loss_dict["stress_iso_loss"].item()
            stress_aniso_loss_sum += val_loss_dict["stress_aniso_loss"].item()

    if n == 0:
        return float("inf")
    return (
        val_loss_sum / n,
        energy_loss_sum / n,
        force_loss_sum / n,
        stress_iso_loss_sum / n,
        stress_aniso_loss_sum / n,
    )


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    pbar,
    graph,
    device,
    distributed=False,
    rank=0,
    patience=5,
    factorize=False,
    results_path=None,
    experiment_results=None,
    data_size_key=None,
    run_entry=None,
    writer=None,
    tensorboard_prefix="model",
    num_visualization_samples=3,
    gradient_clip=1,
    validate_every=500,
    visualize_every=500,
    use_mixed_precision=False,
):
    """
    Train model with validation at epoch 0 and every 'validate_every' epochs.
    Includes early stopping and optional JSON + TensorBoard logging.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler, or None.
        pbar (tqdm): A tqdm progress bar initialized with the total number of epochs.
        device (torch.device): The device to run training on.
        distributed (bool, optional): Whether to use distributed training.
        rank (int, optional): The rank of the process.
        patience (int): Early stopping patience (number of checks with no improvement).
        results_path (str, optional): Path to JSON results file. If provided, partial logs are written.
        experiment_results (dict, optional): Dict for storing experiment results.
        data_size_key (str, optional): Key to label experiment_results by dataset size.
        run_entry (dict, optional): Dictionary describing the current run (e.g., model_name, config).
        writer (SummaryWriter, optional): TensorBoard writer for logging.
        tensorboard_prefix (str, optional): Prefix for naming logs in TensorBoard.
        num_visualization_samples (int, optional): Number of samples to visualize in logs.
        gradient_clip (int, optional): Gradient clipping value.
        validate_every (int, optional): Frequency (in epochs) to run validation.
        visualize_every (int, optional): Frequency (in epochs) to collect visualization samples.
        use_mixed_precision (bool, optional): Whether to use mixed precision training. Defaults to False.

    Returns:
        (nn.Module, dict): The trained model and a dictionary of recorded losses.
    """
    # Set up mixed precision
    amp_context, scaler = get_amp_context(use_mixed_precision, device.type)

    model.to(device)
    can_write_partial = all(
        [results_path, experiment_results, data_size_key, run_entry]
    )
    losses = {}

    is_main_process = not distributed or rank == 0
    n_train_batches = len(train_loader)
    total_epochs = len(pbar) if is_main_process else pbar[-1] + 1

    # Initial validation at epoch 0
    (
        val_loss,
        val_energy_loss,
        val_force_loss,
        val_stress_iso_loss,
        val_stress_aniso_loss,
    ) = run_validation(model, val_loader, graph, device, factorize, use_mixed_precision)
    losses[0] = {"val_loss": float(val_loss)}
    if writer is not None:
        # Logging each metric individually using log_tb_metrics
        log_tb_metrics(
            {
                "": val_loss,
                "energy": val_energy_loss,
                "force": val_force_loss,
                "stress_iso": val_stress_iso_loss,
                "stress_aniso": val_stress_aniso_loss,
            },
            writer,
            0,
            tensorboard_prefix,
            train=False,
        )

    # Write partial JSON if everything is provided
    if can_write_partial:
        partial_json_log(
            experiment_results,
            data_size_key,
            run_entry,
            0,
            float("nan"),
            val_loss,
            results_path,
        )

    # Early stopping setup
    best_val_loss = val_loss
    best_val_model = copy.deepcopy(model)
    best_val_loss_dict = copy.deepcopy(losses)
    epochs_since_improvement = 0

    samples = None
    n_train_batches = len(train_loader)

    # Training loop
    flop_counter = FlopCounterMode(display=False)
    flops_per_epoch = 0
    for epoch in range(1, total_epochs):
        if (
            distributed
            and hasattr(train_loader, "sampler")
            and hasattr(train_loader.sampler, "set_epoch")
        ):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = 0.0
        energy_loss_sum = 0.0
        force_loss_sum = 0.0
        stress_iso_loss_sum = 0.0
        stress_aniso_loss_sum = 0.0

        context = flop_counter if epoch == 1 else nullcontext()
        with context:
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Use autocast for mixed precision during forward pass
                with amp_context:
                    (
                        pred_forces,
                        pred_energy,
                        pred_stress,
                        true_forces,
                        true_energy,
                        true_stress,
                        mask,
                        natoms,
                    ) = forward_pass(
                        model=model,
                        batch=batch,
                        graph=graph,
                        training=True,
                        device=device,
                        factorize=factorize,
                    )
                    
                    # Mapping atoms to their respective structures (for graphs)
                    structure_index = (
                        batch.batch if graph and hasattr(batch, "batch") else []
                    )
                    train_loss_dict = compute_loss(
                        pred_forces,
                        pred_energy,
                        pred_stress,
                        true_forces,
                        true_energy,
                        true_stress,
                        mask,
                        device,
                        natoms,
                        graph,
                        structure_index,
                    )
                    total_train_loss = train_loss_dict["total_loss"]
                
                # Use scaler for backward pass
                scaler.scale(total_train_loss).backward()

                if use_mixed_precision:
                    scaler.unscale_(optimizer)

                if distributed:
                    # Synchronize gradients across processes
                    for param in model.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= dist.get_world_size()

                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                # Update with scaler
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += total_train_loss.item()
                energy_loss_sum += train_loss_dict["energy_loss"].item()
                force_loss_sum += train_loss_dict["force_loss"].item()
                stress_iso_loss_sum += train_loss_dict["stress_iso_loss"].item()
                stress_aniso_loss_sum += train_loss_dict["stress_aniso_loss"].item()
                current_avg_loss = train_loss_sum / (batch_idx + 1)

                if is_main_process:
                    pbar.set_description(
                        f"train_loss={current_avg_loss:.2f} val_loss={val_loss:.2f}"
                    )
        if epoch == 1:
            flops_per_epoch = flop_counter.get_total_flops()

        # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step()

        if distributed:
            train_loss_tensor = torch.tensor(train_loss_sum, device=device)
            energy_loss_tensor = torch.tensor(energy_loss_sum, device=device)
            force_loss_tensor = torch.tensor(force_loss_sum, device=device)
            stress_iso_loss_tensor = torch.tensor(stress_iso_loss_sum, device=device)
            stress_aniso_loss_tensor = torch.tensor(
                stress_aniso_loss_sum, device=device
            )

            train_loss_sum = reduce_losses(train_loss_tensor, average=True).item()
            energy_loss_sum = reduce_losses(energy_loss_tensor, average=True).item()
            force_loss_sum = reduce_losses(force_loss_tensor, average=True).item()
            stress_iso_loss_sum = reduce_losses(
                stress_iso_loss_tensor, average=True
            ).item()
            stress_aniso_loss_sum = reduce_losses(
                stress_aniso_loss_tensor, average=True
            ).item()

        avg_epoch_train_loss = train_loss_sum / n_train_batches
        avg_epoch_energy_loss = energy_loss_sum / n_train_batches
        avg_epoch_force_loss = force_loss_sum / n_train_batches
        avg_epoch_stress_iso_loss = stress_iso_loss_sum / n_train_batches
        avg_epoch_stress_aniso_loss = stress_aniso_loss_sum / n_train_batches

        losses[epoch] = {"train_loss": float(avg_epoch_train_loss)}
        # TensorBoard logging for training loss
        if writer is not None:
            # Log parameter norms (example usage) using log_tb_metrics
            log_tb_metrics(
                {
                    "": avg_epoch_train_loss,
                    "energy": avg_epoch_energy_loss,
                    "force": avg_epoch_force_loss,
                    "stress_iso": avg_epoch_stress_iso_loss,
                    "stress_aniso": avg_epoch_stress_aniso_loss,
                },
                writer,
                epoch,
                tensorboard_prefix,
                train=True,
            )
            # Simple gradient logging for debugging (skip bias layers)
            for name, param in model.named_parameters():
                if (
                    param is not None
                    and param.requires_grad
                    and param.grad is not None
                    and not name.endswith("bias")
                ):  # Skip bias layers
                    # Log mean gradient - key indicator for vanishing/exploding gradients
                    grad_mean = param.grad.abs().mean().item()
                    writer.add_scalar(
                        f"{tensorboard_prefix}/Grads/{name}",
                        grad_mean,
                        global_step=epoch,
                    )

                    # Log gradient-to-weight ratio - indicates if updates are well-scaled
                    grad_to_weight = (
                        param.grad.abs().mean() / (param.data.abs().mean() + 1e-8)
                    ).item()
                    writer.add_scalar(
                        f"{tensorboard_prefix}/G2W/{name}",
                        grad_to_weight,
                        global_step=epoch,
                    )
        # Validate every 'validate_every' epochs
        if epoch % validate_every == 0:
            (
                val_loss,
                val_energy_loss,
                val_force_loss,
                val_stress_iso_loss,
                val_stress_aniso_loss,
            ) = run_validation(model, val_loader, graph, device, factorize, use_mixed_precision)

            if distributed:
                val_loss_tensor = torch.tensor(val_loss, device=device)
                val_energy_loss_tensor = torch.tensor(val_energy_loss, device=device)
                val_force_loss_tensor = torch.tensor(val_force_loss, device=device)
                val_stress_iso_loss_tensor = torch.tensor(
                    val_stress_iso_loss, device=device
                )
                val_stress_aniso_loss_tensor = torch.tensor(
                    val_stress_aniso_loss, device=device
                )

                val_loss = reduce_losses(val_loss_tensor, average=True).item()
                val_energy_loss = reduce_losses(
                    val_energy_loss_tensor, average=True
                ).item()
                val_force_loss = reduce_losses(
                    val_force_loss_tensor, average=True
                ).item()
                val_stress_iso_loss = reduce_losses(
                    val_stress_iso_loss_tensor, average=True
                ).item()
                val_stress_aniso_loss = reduce_losses(
                    val_stress_aniso_loss_tensor, average=True
                ).item()

            if writer is not None:
                log_tb_metrics(
                    {
                        "": val_loss,
                        "energy": val_energy_loss,
                        "force": val_force_loss,
                        "stress_iso": val_stress_iso_loss,
                        "stress_aniso": val_stress_aniso_loss,
                    },
                    writer,
                    epoch,
                    tensorboard_prefix,
                    train=False,
                )
            losses[epoch]["val_loss"] = val_loss

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
                best_val_model = copy.deepcopy(model)
                best_val_loss_dict = copy.deepcopy(losses)
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    return best_val_model, best_val_loss_dict

        # Visualization samples every 'visualize_every' epochs
        if epoch % visualize_every == 0:
            samples = collect_samples_for_visualizing(
                model,
                graph,
                train_loader,
                val_loader,
                device,
                num_visualization_samples,
            )

        # Write partial JSON if requested
        if can_write_partial:
            partial_json_log(
                experiment_results,
                data_size_key,
                run_entry,
                epoch,
                avg_epoch_train_loss,
                val_loss if epoch % validate_every == 0 else float("nan"),
                results_path,
                samples if epoch % visualize_every == 0 else None,
                flops_per_epoch * epoch,
            )

        if is_main_process:
            pbar.update(1)

    return model, losses


class CosineLRLambda:
    def __init__(self, scheduler_params) -> None:
        self.warmup_epochs = aii(scheduler_params["warmup_epochs"], int)
        self.lr_warmup_factor = aii(scheduler_params["warmup_factor"], float)
        self.max_epochs = aii(scheduler_params["epochs"], int)
        self.lr_min_factor = aii(scheduler_params["lr_min_factor"], float)

    def __call__(self, current_step: int):
        # `warmup_epochs` is already multiplied with the num of iterations
        print(f"current_step is {current_step}")
        print(f"self.warmup_epochs is {self.warmup_epochs}")
        if current_step <= self.warmup_epochs:
            alpha = current_step / float(self.warmup_epochs)
            return self.lr_warmup_factor * (1.0 - alpha) + alpha
        else:
            if current_step >= self.max_epochs:
                return self.lr_min_factor
            return self.lr_min_factor + 0.5 * (1 - self.lr_min_factor) * (
                1 + math.cos(math.pi * (current_step / self.max_epochs))
            )


class LRScheduler:
    """
    Notes:
        1. scheduler.step() is called for every step for OC20 training.
        2. We use "scheduler_params" in .yml to specify scheduler parameters.
        3. For cosine learning rate, we use LambdaLR with lambda function being cosine:
            scheduler: LambdaLR
            scheduler_params:
                lambda_type: cosine
                ...
        4. Following 3., if `cosine` is used, `scheduler_params` in .yml looks like:
            scheduler: LambdaLR
            scheduler_params:
                lambda_type: cosine
                warmup_epochs: ...
                warmup_factor: ...
                lr_min_factor: ...

    Args:
        optimizer (obj): torch optim object
        config (dict): Optim dict from the input config
    """

    def __init__(self, optimizer, scheduler_params) -> None:
        self.optimizer = optimizer
        print(scheduler_params)
        self.scheduler_type = aii(scheduler_params["scheduler_type"], str)
        self.scheduler_params = scheduler_params.copy()

        # Use `LambdaLR` for multi-step and cosine learning rate
        if self.scheduler_type == "LambdaLR":
            scheduler_lambda_fn = None
            self.lambda_type = self.scheduler_params["lambda_type"]
            if self.lambda_type == "cosine":
                scheduler_lambda_fn = CosineLRLambda(self.scheduler_params)
            else:
                raise ValueError
            self.scheduler_params["lr_lambda"] = scheduler_lambda_fn

        if self.scheduler_type != "Null":
            self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_type)
            scheduler_args = self.filter_kwargs(self.scheduler_params)
            self.scheduler = self.scheduler(optimizer, **scheduler_args)

    def step(self, metrics=None, epoch=None):
        if self.scheduler_type == "Null":
            return
        if self.scheduler_type == "ReduceLROnPlateau":
            if metrics is None:
                raise Exception("Validation set required for ReduceLROnPlateau.")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()

    def filter_kwargs(self, config):
        # adapted from https://stackoverflow.com/questions/26515595/
        sig = inspect.signature(self.scheduler)
        filter_keys = [
            param.name
            for param in sig.parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        filter_keys.remove("optimizer")
        return {arg: config[arg] for arg in config if arg in filter_keys}

    def get_lr(self) -> float | None:
        for group in self.optimizer.param_groups:
            return aii(group["lr"], float)
        return None