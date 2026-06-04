import os
import re
import torch
import inspect
import argparse
import logging
import wandb

from tqdm import tqdm
from dataclasses import asdict

from metrics import calculate_similarity_metrics, identify_dead_neurons, orthogonal_decoder, cknna, explained_variance, compute_matryoshka_metrics
from utils import SAEDataset, set_seed, get_device, geometric_median, calculate_vector_mean, LinearDecayLR, CosineWarmupScheduler
from config import get_config
from sae import Autoencoder, MatryoshkaAutoencoder
from loss import SAELoss
from callbacks import CallbackRunner, TrainingCallback

"""
Sparse Autoencoder (SAE) Training Script

This script provides a complete pipeline for training various types of sparse autoencoder models,
including standard SAEs with different activation functions and Matryoshka SAEs with nested
feature hierarchies. It handles training, evaluation, and model saving with configurable
hyperparameters.
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the SAE training script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments with the following fields:
            - dataset_train: Path to the training dataset
            - dataset_test: Path to the testing/validation dataset
            - model: Model architecture to train (e.g., "ReLUSAE", "TopKSAE")
            - activation: Activation function to use
            - epochs: Number of training epochs
            - learning_rate: Initial learning rate
            - expansion_factor: Ratio of latent dimensions to input dimensions
    """
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder (SAE) models")
    parser.add_argument("-dt", "--dataset_train", type=str, required=True, 
                       help="Path to training dataset file (.npy)")
    parser.add_argument("-ds", "--dataset_test", type=str, required=True, 
                       help="Path to testing/validation dataset file (.npy)")
    parser.add_argument("-dm", "--dataset_second_modality", type=str, default=None,
                       help="Path to second modality dataset file (.npy) for eval.")
    parser.add_argument("-m", "--model", type=str, required=True,
                       choices=["ReLUSAE", "TopKSAE", "BatchTopKSAE", "MSAE_UW", "MSAE_RW", "MSAE_UW_L1", "MSAE_RW_L1"],
                       help="Model architecture to train")
    parser.add_argument("-a", "--activation", type=str, required=True, 
                       help="Activation function (e.g., 'ReLU_003', 'TopKReLU_64')")
    parser.add_argument("-e", "--epochs", type=int, default=100, 
                       help="Number of training epochs")
    parser.add_argument("-ef", "--expansion_factor", type=float, default=1.0, 
                       help="Ratio of latent dimensions to input dimensions")
    parser.add_argument("--out_dir", type=str, default="./",
                        help="Directory to write files")
    parser.add_argument("--best_model_path", type=str, default="./",
                        help="Path to save the best model based on validation loss")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Optional wandb run name")
    parser.add_argument("--wandb_tags", type=str, default=[], nargs="*",)
    parser.add_argument("--loss_func", type=str, default=None,)
    parser.add_argument("--max_nesting", type=int, default=0,
                        help="Max nesting level for MSAE (0 = n_latents)")
    parser.add_argument("--l1_granularity_threshold", type=int, default=0,
                        help="For MSAE_UW_L1/MSAE_RW_L1: apply L1 only to nesting levels >= this value (0 = disabled)")

    return parser.parse_args()


def eval(model, eval_loader, loss_fn, device, cfg):
    # Evaluation phase
    loss_all = 0.0
    recon_loss_all = 0.0
    sparse_loss_all = 0.0
    cknna_score_sparse_sum = 0.0
    cknna_score_all_sum = 0.0
    fvu_score_all_sum = 0.0
    fvu_score_sparse_sum = 0.0
    diagonal_cs_sparse_sum = 0.0
    diagonal_cs_all_sum = 0.0
    mae_distance_sparse_sum = 0.0
    mae_distance_all_sum = 0.0
    od_sum = 0.0
    sparsity_sparse_sum = 0.0
    sparsity_all_sum = 0.0
    
    # Switch to evaluation mode
    model.eval()
    for step, embeddings in enumerate(tqdm(eval_loader, desc="Evaluation")):
        embeddings = embeddings.to(device)
        
        # Forward pass without gradient computation
        with torch.no_grad():
            recons_sparse, repr_sparse, recons_all, repr_all = model(embeddings)
            loss, recon_loss, sparse_loss = loss_fn(recons_all, embeddings, repr_all)
        
        if cfg.model.use_matryoshka:
            recons_sparse = recons_sparse[0]
            repr_sparse = repr_sparse[0]
        
        # Accumulate loss metrics
        loss_all += loss.item()
        recon_loss_all += recon_loss.item()
        sparse_loss_all += sparse_loss.item()
        
        # Accumulate CKNNA scores
        cknna_score_sparse_sum += cknna(recons_sparse, embeddings)
        cknna_score_all_sum += cknna(recons_all, embeddings)
        
        # Accumulate similarity metrics
        fvu_score_sparse_sum += explained_variance(embeddings, recons_sparse)
        fvu_score_all_sum += explained_variance(embeddings, recons_all)
        distance_sparse = calculate_similarity_metrics(embeddings, recons_sparse)
        distance_all = calculate_similarity_metrics(embeddings, recons_all)
        diagonal_cs_sparse_sum += distance_sparse[0]
        mae_distance_sparse_sum += distance_sparse[1]
        diagonal_cs_all_sum += distance_all[0]
        mae_distance_all_sum += distance_all[1]
        
        # Accumulate orthogonality measure
        od_sum += orthogonal_decoder(model.decoder)
        
        # Accumulate sparsity measures
        sparsity_sparse_sum += (repr_sparse == 0.0).float().mean(axis=-1).mean()
        sparsity_all_sum += (repr_all == 0.0).float().mean(axis=-1).mean()
    
    # Log evaluation metrics (averaged over batches)
    # logger.info("Evaluation results:")
    # logger.info(f"  Loss: {loss_all / len(eval_loader):.6f}")
    # logger.info(f"  Reconstruction Loss: {recon_loss_all / len(eval_loader):.6f}")
    # logger.info(f"  Sparsity Loss: {sparse_loss_all / len(eval_loader):.6f}")
    # logger.info(f"  FVU Sparse: {fvu_score_sparse_sum / len(eval_loader):.4f}")
    # logger.info(f"  FVU All: {fvu_score_all_sum / len(eval_loader):.4f}")
    # logger.info(f"  CKNNA Sparse: {cknna_score_sparse_sum / len(eval_loader):.4f}")
    # logger.info(f"  CKNNA All: {cknna_score_all_sum / len(eval_loader):.4f}")
    # logger.info(f"  Cosine Similarity Sparse: {diagonal_cs_sparse_sum / len(eval_loader):.4f}")
    # logger.info(f"  MAE Distance Sparse: {mae_distance_sparse_sum / len(eval_loader):.4f}")
    # logger.info(f"  Cosine Similarity All: {diagonal_cs_all_sum / len(eval_loader):.4f}")
    # logger.info(f"  MAE Distance All: {mae_distance_all_sum / len(eval_loader):.4f}")
    # logger.info(f"  Sparsity Sparse: {sparsity_sparse_sum / len(eval_loader):.4f}")
    # logger.info(f"  Sparsity All: {sparsity_all_sum / len(eval_loader):.4f}")
    # logger.info(f"  Orthogonal Decoder Loss: {od_sum / len(eval_loader):.6f}")
    #
    metrics = {
        "loss": loss_all / len(eval_loader),
        "recon_loss": recon_loss_all / len(eval_loader),
        "sparse_loss": sparse_loss_all / len(eval_loader),
        "fvu_sparse": fvu_score_sparse_sum / len(eval_loader),
        "fvu_all": fvu_score_all_sum / len(eval_loader),
        "cknna_sparse": cknna_score_sparse_sum / len(eval_loader),
        "cknna_all": cknna_score_all_sum / len(eval_loader),
        "cos_sim_sparse": diagonal_cs_sparse_sum / len(eval_loader),
        "mae_sparse": mae_distance_sparse_sum / len(eval_loader),
        "cos_sim_all": diagonal_cs_all_sum / len(eval_loader),
        "mae_all": mae_distance_all_sum / len(eval_loader),
        "sparsity_sparse": sparsity_sparse_sum / len(eval_loader),
        "sparsity_all": sparsity_all_sum / len(eval_loader),
        "orthogonal_decoder": od_sum / len(eval_loader),
    }

    # Log metrics
    logger.info("Evaluation results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.6f}")

    return metrics


def compute_matryoshka_recon_loss(recons_sparse, repr_sparse, embeddings, loss_fn, weights, device):
    loss_recon_all = torch.tensor(0.0, requires_grad=True, device=device)
    for i in range(len(recons_sparse)):
        current_loss = loss_fn(recons_sparse[i], embeddings, repr_sparse[i])[1]
        loss_recon_all = loss_recon_all + current_loss * weights[i]
    return loss_recon_all / sum(weights)


def build_model_identifier(args, cfg, model):
    if args.model == "ReLUSAE":
        activation = f"{args.activation}_{str(cfg.loss.sparse_weight).split('.')[1]}"
    else:
        activation = cfg.model.activation

    model_appendix = ""
    if cfg.model.use_matryoshka:
        activation += f"_{model.nesting_list[0]}"
        model_appendix = "_RW" if "RW" in args.model else "_UW"
        if args.l1_granularity_threshold > 0:
            model_appendix += f"_L1_{args.l1_granularity_threshold}"

    model_params = (
        f"{cfg.model.n_latents}_"
        f"{cfg.model.n_inputs}_"
        f"{activation}{model_appendix}_"
        f"{cfg.model.tied}_"
        f"{cfg.model.normalize}_"
        f"{cfg.model.latent_soft_cap}"
    )
    return model_params


def save_model_checkpoint(
    model,
    save_dir,
    filename,
    train_ds,
    val_metrics,
    val_metrics_second,
    epoch,
):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    torch.save(
        {
            "model": model.state_dict(),
            "mean_center": train_ds.mean,
            "scaling_factor": train_ds.scaling_factor,
            "target_norm": train_ds.target_norm,
            "metrics_primary": val_metrics,
            "metrics_second": val_metrics_second,
            "epoch": epoch,
        },
        save_path,
    )
    return save_path


def save_best_model(
    *,
    current_metrics,
    best_loss,
    args,
    cfg,
    model,
    train_ds,
    dataset_name,
    test_dataset_name,
    epoch,
    val_metrics_second=None,
    suffix="",
):
    if current_metrics is None:
        return best_loss, None

    recon_loss = current_metrics["recon_loss"]
    if recon_loss >= best_loss:
        return best_loss, None

    model_id = build_model_identifier(args, cfg, model)
    filename = f"{model_id}_{dataset_name}_{test_dataset_name}{suffix}.pth"

    save_path = save_model_checkpoint(
        model=model,
        save_dir=args.best_model_path,
        filename=filename,
        train_ds=train_ds,
        val_metrics=current_metrics,
        val_metrics_second=val_metrics_second,
        epoch=epoch,
    )

    logger.info(
        f"New best model saved to {save_path} with recon loss {recon_loss:.6f}"
    )

    return recon_loss, save_path


def wandb_log_if_enabled(args, metrics, step=None):
    if args.wandb_project is not None:
        wandb.log(metrics, step=step)


def wandb_save(args, path, model_name="best-model", extra_metrics=None):
    if args.wandb_project is not None:
        import re
        safe_name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", model_name)
        artifact = wandb.Artifact(
            safe_name,
            type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)

        if extra_metrics:
            wandb.log(extra_metrics)


def main(args):
    """
    Main training function for Sparse Autoencoders.
    
    This function handles the complete training pipeline:
    1. Setting up configuration based on model type and arguments
    2. Loading and preparing datasets
    3. Initializing the appropriate model (standard or Matryoshka SAE)
    4. Setting up loss function, optimizer, and learning rate scheduler
    5. Executing the training loop with periodic evaluation
    6. Tracking metrics including reconstruction quality, sparsity, and dead neurons
    7. Saving the trained model with relevant metadata
    
    Args:
        args (argparse.Namespace): Command line arguments from parse_args()
    """
    logger.info("Starting training with the following arguments:")
    logger.info(args)
    
    # Get configuration based on model type
    cfg = get_config(args.model)
    cfg.training.epochs = args.epochs
    
    # Set the random seed for reproducibility
    set_seed(cfg.training.seed)
    
    # Set the device (GPU/CPU)
    device = get_device()
    logger.info(f"Using device: {device}")

    if args.wandb_project is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            mode=os.getenv("WANDB_MODE", "online"),
             tags=args.wandb_tags,
        )
        wandb.config.update({
            "model": args.model,
            "dataset/train": args.dataset_train,
            "dataset/test": args.dataset_test,
            "dataset/second": args.dataset_second_modality,
            "n_inputs": cfg.model.n_inputs,
            "n_latents": cfg.model.n_latents,
            "activation": cfg.model.activation,
            "scheduler": cfg.training.scheduler,
            "batch_size": cfg.training.batch_size,
        }, allow_val_change=True)

        run_id = wandb.run.id
        logger.info(f"WandB run_id: {run_id}")
    else:
        run_id = None

    try:

        # Load datasets
        train_ds = SAEDataset(
            args.dataset_train,
            dtype=cfg.training.dtype,
            mean_center=cfg.training.mean_center,
            target_norm=cfg.training.target_norm
        )
        eval_ds = SAEDataset(
            args.dataset_test,
            dtype=cfg.training.dtype,
            mean_center=cfg.training.mean_center,
            target_norm=cfg.training.target_norm
        )
        logger.info(f"Training dataset length: {len(train_ds)}, Evaluation dataset length: {len(eval_ds)}, Embedding size: {train_ds.vector_size}")
        logger.info(f"Training dataset mean center: {train_ds.mean.mean()}, Scaling factor: {train_ds.scaling_factor} with target norm {train_ds.target_norm}")
        logger.info(f"Evaluation dataset mean center: {eval_ds.mean.mean()}, Scaling factor: {eval_ds.scaling_factor} with target norm {eval_ds.target_norm}")
        assert train_ds.vector_size == eval_ds.vector_size, "Training and evaluation datasets must have the same embedding size"
        if args.dataset_second_modality is not None:
            eval_ds_second = SAEDataset(
                args.dataset_second_modality,
                dtype=cfg.training.dtype,
                mean_center=cfg.training.mean_center,
                target_norm=cfg.training.target_norm
            )
            logger.info(f"Second modality dataset mean center: {eval_ds_second.mean.mean()}, Scaling factor: {eval_ds_second.scaling_factor} with target norm {eval_ds_second.target_norm}")
            assert train_ds.vector_size == eval_ds_second.vector_size, "Training and second modality datasets must have the same embedding size"

        # Set model parameters based on dataset and arguments
        cfg.model.n_inputs = train_ds.vector_size

        # Calculate number of latent dimensions using expansion factor
        cfg.model.n_latents = int(args.expansion_factor * train_ds.vector_size)
        logger.info(f"Expansion factor: {args.expansion_factor}, Latent dimensions: {cfg.model.n_latents}")

        # Extract l1 from ReLU if applied
        if args.model == "ReLUSAE" and "_" in args.activation:
            args.activation, sparse_weight = args.activation.split("_")
            cfg.loss.sparse_weight = float(f"0.{sparse_weight}")
            logger.info(f"Changing sparsity weight value to {cfg.loss.sparse_weight}")

        # Override activation if specified in arguments
        if args.activation:
            cfg.model.activation = args.activation

        # Configure Matryoshka SAE parameters if applicable
        if cfg.model.use_matryoshka:
            # CLI override for max_nesting
            if args.max_nesting > 0:
                cfg.model.max_nesting = args.max_nesting

            # Max nesting defaults to n_latents (paper: "for higher expansion rates we adjust the upper limit")
            if cfg.model.max_nesting <= 0:
                max_nesting = cfg.model.n_latents
            else:
                max_nesting = min(cfg.model.max_nesting, cfg.model.n_latents)

            # Generate nesting list if a single value was provided
            if isinstance(cfg.model.nesting_list, int):
                logger.info(f"Generating nesting list from {cfg.model.nesting_list} to {max_nesting}")
                start = [cfg.model.nesting_list]
                while start[-1] < max_nesting:
                    new_k = start[-1] * 2
                    if new_k > max_nesting:
                        break
                    start.append(new_k)

                if max_nesting not in start:
                    start.append(max_nesting)
                cfg.model.nesting_list = start

            # Set importance weights for different nesting levels
            if cfg.model.relative_importance == "RW":
                # Reverse weighting - higher weight for larger k values
                cfg.model.relative_importance = list(reversed(list(range(1, len(cfg.model.nesting_list)+1))))
            elif cfg.model.relative_importance == "UW":
                # Uniform weighting - equal weight for all k values
                cfg.model.relative_importance = [1.0] * len(cfg.model.nesting_list)

            logger.info(f"Using Matryoshka with nesting list: {cfg.model.nesting_list} and weighting function: {cfg.model.relative_importance}")
        else:
            logger.info(f"Using standard SAE with {cfg.model.activation} activation")

        # Calculate bias initialization (median or zero)
        logger.info(f"Calculating bias initialization with median: {cfg.training.bias_init_median}")
        bias_init = 0.0
        if cfg.training.bias_init_median:
            # Use geometric median of a subset of data points for robustness
            bias_init = geometric_median(train_ds, device=device, max_number=len(train_ds)//10)
        logger.info(f"Bias initialization: {bias_init}")

        # Initialize the appropriate model type
        if cfg.model.use_matryoshka:
            model = MatryoshkaAutoencoder(bias_init=bias_init, **asdict(cfg.model))
        else:
            model = Autoencoder(bias_init=bias_init, **asdict(cfg.model))
        model = model.to(device)

        # Prepare loss function
        # Use zeros or calculate mean from dataset depending on config
        mean_input = torch.zeros((train_ds.vector_size,), dtype=cfg.training.dtype)
        if not cfg.training.mean_center:
            mean_input = calculate_vector_mean(train_ds, num_workers=cfg.training.num_workers)

        mean_input = mean_input.to(device)
        loss_fn = SAELoss(
            reconstruction_loss=args.loss_func or cfg.loss.reconstruction_loss,
            sparse_loss=cfg.loss.sparse_loss,
            sparse_weight=cfg.loss.sparse_weight,
            mean_input=mean_input,
        )

        if args.wandb_project is not None:
            wandb.config.update({"loss_fn":loss_fn})

        # Prepare the optimizer with adaptive settings based on device
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device.type
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            betas=(cfg.training.beta1, cfg.training.beta2),
            eps=cfg.training.eps,
            weight_decay=cfg.training.weight_decay,
            fused=use_fused
        )

        # Prepare the learning rate scheduler
        if cfg.training.scheduler == 1:
            # Linear decay scheduler
            logger.info(f"Using linear decay")
            scheduler = LinearDecayLR(optimizer, cfg.training.epochs, decay_time=cfg.training.decay_time)
        elif cfg.training.scheduler == 2:
            # Cosine annealing with warmup
            logger.info("using cosine warmup")
            scheduler = CosineWarmupScheduler(
                optimizer,
                max_lr=cfg.training.lr,
                warmup_epoch=1,
                final_lr_factor=0.1,
                total_epochs=cfg.training.epochs
            )
        else:
            # No scheduler
            logger.info(f"Using no scheduler")
            scheduler = None

        # Prepare the dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=True
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=False
        )
        if args.dataset_second_modality is not None:
            eval_loader_second = torch.utils.data.DataLoader(
                eval_ds_second,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.training.num_workers,
                shuffle=False
            )

        # Set up callbacks
        callbacks_list: list[TrainingCallback] = []
        cb_runner = CallbackRunner(callbacks_list)

        # Training loop
        global_step = 0
        numb_of_dead_neurons = 0
        dead_neurons = []
        best_val_loss = float("inf")
        best_val_loss_second = float("inf")
        best_model_info = None
        dataset_name = args.dataset_train.split("/")[-1].split(".")[0]
        test_dataset_name = args.dataset_test.split("/")[-1].split(".")[0]

        cb_runner.on_train_begin(model=model, cfg=cfg, args=args)

        for epoch in range(cfg.training.epochs):
            model.train()
            logger.info(f"Epoch {epoch+1}/{cfg.training.epochs}")

            # Training loop for current epoch
            for step, embeddings in enumerate(tqdm(train_loader, desc="Training")):
                optimizer.zero_grad()
                global_step += 1
                embeddings = embeddings.to(device)

                # Forward pass through model
                recons_sparse, repr_sparse, recons_all, repr_all = model(embeddings)

                # Compute loss based on model type
                if cfg.model.use_matryoshka:
                    # For Matryoshka models, compute weighted loss across all nesting levels
                    sparse_loss = loss_fn(recons_all, embeddings, repr_all)[-1]
                    recon_loss = loss_fn(recons_sparse[0], embeddings, repr_all)[1]
                    loss = compute_matryoshka_recon_loss(
                        recons_sparse,
                        repr_sparse,
                        embeddings,
                        loss_fn,
                        model.relative_importance,
                        device,
                    )

                    # L1 regularization for nesting levels >= threshold
                    if args.l1_granularity_threshold > 0 and cfg.loss.sparse_weight > 0:
                        l1_levels = [i for i, k in enumerate(model.nesting_list) if k >= args.l1_granularity_threshold]
                        if l1_levels:
                            l1_reg = sum(
                                loss_fn.sparse_loss_fn(repr_sparse[i], embeddings)
                                for i in l1_levels
                            ) / len(l1_levels)
                            loss = loss + cfg.loss.sparse_weight * l1_reg
                            sparse_loss = l1_reg.detach()

                    # Use first nesting level for metrics
                    repr_sparse = repr_sparse[0]
                    recons_sparse = recons_sparse[0]
                else:
                    # Standard SAE loss computation
                    loss, recon_loss, sparse_loss = loss_fn(recons_sparse, embeddings, repr_sparse)

                # Backpropagation
                loss.backward()

                # Weight normalization and gradient projection
                model.scale_to_unit_norm()
                model.project_grads_decode()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad)

                # Update model parameters
                optimizer.step()

                # Detach tensors for metric calculation
                recons_sparse, recons_all, embeddings = (
                    recons_sparse.detach(),
                    recons_all.detach(),
                    embeddings.detach(),
                )

                # Calculate evaluation metrics
                # CKNNA (Centered Kernel Nearest Neighbor Alignment) scores
                cknna_score_sparse = cknna(recons_sparse, embeddings)
                cknna_score_all = cknna(recons_all, embeddings)

                # FVU (Explained Variance) metric
                fvu_score_sparse = explained_variance(embeddings, recons_sparse)
                fvu_score_all = explained_variance(embeddings, recons_all)

                # Reconstruction quality metrics
                diagonal_cs_sparse, mae_distance_sparse = calculate_similarity_metrics(recons_sparse, embeddings)
                diagonal_cs_all, mae_distance_all = calculate_similarity_metrics(recons_all, embeddings)

                # Orthogonality of decoder features
                od = orthogonal_decoder(model.decoder)

                # Sparsity measurements
                sparsity_sparse = (repr_sparse == 0.0).float().mean(axis=-1).mean()
                sparsity_all = (repr_all == 0.0).float().mean(axis=-1).mean()

                # Representation Metrics
                repr_norm = repr_all.norm(dim=-1).mean().item()
                repr_max = repr_all.max(dim=-1).values.mean().item()
                active_frac = (repr_all.abs() > 1e-6).float().mean().item()

                # Check for dead neurons periodically
                if global_step % cfg.training.check_dead == 0:
                    activations = model.get_and_reset_stats()
                    dead_neurons = identify_dead_neurons(activations).numpy().tolist()
                    numb_of_dead_neurons = len(dead_neurons)
                    logger.info(f"Number of dead neurons: {numb_of_dead_neurons}")

                # Log metrics periodically
                if global_step % cfg.training.print_freq == 0:
                    logger.info(f"Epoch: {epoch+1}, Step: {step}, Loss: {loss.item():.6f}, Recon Loss: {recon_loss.item():.6f}, Sparse Loss: {sparse_loss.item():.6f}")
                    logger.info(f"FVU Sparse: {fvu_score_sparse:.4f}, FVU All: {fvu_score_all:.4f}")
                    logger.info(f"CKNNA Sparse: {cknna_score_sparse:.4f}, CKNNA All: {cknna_score_all:.4f}")
                    logger.info(f"Cosine Similarity Sparse: {diagonal_cs_sparse:.4f}, MAE Distance Sparse: {mae_distance_sparse:.4f}")
                    logger.info(f"Cosine Similarity All: {diagonal_cs_all:.4f}, MAE Distance All: {mae_distance_all:.4f}")
                    logger.info(f"Sparsity Sparse: {sparsity_sparse:.4f}, Sparsity All: {sparsity_all:.4f}")
                    logger.info(f"Orthogonal Decoder Loss: {od:.6f}, Representation norm {repr_norm:.4f} and max {repr_max:.2f}")

                if args.wandb_project is not None:
                    log_payload = {
                        "train/loss": loss.item(),
                        "train/recon_loss": recon_loss.item(),
                        "train/sparse_loss": sparse_loss.item(),
                        "train/fvu_sparse": fvu_score_sparse,
                        "train/fvu_all": fvu_score_all,
                        "train/cknna_sparse": cknna_score_sparse,
                        "train/cknna_all": cknna_score_all,
                        "train/sparsity_sparse": sparsity_sparse,
                        "train/sparsity_all": sparsity_all,
                        "train/orthogonal_decoder": od,
                        "repr/dead_neurons": numb_of_dead_neurons,
                        "repr/active_fraction": active_frac,
                        "global_step": global_step,
                    }
                    wandb.log(log_payload, step=global_step)


            # Update learning rate
            if scheduler:
                scheduler.step()

            # Log epoch summary
            lr_rate = scheduler.get_last_lr()[0] if scheduler else cfg.training.lr
            logger.info(f"Epoch: {epoch+1}, Learning Rate: {lr_rate:.6f}, Loss: {loss.item():.6f}, Recon Loss: {recon_loss.item():.6f}, Sparse Loss: {sparse_loss.item():.6f}, Dead neurons: {numb_of_dead_neurons}")

            # Evaluate the model on the validation set
            val_metrics = eval(model, eval_loader, loss_fn, device, cfg)

            # Evaluate second modality if provided
            if args.dataset_second_modality is not None:
                val_metrics_second = eval(model, eval_loader_second, loss_fn, device, cfg)
            else:
                val_metrics_second = None

            if cfg.model.use_matryoshka:
                mat_metrics = compute_matryoshka_metrics(model, eval_loader, device)
                val_metrics.update(mat_metrics)
                logger.info("Matryoshka metrics:")
                for k, v in mat_metrics.items():
                    logger.info(f"  {k}: {v:.6f}")

            if args.wandb_project is not None:
                wandb.log(
                    {f"val/{k}": v for k, v in val_metrics.items()},
                    step=global_step,
                )

                if val_metrics_second is not None:
                    wandb.log(
                        {f"val_second/{k}": v for k, v in val_metrics_second.items()},
                        step=global_step,
                    )
                    wandb.log(
                        {"val/cross_modality_gap":val_metrics["recon_loss"] - val_metrics_second["recon_loss"]},
                        step=global_step
                    )

            best_val_loss, save_path = save_best_model(
                current_metrics=val_metrics,
                best_loss=best_val_loss,
                args=args,
                cfg=cfg,
                model=model,
                train_ds=train_ds,
                dataset_name=dataset_name,
                test_dataset_name=test_dataset_name,
                epoch=epoch + 1,
                val_metrics_second=val_metrics_second,
            )

            if save_path:
                wandb_save(
                    args,
                    save_path,
                    f"best-model-{args.wandb_run_name}-{run_id}",
                    {
                        "best_model_epoch": epoch + 1,
                        "best_val_loss": best_val_loss,
                    },
                )

            if val_metrics_second is not None:
                test_dataset_name_2 = args.dataset_second_modality.split("/")[-1].split(".")[0]

                best_val_loss_second, save_path_2 = save_best_model(
                    current_metrics=val_metrics_second,
                    best_loss=best_val_loss_second,
                    args=args,
                    cfg=cfg,
                    model=model,
                    train_ds=train_ds,
                    dataset_name=dataset_name,
                    test_dataset_name=test_dataset_name_2,
                    epoch=epoch + 1,
                    val_metrics_second=val_metrics_second,
                    suffix="_second",
                )

                if save_path_2:
                    wandb_save(
                        args,
                        save_path_2,
                        f"best-model-second_modality-{args.wandb_run_name}-{run_id}",
                        {
                            "best_model_epoch_second": epoch + 1,
                            "best_val_loss_second": best_val_loss_second,
                        },
                    )

            cb_runner.on_epoch_end(
                epoch=epoch,
                model=model,
                train_ds=train_ds,
                val_metrics=val_metrics,
                global_step=global_step,
                device=device,
                args=args,
                cfg=cfg,
            )

        cb_runner.on_train_end(model=model, cfg=cfg, args=args)

        # Save the trained model
        # For Matryoshka models, append the first nesting level to activation name
        model_appendix = ""
        if args.model == "ReLUSAE":
            activation = f"{args.activation}_{str(cfg.loss.sparse_weight).split('.')[1]}"
        else:
            activation = cfg.model.activation

        if cfg.model.use_matryoshka:
            activation += f"_{model.nesting_list[0]}"
            if "RW" in args.model:
                model_appendix = "_RW"
            else:
                model_appendix = "_UW"
            if args.l1_granularity_threshold > 0:
                model_appendix += f"_L1_{args.l1_granularity_threshold}"

        # Construct filename with key hyperparameters
        model_params = f"{cfg.model.n_latents}_{cfg.model.n_inputs}_{activation}{model_appendix}_{cfg.model.tied}_{cfg.model.normalize}_{cfg.model.latent_soft_cap}"
        dataset_name = args.dataset_train.split("/")[-1].split(".")[0]
        save_path = os.path.join(
            args.out_dir,
            f"{model_params}_{dataset_name}.pth"
        )

        # Save model state and preprocessing parameters
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "mean_center": train_ds.mean,
            "scaling_factor": train_ds.scaling_factor,
            "target_norm": train_ds.target_norm,
        }, save_path)

        if args.wandb_project is not None:
            safe_final_name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", f"final-model-{args.wandb_run_name}-{run_id}")
            final_artifact = wandb.Artifact(
                name=safe_final_name,
                type="model",
                metadata={
                    "epoch": cfg.training.epochs,
                    "dataset_train": dataset_name,
                    "model": args.model,
                    "n_latents": cfg.model.n_latents,
                    "activation": cfg.model.activation,
                },
            )
            final_artifact.add_file(save_path)
            wandb.log_artifact(final_artifact)


        logger.info(f"Model saved to {save_path}")

    finally:
        if args.wandb_project is not None:
            wandb.finish()

    
if __name__ == "__main__":
    args = parse_args()
    main(args)
