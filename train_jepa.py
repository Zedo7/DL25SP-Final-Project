import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from torch.utils.data import DataLoader
import wandb

# Import your JEPA model
from models import JEPAWorldModel

# Import dataset and other components
from dataset import create_wall_dataloader
from schedulers import LRSchedule, Scheduler
from normalizer import Normalizer
from evaluator import ProbingEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Train a JEPA model for the wall environment')
    parser.add_argument('--data_path', type=str, default='/scratch/DL25SP', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--repr_dim', type=int, default=256, help='Representation dimension')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--no_latent', action='store_true', help='Disable latent variables')
    parser.add_argument('--no_vicreg', action='store_true', help='Disable VICReg regularization')
    parser.add_argument('--reg_weight', type=float, default=0.1, help='Weight for latent regularization')
    parser.add_argument('--vicreg_weight', type=float, default=1.0, help='Weight for VICReg loss')
    parser.add_argument('--lambda_var', type=float, default=25.0, help='Weight for variance term in VICReg')
    parser.add_argument('--lambda_cov', type=float, default=1.0, help='Weight for covariance term in VICReg')
    parser.add_argument('--save_dir', type=str, default='./', help='Directory to save model checkpoints')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def create_optimizer_and_scheduler(model, args, train_loader):
    """Create optimizer and scheduler for training the model."""
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.epochs
    
    scheduler = Scheduler(
        schedule=LRSchedule.Cosine,
        base_lr=args.lr,
        data_loader=train_loader,
        epochs=args.epochs,
        optimizer=optimizer,
    )
    
    return optimizer, scheduler


def train_jepa(args, train_loader, val_loader, model, device):
    """Train the JEPA model with VICReg regularization."""
    # Initialize wandb
    wandb.init(
        project="jepa-wall",
        config=vars(args),
        name=f"jepa-{args.repr_dim}-{args.latent_dim}-{args.hidden_dim}"
    )
    
    model.to(device)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, train_loader)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'model_weights.pth')
    
    best_val_loss = float('inf')
    step = 0
    
    jepa = model.jepa  # Access the JEPA model inside JEPAWorldModel
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        pred_loss_total = 0
        reg_loss_total = 0
        vicreg_loss_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            states = batch.states.to(device)
            actions = batch.actions.to(device)
            
            # Forward pass
            # Encode all states to get ground truth representations
            true_reprs = jepa.encode(states)
            
            # Sample latent variables and get predicted representations
            latents, latent_info = jepa.sample_latent(states, actions, deterministic=False)
            pred_reprs = model(states, actions)
            
            # Calculate prediction loss - MSE between predicted and true representations
            pred_loss = F.mse_loss(pred_reprs, true_reprs)
            
            # Calculate regularization losses
            
            # 1. Latent variable regularization
            if jepa.use_latent:
                reg_loss = jepa.compute_latent_regularization(latent_info)
            else:
                reg_loss = torch.tensor(0.0).to(device)
            
            # 2. VICReg regularization to prevent collapse
            if not args.no_vicreg:
                # Apply VICReg to final predicted representations in batch
                last_reprs = pred_reprs[:, -1]  # [B, repr_dim]
                vicreg_loss, vicreg_info = jepa.compute_vicreg_loss(
                    last_reprs, 
                    lambda_var=args.lambda_var,
                    lambda_cov=args.lambda_cov
                )
            else:
                vicreg_loss = torch.tensor(0.0).to(device)
                vicreg_info = {"var_loss": 0.0, "cov_loss": 0.0}
            
            # Total loss
            loss = pred_loss + args.reg_weight * reg_loss + args.vicreg_weight * vicreg_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.adjust_learning_rate(step)
            
            # Log progress
            train_loss += loss.item()
            pred_loss_total += pred_loss.item()
            reg_loss_total += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0
            vicreg_loss_total += vicreg_loss.item() if isinstance(vicreg_loss, torch.Tensor) else 0
            
            # Log batch metrics to wandb
            if (batch_idx + 1) % args.log_interval == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_pred_loss": pred_loss.item(),
                    "batch_reg_loss": reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0,
                    "batch_vicreg_loss": vicreg_loss.item() if isinstance(vicreg_loss, torch.Tensor) else 0,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "step": step
                })
                
                if not args.no_vicreg:
                    wandb.log({
                        "batch_vicreg_var_loss": vicreg_info['var_loss'],
                        "batch_vicreg_cov_loss": vicreg_info['cov_loss'],
                        "step": step
                    })
            
            step += 1
        
        # Calculate epoch averages
        avg_train_loss = train_loss / len(train_loader)
        avg_pred_loss = pred_loss_total / len(train_loader)
        avg_reg_loss = reg_loss_total / len(train_loader)
        avg_vicreg_loss = vicreg_loss_total / len(train_loader)
        
        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_pred_loss": avg_pred_loss,
            "train_reg_loss": avg_reg_loss,
            "train_vicreg_loss": avg_vicreg_loss,
        })
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                states = batch.states.to(device)
                actions = batch.actions.to(device)
                
                # Encode all states to get ground truth representations
                true_reprs = jepa.encode(states)
                
                # Get predicted representations (deterministic for validation)
                pred_reprs = model(states, actions)
                
                # Calculate prediction loss
                loss = F.mse_loss(pred_reprs, true_reprs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Log validation metrics
        wandb.log({
            "epoch": epoch,
            "val_loss": avg_val_loss
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, save_path)
            print(f"Model saved to {save_path}")
            
            # Log best model metrics
            wandb.log({
                "best_val_loss": best_val_loss,
                "best_epoch": epoch
            })
    
    # Save final model
    final_save_path = os.path.join(args.save_dir, 'model_weights_final.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
    }, final_save_path)
    print(f"Final model saved to {final_save_path}")
    
    # Log final metrics
    wandb.log({
        "final_val_loss": avg_val_loss,
        "final_epoch": args.epochs
    })
    
    # Finish wandb run
    wandb.finish()
    
    return model


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader = create_wall_dataloader(
        data_path=f"{args.data_path}/train",
        device="cpu",  # Load data on CPU, will be moved to GPU in training loop
        batch_size=args.batch_size,
        train=True
    )
    
    val_loader = create_wall_dataloader(
        data_path=f"{args.data_path}/probe_normal/val",
        device="cpu",
        batch_size=args.batch_size,
        train=False
    )
    
    # Create model
    model = JEPAWorldModel(
        input_channels=2,
        repr_dim=args.repr_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        use_latent=not args.no_latent,
        use_vicreg=not args.no_vicreg,
        device=device
    )
    
    # Print model architecture and parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Train model
    trained_model = train_jepa(args, train_loader, val_loader, model, device)


if __name__ == "__main__":
    main()