import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm

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
    parser.add_argument('--device', type=str, default='cuda')
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


def create_dataloader(data_path, split, batch_size, num_workers=4):
    dataset = WallDataset(data_path, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )


class WallDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.states = np.load(os.path.join(data_path, split, 'states.npy'))
        self.actions = np.load(os.path.join(data_path, split, 'actions.npy'))
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = torch.from_numpy(self.states[idx]).float()
        action = torch.from_numpy(self.actions[idx]).float()
        return state, action


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for states, actions in tqdm(train_loader, desc='Training'):
        states = states.to(device)
        actions = actions.to(device)
        
        # Forward pass
        predictions = model(states, actions)
        
        # Compute loss
        loss = F.mse_loss(predictions, model.jepa.encode(states))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for states, actions in tqdm(val_loader, desc='Validating'):
            states = states.to(device)
            actions = actions.to(device)
            
            predictions = model(states, actions)
            loss = F.mse_loss(predictions, model.jepa.encode(states))
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


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
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validation
        val_loss = validate(model, val_loader, device)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"Model saved to {save_path}")
            
            # Log best model metrics
            wandb.log({
                "best_val_loss": best_val_loss,
                "best_epoch": epoch
            })
        
        step += 1
    
    # Save final model
    final_save_path = os.path.join(args.save_dir, 'model_weights_final.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, final_save_path)
    print(f"Final model saved to {final_save_path}")
    
    # Log final metrics
    wandb.log({
        "final_val_loss": val_loss,
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader = create_dataloader(args.data_path, 'train', args.batch_size)
    val_loader = create_dataloader(args.data_path, 'val', args.batch_size)
    
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