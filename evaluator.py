from typing import NamedTuple, List, Any, Optional, Dict
from itertools import chain
from dataclasses import dataclass
import itertools
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import wandb

from schedulers import Scheduler, LRSchedule
from models import Prober, build_mlp
from configs import ConfigBase

from dataset import WallDataset
from normalizer import Normalizer
from datasets import ProbingDataset


@dataclass
class ProbingConfig(ConfigBase):
    probe_targets: str = "locations"
    lr: float = 0.0002
    epochs: int = 20
    schedule: LRSchedule = LRSchedule.Cosine
    sample_timesteps: int = 30
    prober_arch: str = "256"


class ProbeResult(NamedTuple):
    model: torch.nn.Module
    average_eval_loss: float
    eval_losses_per_step: List[float]
    plots: List[Any]


default_config = ProbingConfig()


def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.shape == target.shape
    mse = (pred - target).pow(2).mean(dim=0)
    return mse


class ProbingEvaluator:
    def __init__(self, model, device, data_path, batch_size=32):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        
        # Initialize datasets
        self.train_ds = ProbingDataset(data_path, split='train')
        self.val_ds = ProbingDataset(data_path, split='val')
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
    
    def train_pred_prober(self):
        """Train a probing classifier for each attribute."""
        prober = {}
        for attr in self.train_ds.attributes:
            # Initialize probe model
            probe_model = nn.Linear(
                self.model.hidden_dim,
                len(self.train_ds.attribute_values[attr])
            ).to(self.device)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(probe_model.parameters())
            
            # Training loop
            for epoch in range(10):  # Adjust number of epochs as needed
                total_loss = 0
                correct = 0
                total = 0
                
                for batch in self.train_loader:
                    # Get model embeddings
                    with torch.no_grad():
                        embeddings = self.model.encode(batch['input'].to(self.device))
                    
                    # Forward pass
                    outputs = probe_model(embeddings)
                    labels = batch[attr].to(self.device)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Calculate metrics
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                # Log training metrics
                avg_loss = total_loss / len(self.train_loader)
                accuracy = 100 * correct / total
                wandb.log({
                    f"{attr}_train_loss": avg_loss,
                    f"{attr}_train_accuracy": accuracy,
                    "epoch": epoch
                })
            
            prober[attr] = probe_model
        
        return prober
    
    def evaluate(self):
        """Evaluate the model using all probing tasks."""
        prober = self.train_pred_prober()
        results = {}
        
        for attr, probe_model in prober.items():
            probe_model.eval()
            total_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in self.val_loader:
                    # Get model embeddings
                    embeddings = self.model.encode(batch['input'].to(self.device))
                    
                    # Forward pass
                    outputs = probe_model(embeddings)
                    labels = batch[attr].to(self.device)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    
                    # Calculate metrics
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            # Calculate final metrics
            avg_loss = total_loss / len(self.val_loader)
            accuracy = 100 * correct / total
            
            results[attr] = {
                'loss': avg_loss,
                'accuracy': accuracy
            }
            
            # Log validation metrics
            wandb.log({
                f"{attr}_val_loss": avg_loss,
                f"{attr}_val_accuracy": accuracy
            })
        
        return results
