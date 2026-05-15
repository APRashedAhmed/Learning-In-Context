"""
Simplified Bouncing Ball Datamodule for In-Context-CPD.

This module contains the essential datamodule components extracted from the timescales repository,
focused specifically on loading the human task dataset for model evaluation.
"""

import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from loguru import logger


class HumanTaskDataset(Dataset):
    """
    Simplified version of HumanTaskDatasetV2 focused on loading preset human task datasets.
    
    This dataset loads the human bouncing ball task data used for evaluation,
    including samples, targets, and metadata needed for state extraction.
    """

    def __init__(
        self,
        preset_dataset: str,
        **kwargs
    ):
        """
        Initialize the dataset from a preset directory.
        
        Parameters
        ----------
        preset_dataset : str
            Path to the preset dataset directory containing:
            - dataset_meta.pkl: Metadata about the dataset
            - trial_meta.csv: Trial information
            - videos/: Directory containing trial data files
        """
        self.preset_dataset = Path(preset_dataset).expanduser()
        
        if not self.preset_dataset.exists():
            raise FileNotFoundError(f"Preset dataset not found: {self.preset_dataset}")
        
        logger.info(f"Loading preset dataset from {self.preset_dataset}")
        
        # Initialize attributes that will be populated by _load_preset_dataset
        self.df_data = None
        self.dict_metadata = None
        
        # Load the preset dataset
        (
            self.samples,
            self.targets,
            self.metadata,
        ) = self._load_preset_dataset()
        
        # Create a mock task object for compatibility with timescales
        self.task = type('MockTask', (), {
            'samples': self.samples.numpy() if hasattr(self.samples, 'numpy') else self.samples,
            'targets': self.targets.numpy() if hasattr(self.targets, 'numpy') else self.targets
        })()

    def _load_preset_dataset(self):
        """Load the preset dataset from disk."""
        dir_dataset = self.preset_dataset

        # Load metadata
        with open(str(dir_dataset / "dataset_meta.pkl"), "rb") as f:
            dict_metadata = pickle.load(f)
        
        # Load trial metadata
        df_data = pd.read_csv(dir_dataset / "trial_meta.csv", index_col=0)
        
        # Store as instance attributes for external access
        self.dict_metadata = dict_metadata
        self.df_data = df_data

        # Load samples for each trial
        list_samples = []
        for block, video in df_data[["Dataset Block", "Dataset Block Video"]].values:
            sample_file = (
                dir_dataset / f"videos/block_{block}/video_{video}/video_{video}_samples.csv"
            )
            if sample_file.exists():
                sample_data = pd.read_csv(sample_file, index_col=0).to_numpy()
                list_samples.append(torch.from_numpy(sample_data).float())
            else:
                logger.warning(f"Sample file not found: {sample_file}")
                # Create dummy data with shape matching others if possible
                if list_samples:
                    dummy_shape = (100, list_samples[0].shape[1])  # Default length
                    list_samples.append(torch.zeros(dummy_shape))
                else:
                    list_samples.append(torch.zeros((100, 5)))  # Default: 100 timesteps, 5 features

        # Pack and pad samples
        packed_samples = pack_sequence(list_samples, enforce_sorted=False)
        samples, _ = pad_packed_sequence(
            packed_samples,
            batch_first=True,
            padding_value=-1,
        )

        # Load targets for each trial
        list_targets = []
        for block, video in df_data[["Dataset Block", "Dataset Block Video"]].values:
            target_file = (
                dir_dataset / f"videos/block_{block}/video_{video}/video_{video}_parameters.csv"
            )
            if target_file.exists():
                target_data = pd.read_csv(target_file, index_col=0).to_numpy()
                list_targets.append(torch.from_numpy(target_data).float())
            else:
                logger.warning(f"Target file not found: {target_file}")
                # Create dummy data with shape matching samples
                if list_targets:
                    dummy_shape = (list_samples[len(list_targets)].shape[0], list_targets[0].shape[1])
                    list_targets.append(torch.zeros(dummy_shape))
                else:
                    dummy_shape = (list_samples[len(list_targets)].shape[0], samples.shape[-1])
                    list_targets.append(torch.zeros(dummy_shape))

        # Pack and pad targets
        packed_targets = pack_sequence(list_targets, enforce_sorted=False)
        targets, _ = pad_packed_sequence(
            packed_targets,
            batch_first=True,
            padding_value=-1,
        )

        # Create mask for valid timesteps
        mask = torch.arange(targets.shape[1]).unsqueeze(0) < torch.from_numpy(
            df_data.length.values
        ).unsqueeze(1)
        
        # Update metadata
        dict_metadata["mask"] = mask
        metadata = (df_data, dict_metadata)

        logger.info(f"Loaded dataset with {samples.shape[0]} trials, "
                   f"max length {samples.shape[1]}, "
                   f"{samples.shape[2]} input features")

        return samples, targets, metadata

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single trial."""
        sample = self.samples[idx]
        target = self.targets[idx]
        
        # Get metadata for this trial
        df_data, dict_metadata = self.metadata
        trial_meta = df_data.iloc[idx]
        mask = dict_metadata["mask"][idx]
        
        metadata = (trial_meta.to_dict(), {"mask": mask})
        
        return sample, target, metadata


class BouncingBallDataModule(LightningDataModule):
    """
    Simplified Lightning DataModule for bouncing ball evaluation.
    
    This datamodule is focused on loading human task datasets for model evaluation
    and state extraction, without the complexity of training data generation.
    """

    def __init__(
        self,
        preset_val_dataset: Optional[str] = None,
        preset_test_dataset: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs
    ):
        """
        Initialize the datamodule.
        
        Parameters
        ----------
        preset_val_dataset : str, optional
            Path to preset validation dataset directory
        preset_test_dataset : str, optional  
            Path to preset test dataset directory (defaults to val dataset if not provided)
        batch_size : int
            Batch size for dataloaders
        num_workers : int
            Number of workers for dataloaders
        pin_memory : bool
            Whether to pin memory in dataloaders
        """
        super().__init__()
        
        self.preset_val_dataset = preset_val_dataset
        self.preset_test_dataset = preset_test_dataset or preset_val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Store for easy access
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for the specified stage."""
        
        if stage == "fit" or stage is None:
            # For training stage - not implemented as we focus on evaluation
            pass
            
        if stage == "validate" or stage is None:
            if self.preset_val_dataset:
                logger.info("Setting up validation dataset")
                self.val_dataset = HumanTaskDataset(
                    preset_dataset=self.preset_val_dataset
                )
            
        if stage == "test" or stage is None:
            if self.preset_test_dataset:
                logger.info("Setting up test dataset")
                self.test_dataset = HumanTaskDataset(
                    preset_dataset=self.preset_test_dataset
                )

    def test_dataloader(self, human_dataset: bool = True):
        """Create test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not set up. Call setup('test') first.")
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not set up. Call setup('validate') first.")
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        """Custom collate function to handle variable-length sequences."""
        samples, targets, metadata = zip(*batch)
        
        # Stack samples and targets
        samples = torch.stack(samples)
        targets = torch.stack(targets)
        
        # Handle metadata - extract masks and other info
        trial_metas, mask_info = zip(*metadata)
        masks = torch.stack([info["mask"] for info in mask_info])
        
        # Combine metadata
        combined_metadata = (
            trial_metas,
            {"mask": masks}
        )
        
        return samples, targets, combined_metadata


# Helper function to create default datamodule configuration
def create_eval_datamodule(dataset_path: str, batch_size: int = 32, **kwargs):
    """
    Create a datamodule configured for evaluation with a human task dataset.
    
    Parameters
    ----------
    dataset_path : str
        Path to the human task dataset directory
    batch_size : int
        Batch size for evaluation
    **kwargs
        Additional keyword arguments for BouncingBallDataModule
        
    Returns
    -------
    BouncingBallDataModule
        Configured datamodule ready for evaluation
    """
    datamodule = BouncingBallDataModule(
        preset_test_dataset=dataset_path,
        preset_val_dataset=dataset_path,
        batch_size=batch_size,
        **kwargs
    )
    
    # Set up for test/evaluation
    datamodule.setup(stage="test")
    
    return datamodule