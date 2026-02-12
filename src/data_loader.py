"""
Data loading utilities for Brain-to-Text dataset.

This module provides PyTorch Dataset classes for loading neural recording data
and associated transcriptions.
"""

import torch
import numpy as np
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class BrainToTextDataset(Dataset):
    """
    PyTorch Dataset for Brain-to-Text decoding.
    
    Loads neural recordings and associated transcriptions from HDF5 files.
    """
    
    def __init__(self,
                 h5_file_path: str,
                 csv_metadata_path: Optional[str] = None,
                 transform=None):
        """
        Args:
            h5_file_path: Path to HDF5 file containing neural data
            csv_metadata_path: Path to CSV with metadata (optional)
            transform: Optional data transform function
        """
        self.h5_file_path = h5_file_path
        self.csv_metadata_path = csv_metadata_path
        self.transform = transform
        
        self.data = {
            'neural_features': [],
            'n_time_steps': [],
            'seq_class_ids': [],
            'seq_len': [],
            'transcriptions': [],
            'sentence_label': [],
            'session': [],
            'block_num': [],
            'trial_num': [],
            'corpus_ids': [],
        }
        
        self.corpus_map = {}
        self.random_corpus_id = -1
        
        self._load_data()
    
    def _load_data(self):
        """Load data from HDF5 file."""
        # Load metadata if provided
        if self.csv_metadata_path:
            df_metadata = pd.read_csv(self.csv_metadata_path)
            all_corpuses = df_metadata['Corpus'].unique()
            self.corpus_map = {name: i for i, name in enumerate(all_corpuses)}
            self.random_corpus_id = self.corpus_map.get('Random', -1)
        
        # Load HDF5 file
        with h5py.File(self.h5_file_path, 'r') as f:
            keys = list(f.keys())
            
            for key in keys:
                g = f[key]
                
                neural_features = g['input_features'][:]
                n_time_steps = g.attrs.get('n_time_steps', neural_features.shape[0])
                seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
                seq_len = g.attrs.get('seq_len', None)
                transcription = g['transcription'][:] if 'transcription' in g else None
                sentence_label = g.attrs.get('sentence_label', None)
                session = g.attrs.get('session', '')
                block_num = g.attrs.get('block_num', -1)
                trial_num = g.attrs.get('trial_num', -1)
                
                # Determine corpus ID
                corpus_id = 1  # Default
                if self.csv_metadata_path and session and block_num >= 0:
                    corpus_id = self._get_corpus_id(session, block_num, df_metadata)
                
                self.data['neural_features'].append(neural_features)
                self.data['n_time_steps'].append(n_time_steps)
                self.data['seq_class_ids'].append(seq_class_ids)
                self.data['seq_len'].append(seq_len)
                self.data['transcriptions'].append(transcription)
                self.data['sentence_label'].append(sentence_label)
                self.data['session'].append(session)
                self.data['block_num'].append(block_num)
                self.data['trial_num'].append(trial_num)
                self.data['corpus_ids'].append([corpus_id])
        
        self.data['corpus_ids'] = torch.tensor(self.data['corpus_ids'], dtype=torch.long)
    
    def _get_corpus_id(self, session: str, block_num: int, df_metadata: pd.DataFrame) -> int:
        """
        Look up corpus ID from metadata.
        
        Args:
            session: Session identifier
            block_num: Block number
            df_metadata: DataFrame with metadata
            
        Returns:
            Corpus ID
        """
        try:
            date_part = session.split('.', 1)[1]
            dt = datetime.strptime(date_part, "%Y.%m.%d")
            formatted_date = dt.strftime("%Y-%m-%d")
            
            corpus_list = df_metadata[
                (df_metadata['Date'] == formatted_date) &
                (df_metadata['Block number'] == block_num)
            ]['Corpus'].values
            
            if len(corpus_list) > 0:
                corpus_name = corpus_list[0]
                return self.corpus_map.get(corpus_name, 1)
        except (IndexError, ValueError, KeyError):
            pass
        
        return 1  # Default
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data['neural_features'])
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with neural features, transcription, and metadata
        """
        sample = {
            'neural_features': torch.tensor(self.data['neural_features'][idx], dtype=torch.float32),
            'n_time_steps': self.data['n_time_steps'][idx],
            'seq_class_ids': self.data['seq_class_ids'][idx],
            'seq_len': self.data['seq_len'][idx],
            'transcription': self.data['transcriptions'][idx],
            'sentence_label': self.data['sentence_label'][idx],
            'session': self.data['session'][idx],
            'block_num': self.data['block_num'][idx],
            'trial_num': self.data['trial_num'][idx],
            'corpus_id': self.data['corpus_ids'][idx],
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def collate_batch(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-length sequences.
    
    Args:
        batch: List of data samples
        
    Returns:
        Batched data with padding
    """
    neural_features = torch.nn.utils.rnn.pad_sequence(
        [item['neural_features'] for item in batch],
        batch_first=True,
        padding_value=0.0
    )
    
    corpus_ids = torch.cat([item['corpus_id'] for item in batch])
    
    # Extract other fields that don't need padding
    collated = {
        'neural_features': neural_features,
        'corpus_ids': corpus_ids,
        'seq_len': [item['seq_len'] for item in batch],
        'session': [item['session'] for item in batch],
        'transcription': [item['transcription'] for item in batch],
    }
    
    return collated


def create_data_loader(h5_file_path: str,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 0,
                      csv_metadata_path: Optional[str] = None) -> DataLoader:
    """
    Create a DataLoader for Brain-to-Text dataset.
    
    Args:
        h5_file_path: Path to HDF5 file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        csv_metadata_path: Optional path to CSV metadata
        
    Returns:
        PyTorch DataLoader instance
    """
    dataset = BrainToTextDataset(
        h5_file_path=h5_file_path,
        csv_metadata_path=csv_metadata_path
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch
    )
    
    return dataloader
