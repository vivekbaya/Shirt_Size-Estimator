"""
Dataset Generation for Shirt Size Prediction Model

This script generates synthetic training data based on realistic body measurement distributions.
Since real-world body measurement data with ground truth shirt sizes is hard to obtain,
we use anthropometric studies and clothing size standards to create realistic synthetic data.

Data Sources:
- ANSUR II (US Army Anthropometric Survey)
- ISO/ASTM clothing size standards
- Body measurement statistics from fashion industry
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
from pathlib import Path


class DatasetGenerator:
    """Generate synthetic shirt size dataset"""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with reproducible seed"""
        np.random.seed(seed)
        
        # Size definitions based on common US sizing standards
        # Values represent typical body measurements in cm
        self.size_standards = {
            'XS': {
                'shoulder_width': (40, 43),   # cm
                'chest_circ': (84, 89),       # cm
                'waist_circ': (71, 76),       # cm
                'torso_length': (60, 65),     # cm
                'height': (160, 170),         # cm
            },
            'S': {
                'shoulder_width': (43, 46),
                'chest_circ': (89, 94),
                'waist_circ': (76, 81),
                'torso_length': (63, 68),
                'height': (165, 175),
            },
            'M': {
                'shoulder_width': (46, 49),
                'chest_circ': (94, 102),
                'waist_circ': (81, 89),
                'torso_length': (66, 71),
                'height': (170, 180),
            },
            'L': {
                'shoulder_width': (49, 52),
                'chest_circ': (102, 110),
                'waist_circ': (89, 97),
                'torso_length': (69, 74),
                'height': (175, 185),
            },
            'XL': {
                'shoulder_width': (52, 55),
                'chest_circ': (110, 118),
                'waist_circ': (97, 107),
                'torso_length': (72, 77),
                'height': (178, 190),
            },
            'XXL': {
                'shoulder_width': (55, 58),
                'chest_circ': (118, 128),
                'waist_circ': (107, 117),
                'torso_length': (75, 80),
                'height': (180, 195),
            }
        }
        
        # Fit type modifiers (relative to regular fit)
        self.fit_modifiers = {
            'slim': {
                'chest_modifier': 0.95,    # 5% tighter
                'waist_modifier': 0.90,    # 10% tighter
            },
            'regular': {
                'chest_modifier': 1.0,
                'waist_modifier': 1.0,
            },
            'relaxed': {
                'chest_modifier': 1.05,    # 5% looser
                'waist_modifier': 1.10,    # 10% looser
            }
        }
        
        # Distribution weights (reflecting real-world size distribution)
        self.size_weights = {
            'XS': 0.10,
            'S': 0.20,
            'M': 0.30,
            'L': 0.25,
            'XL': 0.10,
            'XXL': 0.05
        }
        
        self.fit_weights = {
            'slim': 0.25,
            'regular': 0.55,
            'relaxed': 0.20
        }
    
    def _generate_measurements(
        self, 
        size: str, 
        fit_type: str
    ) -> Dict[str, float]:
        """Generate realistic body measurements for a given size and fit"""
        
        standards = self.size_standards[size]
        modifiers = self.fit_modifiers[fit_type]
        
        # Generate base measurements with normal distribution
        shoulder_width = np.random.uniform(*standards['shoulder_width'])
        chest_circ = np.random.uniform(*standards['chest_circ']) * modifiers['chest_modifier']
        waist_circ = np.random.uniform(*standards['waist_circ']) * modifiers['waist_modifier']
        torso_length = np.random.uniform(*standards['torso_length'])
        height = np.random.uniform(*standards['height'])
        
        # Calculate normalized ratios (relative to image dimensions)
        # We simulate a person at 2 meters distance from 640x480 camera
        # Using typical field of view calculations
        
        # Assume person occupies ~60-80% of image width
        image_width_cm = height * 0.4  # Approximate visible width in frame
        image_height_cm = height * 1.2  # Approximate visible height in frame
        image_diagonal_cm = np.sqrt(image_width_cm**2 + image_height_cm**2)
        
        # Calculate ratios
        shoulder_ratio = shoulder_width / image_diagonal_cm
        chest_ratio = (chest_circ / (2 * np.pi)) / image_diagonal_cm  # Convert circumference to width
        waist_ratio = (waist_circ / (2 * np.pi)) / image_diagonal_cm
        torso_proportion = torso_length / shoulder_width
        
        # Add realistic noise (camera angle, posture variations)
        noise_factor = 0.05
        shoulder_ratio *= np.random.normal(1.0, noise_factor)
        chest_ratio *= np.random.normal(1.0, noise_factor)
        waist_ratio *= np.random.normal(1.0, noise_factor)
        torso_proportion *= np.random.normal(1.0, noise_factor)
        
        return {
            'shoulder_ratio': float(shoulder_ratio),
            'chest_ratio': float(chest_ratio),
            'waist_ratio': float(waist_ratio),
            'torso_proportion': float(torso_proportion),
            # Store raw measurements for reference
            'shoulder_width_cm': float(shoulder_width),
            'chest_circ_cm': float(chest_circ),
            'waist_circ_cm': float(waist_circ),
            'torso_length_cm': float(torso_length),
            'height_cm': float(height)
        }
    
    def generate_dataset(
        self, 
        num_samples: int = 10000,
        validation_split: float = 0.15,
        test_split: float = 0.15
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate complete dataset with train/val/test splits
        
        Args:
            num_samples: Total number of samples to generate
            validation_split: Fraction for validation set
            test_split: Fraction for test set
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        
        print(f"Generating {num_samples} samples...")
        
        data = []
        
        sizes = list(self.size_weights.keys())
        size_probs = list(self.size_weights.values())
        
        fits = list(self.fit_weights.keys())
        fit_probs = list(self.fit_weights.values())
        
        for i in range(num_samples):
            # Sample size and fit based on realistic distributions
            size = np.random.choice(sizes, p=size_probs)
            fit_type = np.random.choice(fits, p=fit_probs)
            
            # Generate measurements
            measurements = self._generate_measurements(size, fit_type)
            
            # Create sample
            sample = {
                'sample_id': i,
                'size': size,
                'fit_type': fit_type,
                **measurements
            }
            
            data.append(sample)
            
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split data
        n_test = int(len(df) * test_split)
        n_val = int(len(df) * validation_split)
        n_train = len(df) - n_test - n_val
        
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val:].copy()
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
        print(f"  Test:  {len(test_df)} samples")
        
        # Print distribution statistics
        print(f"\nSize distribution:")
        print(train_df['size'].value_counts(normalize=True).sort_index())
        
        print(f"\nFit distribution:")
        print(train_df['fit_type'].value_counts(normalize=True).sort_index())
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def save_dataset(
        self, 
        datasets: Dict[str, pd.DataFrame],
        output_dir: str = 'data'
    ):
        """Save datasets to CSV files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True,exist_ok=True)
        
        for split_name, df in datasets.items():
            filepath = output_path / f'{split_name}.csv'
            df.to_csv(filepath, index=False)
            print(f"Saved {split_name} set to {filepath}")
        
        # Save metadata
        metadata = {
            'num_sizes': len(self.size_standards),
            'size_classes': list(self.size_standards.keys()),
            'num_fits': len(self.fit_modifiers),
            'fit_classes': list(self.fit_modifiers.keys()),
            'feature_names': ['shoulder_ratio', 'chest_ratio', 'waist_ratio', 'torso_proportion'],
            'size_standards': self.size_standards,
            'size_distribution': self.size_weights,
            'fit_distribution': self.fit_weights
        }
        
        metadata_path = output_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")


def main():
    """Generate and save the dataset"""
    
    # Create generator
    generator = DatasetGenerator(seed=42)
    
    # Generate datasets
    datasets = generator.generate_dataset(
        num_samples=10000,
        validation_split=0.15,
        test_split=0.15
    )
    
    # Save to disk
    generator.save_dataset(datasets, output_dir='data/synthetic_sizes')
    
    print("\n✅ Dataset generation complete!")
    print("\nNext steps:")
    print("1. Review the generated data in data/synthetic_sizes/")
    print("2. Run train_model.py to train the neural network")
    print("3. Evaluate the model using evaluate_model.py")


if __name__ == '__main__':
    main()
