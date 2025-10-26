"""
Path-X benchmark from Long Range Arena.
A challenging long-range dependency task requiring models to determine 
if there exists a path between two marked nodes in a 2D grid.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class PathXDataset(Dataset):
    """
    Path-X Dataset: Binary classification task on 2D grids.
    
    The task is to determine if there is a path between two marked positions
    in a binary grid (containing walls and open spaces).
    
    Args:
        num_samples: Number of samples to generate
        resolution: Grid resolution (default 128x128)
        seq_len: Flattened sequence length (resolution * resolution)
    """
    
    def __init__(self, num_samples=10000, resolution=16, difficulty=0.3):
        self.num_samples = num_samples
        self.resolution = resolution
        self.seq_len = resolution * resolution
        self.difficulty = difficulty  # Probability of a cell being a wall
        
        # Generate dataset
        self.data, self.labels = self._generate_dataset()
    
    def _generate_grid_with_path(self):
        """Generate a grid with guaranteed path between two points."""
        grid = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        
        # Randomly place walls
        wall_mask = np.random.rand(self.resolution, self.resolution) < self.difficulty
        grid[wall_mask] = 1.0
        
        # Choose start and end points
        start = (np.random.randint(0, self.resolution), np.random.randint(0, self.resolution))
        end = (np.random.randint(0, self.resolution), np.random.randint(0, self.resolution))
        
        # Ensure start and end are not walls
        grid[start] = 0.0
        grid[end] = 0.0
        
        # Mark start and end positions with special values
        grid[start] = 0.5  # Start marker
        grid[end] = -0.5   # End marker
        
        # Check if path exists using BFS
        has_path = self._bfs_path_exists(grid, start, end)
        
        return grid, has_path
    
    def _generate_grid_no_path(self):
        """Generate a grid with no path between two points."""
        grid = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        
        # Place a wall barrier
        wall_col = self.resolution // 2
        grid[:, wall_col] = 1.0
        
        # Choose start and end on opposite sides
        start = (np.random.randint(0, self.resolution), np.random.randint(0, wall_col))
        end = (np.random.randint(0, self.resolution), np.random.randint(wall_col + 1, self.resolution))
        
        grid[start] = 0.5  # Start marker
        grid[end] = -0.5   # End marker
        
        return grid, False
    
    def _bfs_path_exists(self, grid, start, end):
        """Check if path exists using BFS."""
        visited = set()
        queue = [start]
        visited.add(start)
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            current = queue.pop(0)
            
            if current == end:
                return True
            
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                
                if (0 <= nx < self.resolution and 
                    0 <= ny < self.resolution and 
                    (nx, ny) not in visited and 
                    grid[nx, ny] != 1.0):  # Not a wall
                    
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return False
    
    def _generate_dataset(self):
        """Generate the full dataset."""
        data = []
        labels = []
        
        for i in range(self.num_samples):
            # Generate half with paths, half without
            if i % 2 == 0:
                grid, has_path = self._generate_grid_with_path()
            else:
                grid, has_path = self._generate_grid_no_path()
            
            # Flatten grid to sequence
            sequence = grid.flatten()
            
            data.append(sequence)
            labels.append(1.0 if has_path else 0.0)
        
        return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])


def get_pathx_data(resolution=16, train_samples=8000, val_samples=1000, 
                   test_samples=1000, batch_size=32, difficulty=0.3):
    """
    Get Path-X dataloaders.
    
    Args:
        resolution: Grid resolution (e.g., 16 for 16x16 grid, 256 sequence length)
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        batch_size: Batch size for dataloaders
        difficulty: Difficulty level (wall probability)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print(f"Generating Path-X dataset with {resolution}x{resolution} grids...")
    print(f"Sequence length: {resolution * resolution}")
    print(f"Train: {train_samples}, Val: {val_samples}, Test: {test_samples}")
    
    train_dataset = PathXDataset(num_samples=train_samples, resolution=resolution, difficulty=difficulty)
    val_dataset = PathXDataset(num_samples=val_samples, resolution=resolution, difficulty=difficulty)
    test_dataset = PathXDataset(num_samples=test_samples, resolution=resolution, difficulty=difficulty)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def visualize_pathx_samples(dataset, num_samples=4, save_path=None):
    """
    Visualize Path-X grid samples.
    
    Args:
        dataset: PathXDataset instance
        num_samples: Number of samples to visualize
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(12, 6))
    axes = axes.flatten()
    
    # Custom colormap: white (open), black (wall), green (start), red (end)
    colors = ['white', 'black', 'green', 'red', 'yellow']
    n_bins = 5
    cmap = ListedColormap(colors)
    
    for i in range(min(num_samples, len(dataset))):
        sequence, label = dataset[i]
        label = label.item()
        
        # Reshape sequence back to grid
        grid = sequence.numpy().reshape(dataset.resolution, dataset.resolution)
        
        # Create display grid with distinct values for visualization
        display_grid = np.zeros_like(grid)
        display_grid[grid == 1.0] = 1  # Walls = black
        display_grid[grid == 0.5] = 2  # Start = green
        display_grid[grid == -0.5] = 3  # End = red
        
        axes[i].imshow(display_grid, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
        axes[i].set_title(f"Sample {i+1}\nPath Exists: {'Yes' if label == 1 else 'No'}", 
                         fontsize=10, fontweight='bold')
        axes[i].axis('off')
        
        # Add legend for first subplot
        if i == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='white', edgecolor='black', label='Open Space'),
                Patch(facecolor='black', label='Wall'),
                Patch(facecolor='green', label='Start'),
                Patch(facecolor='red', label='End')
            ]
            axes[i].legend(handles=legend_elements, loc='upper left', 
                          bbox_to_anchor=(0, 1), fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    return fig


def visualize_pathx_predictions(dataset, predictions, indices, save_path=None):
    """
    Visualize Path-X predictions with ground truth.
    
    Args:
        dataset: PathXDataset instance
        predictions: Array of predicted labels (0 or 1)
        indices: Indices of samples to visualize
        save_path: Optional path to save the figure
    """
    num_samples = len(indices)
    fig, axes = plt.subplots(2, (num_samples + 1) // 2, figsize=(12, 6))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = ['white', 'black', 'green', 'red']
    cmap = ListedColormap(colors)
    
    for i, idx in enumerate(indices):
        sequence, label = dataset[idx]
        label = label.item()
        pred = predictions[idx]
        
        # Reshape sequence back to grid
        grid = sequence.numpy().reshape(dataset.resolution, dataset.resolution)
        
        # Create display grid
        display_grid = np.zeros_like(grid)
        display_grid[grid == 1.0] = 1  # Walls
        display_grid[grid == 0.5] = 2  # Start
        display_grid[grid == -0.5] = 3  # End
        
        axes[i].imshow(display_grid, cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
        
        # Color code title based on correctness
        correct = (pred == label)
        color = 'green' if correct else 'red'
        axes[i].set_title(f"True: {'Path' if label == 1 else 'No Path'}\n"
                         f"Pred: {'Path' if pred == 1 else 'No Path'}", 
                         fontsize=10, fontweight='bold', color=color)
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prediction visualization saved to: {save_path}")
    
    plt.show()
    return fig
