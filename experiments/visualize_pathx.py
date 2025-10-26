"""
Standalone script to visualize Path-X dataset samples.
Quick way to see what the Path-X task looks like.
"""
import argparse
from acds.benchmarks.pathx import PathXDataset, visualize_pathx_samples

parser = argparse.ArgumentParser(description="Visualize Path-X dataset samples")
parser.add_argument("--resolution", type=int, default=16, help="Grid resolution")
parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to show")
parser.add_argument("--difficulty", type=float, default=0.3, help="Wall probability")
parser.add_argument("--save_path", type=str, default=None, help="Path to save figure")

args = parser.parse_args()

print("="*60)
print("PATH-X DATASET VISUALIZATION")
print("="*60)
print(f"Resolution: {args.resolution}x{args.resolution}")
print(f"Sequence Length: {args.resolution * args.resolution}")
print(f"Difficulty: {args.difficulty}")
print(f"Generating {args.num_samples} samples...")
print("="*60)

# Generate dataset
dataset = PathXDataset(
    num_samples=args.num_samples,
    resolution=args.resolution,
    difficulty=args.difficulty
)

# Visualize
visualize_pathx_samples(
    dataset,
    num_samples=args.num_samples,
    save_path=args.save_path
)

print("\nVisualization complete!")
print("\nColor Legend:")
print("  White = Open space (0)")
print("  Black = Wall (1)")
print("  Green = Start position (0.5)")
print("  Red = End position (-0.5)")
print("\nTask: Determine if a path exists from start (green) to end (red)")
