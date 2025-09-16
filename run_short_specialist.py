#!/usr/bin/env python3
"""
Runner Script for Short Sequence Specialist Training and Evaluation

This script provides an easy interface to train a Conditional GFlowNet specialist
on short protein sequences and then evaluate it on unseen proteins.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_training(args):
    """Run the training script"""
    print("=" * 60)
    print("TRAINING SHORT SEQUENCE SPECIALIST")
    print("=" * 60)

    # Build training command
    cmd = [
        sys.executable, "train_short_specialist.py",
        "--n_iterations", str(args.n_iterations),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--lr_logz", str(args.lr_logz),
        "--embedding_dim", str(args.embedding_dim),
        "--hidden_dim", str(args.hidden_dim),
        "--n_hidden_layers", str(args.n_hidden_layers),
        "--min_length", str(args.min_length),
        "--max_length", str(args.max_length),
        "--num_sequences", str(args.num_sequences),
        "--wandb_project", args.wandb_project
    ]

    if args.run_name:
        cmd.extend(["--run_name", args.run_name])

    print(f"Running training command:")
    print(" ".join(cmd))
    print()

    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\nTraining completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error: {e}")
        return False


def run_evaluation(args, model_path):
    """Run the evaluation script"""
    print("=" * 60)
    print("EVALUATING SHORT SEQUENCE SPECIALIST")
    print("=" * 60)

    # Build evaluation command
    cmd = [
        sys.executable, "evaluate_short_specialist.py",
        "--model_path", model_path,
        "--test_type", args.test_type,
        "--num_sequences", str(args.num_test_sequences),
        "--num_samples", str(args.num_samples),
        "--wandb_project", args.wandb_project
    ]

    if args.run_name:
        cmd.extend(["--run_name", args.run_name])

    print(f"Running evaluation command:")
    print(" ".join(cmd))
    print()

    # Run evaluation
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\nEvaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nEvaluation failed with error: {e}")
        return False


def find_latest_model():
    """Find the most recently created model file"""
    outputs_dir = Path("outputs/short_specialist")
    if not outputs_dir.exists():
        return None

    # Find the most recent training directory
    training_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("Short_Specialist_")]
    if not training_dirs:
        return None

    latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
    model_path = latest_dir / "trained_short_specialist_model.pth"

    if model_path.exists():
        return str(model_path)
    else:
        return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Conditional GFlowNet specialist on short protein sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a specialist on 30 sequences (30-40 AA)
  python run_short_specialist.py --mode train --num_sequences 30

  # Train and then evaluate
  python run_short_specialist.py --mode both --num_sequences 50 --num_test_sequences 20

  # Only evaluate (uses latest trained model)
  python run_short_specialist.py --mode eval --test_type medium --num_test_sequences 15

  # Train with custom parameters
  python run_short_specialist.py --mode train --n_iterations 500 --batch_size 32 --lr 0.01
        """
    )

    # Mode selection
    parser.add_argument("--mode", type=str, required=True,
                       choices=["train", "eval", "both"],
                       help="Mode: train only, evaluate only, or both")

    # Training parameters
    parser.add_argument("--n_iterations", type=int, default=200,
                       help="Number of training iterations per sequence")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005,
                       help="Learning rate")
    parser.add_argument("--lr_logz", type=float, default=0.1,
                       help="LogZ learning rate")
    parser.add_argument("--embedding_dim", type=int, default=32,
                       help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden dimension")
    parser.add_argument("--n_hidden_layers", type=int, default=4,
                       help="Number of hidden layers")

    # Data parameters
    parser.add_argument("--min_length", type=int, default=30,
                       help="Minimum protein sequence length")
    parser.add_argument("--max_length", type=int, default=40,
                       help="Maximum protein sequence length")
    parser.add_argument("--num_sequences", type=int, default=50,
                       help="Number of sequences to train on")

    # Evaluation parameters
    parser.add_argument("--test_type", type=str, default="unseen_short",
                       choices=["unseen_short", "medium", "long", "mixed"],
                       help="Type of test sequences")
    parser.add_argument("--num_test_sequences", type=int, default=20,
                       help="Number of test sequences")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples per weight combination")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to model file (if not provided, uses latest)")

    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default="short-sequence-specialist",
                       help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                       help="W&B run name")

    args = parser.parse_args()

    success = True

    if args.mode in ["train", "both"]:
        success = run_training(args)
        if not success:
            print("Training failed. Exiting.")
            return 1

    if args.mode in ["eval", "both"]:
        # Determine model path
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = find_latest_model()
            if model_path is None:
                print("No trained model found. Please train a model first or specify --model_path")
                return 1
            print(f"Using latest model: {model_path}")

        success = run_evaluation(args, model_path)
        if not success:
            print("Evaluation failed.")
            return 1

    if success:
        print("\n" + "=" * 60)
        print("ALL OPERATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nCheck the outputs/ directory for results:")
        print("- Training results: outputs/short_specialist/")
        print("- Evaluation results: outputs/short_specialist_evaluation/")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
