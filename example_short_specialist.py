#!/usr/bin/env python3
"""
Example Script for Short Sequence Specialist Training

This script demonstrates how to use the short sequence specialist training system
with a minimal example that trains on just 5 sequences and evaluates on 3 sequences.
"""

import subprocess
import sys
from pathlib import Path


def run_example():
    """Run a minimal example of the short specialist training system"""

    print("=" * 70)
    print("SHORT SEQUENCE SPECIALIST - MINIMAL EXAMPLE")
    print("=" * 70)
    print()
    print("This example will:")
    print("1. Train a specialist on 5 short sequences (30-40 AA)")
    print("2. Evaluate it on 3 unseen short sequences")
    print("3. Show you the results")
    print()

    # Check if required files exist
    required_files = [
        "train_short_specialist.py",
        "evaluate_short_specialist.py",
        "run_short_specialist.py"
    ]

    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        return False

    # Check if training datasets exist
    dataset_files = [
        "training_dataset_very_short.csv",
        "training_dataset_short.csv"
    ]

    available_datasets = [f for f in dataset_files if Path(f).exists()]
    if not available_datasets:
        print("Warning: No training dataset files found.")
        print("Please ensure you have the training dataset CSV files.")
        return False

    print(f"Found training datasets: {available_datasets}")
    print()

    try:
        # Run training with minimal parameters
        print("Step 1: Training specialist on 5 sequences...")
        print("-" * 50)

        train_cmd = [
            sys.executable, "run_short_specialist.py",
            "--mode", "both",  # Train and evaluate
            "--num_sequences", "5",  # Small number for quick example
            "--num_test_sequences", "3",  # Small number for quick example
            "--n_iterations", "50",  # Reduced iterations for speed
            "--batch_size", "8",  # Smaller batch size
            "--wandb_project", "short-specialist-example"
        ]

        print(f"Running: {' '.join(train_cmd)}")
        print()

        result = subprocess.run(train_cmd, check=True, capture_output=False)

        print()
        print("=" * 70)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print()
        print("Results are saved in:")
        print("- Training: outputs/short_specialist/")
        print("- Evaluation: outputs/short_specialist_evaluation/")
        print()
        print("To run a full training session, use:")
        print("python run_short_specialist.py --mode both --num_sequences 50 --num_test_sequences 20")
        print()

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error running example: {e}")
        return False
    except KeyboardInterrupt:
        print("\nExample interrupted by user.")
        return False


def show_usage_examples():
    """Show various usage examples"""

    print("=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)
    print()

    examples = [
        {
            "title": "Quick Training (5 sequences)",
            "command": "python run_short_specialist.py --mode train --num_sequences 5 --n_iterations 50",
            "description": "Train on 5 sequences with 50 iterations each (fast)"
        },
        {
            "title": "Full Training (50 sequences)",
            "command": "python run_short_specialist.py --mode train --num_sequences 50 --n_iterations 200",
            "description": "Train on 50 sequences with 200 iterations each (comprehensive)"
        },
        {
            "title": "Train and Evaluate",
            "command": "python run_short_specialist.py --mode both --num_sequences 30 --num_test_sequences 15",
            "description": "Train on 30 sequences, then evaluate on 15 unseen sequences"
        },
        {
            "title": "Evaluate on Medium Sequences",
            "command": "python run_short_specialist.py --mode eval --test_type medium --num_test_sequences 10",
            "description": "Test generalization to medium-length sequences (50-100 AA)"
        },
        {
            "title": "Custom Architecture",
            "command": "python run_short_specialist.py --mode train --hidden_dim 512 --n_hidden_layers 6 --lr 0.01",
            "description": "Train with larger model architecture and higher learning rate"
        },
        {
            "title": "High-Quality Evaluation",
            "command": "python run_short_specialist.py --mode eval --num_samples 500 --num_test_sequences 25",
            "description": "Generate 500 samples per weight combination for thorough evaluation"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['title']}")
        print(f"   Command: {example['command']}")
        print(f"   Description: {example['description']}")
        print()


def main():
    """Main function"""

    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        show_usage_examples()
        return 0

    print("Short Sequence Specialist Training System")
    print("Example and Usage Guide")
    print()

    # Ask user what they want to do
    print("What would you like to do?")
    print("1. Run minimal example (5 sequences)")
    print("2. Show usage examples")
    print("3. Exit")
    print()

    try:
        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            success = run_example()
            return 0 if success else 1
        elif choice == "2":
            show_usage_examples()
            return 0
        elif choice == "3":
            print("Goodbye!")
            return 0
        else:
            print("Invalid choice. Please run the script again.")
            return 1

    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
