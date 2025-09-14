import sys
import os
import argparse
import torch
import logging
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torchgfn', 'src'))

from env import CodonDesignEnv
from preprocessor import CodonSequencePreprocessor
from utils import load_config, set_seed
from generalization_tests import GeneralizationTester, run_generalization_tests

# Import model building functions
from main import set_up_logF_estimator
from main_conditional import build_subTB_gflownet, build_conditional_pf_pb
from torchgfn.src.gfn.gflownet import SubTBGFlowNet
from torchgfn.src.gfn.modules import DiscretePolicyEstimator
from torchgfn.src.gfn.samplers import Sampler
from torchgfn.src.gfn.utils.modules import MLP


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_trained_model(model_path: str, config, args, device):
    """
    Load a trained model from a checkpoint file.

    Args:
        model_path: Path to the saved model
        config: Configuration object
        args: Arguments object
        device: PyTorch device

    Returns:
        Tuple of (gflownet, sampler, env, preprocessor)
    """
    logging.info(f"Loading trained model from: {model_path}")

    # Create environment
    env = CodonDesignEnv(protein_seq=config.protein_seq, device=device)
    preprocessor = CodonSequencePreprocessor(
        env.seq_length, embedding_dim=args.embedding_dim, device=device
    )

    # Load model
    checkpoint = torch.load(model_path, map_location=device)

    # Determine model type from checkpoint or args
    is_conditional = args.conditional or 'conditional' in model_path.lower()

    if is_conditional:
        logging.info("Loading conditional model...")
        gflownet, pf_estimator, pb_estimator = build_subTB_gflownet(
            env, preprocessor, args, lamda=args.subTB_lambda
        )
    else:
        logging.info("Loading unconditional model...")
        # Build unconditional model
        arch = getattr(config, 'arch', 'MLP')

        if arch == 'MLP_EHH':

            from ENN_ENH import MLP_ENN
            module_PF = MLP_ENN(
                input_dim=preprocessor.output_dim,
                output_dim=env.n_actions,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
            )
            module_PB = MLP_ENN(
                input_dim=preprocessor.output_dim,
                output_dim=env.n_actions - 1,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
                trunk=module_PF.trunk if args.tied else None,
            )

        elif arch == 'Transformer':
            from DeepArchi import TransformerModel
            module_PF = TransformerModel(
                input_dim=preprocessor.output_dim,
                hidden_dim=args.hidden_dim,
                output_dim=env.n_actions,
                n_layers=args.n_hidden,
                n_head=8
            )
            module_PB = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=env.n_actions - 1,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
                trunk=None,
            )
        else:
            module_PF = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=env.n_actions,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
            )
            module_PB = MLP(
                input_dim=preprocessor.output_dim,
                output_dim=env.n_actions - 1,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
                trunk=module_PF.trunk if args.tied else None,
            )

        pf_estimator = DiscretePolicyEstimator(
            module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False
        )
        pb_estimator = DiscretePolicyEstimator(
            module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True
        )

        logF_estimator = set_up_logF_estimator(args, preprocessor, pf_estimator)

        gflownet = SubTBGFlowNet(
            pf=pf_estimator,
            pb=pb_estimator,
            logF=logF_estimator,
            weighting=args.subTB_weighting,
            lamda=args.subTB_lambda,
        )

    gflownet.load_state_dict(checkpoint['model_state'])
    gflownet = gflownet.to(device)

    sampler = Sampler(estimator=pf_estimator)

    logging.info("Model loaded successfully!")
    return gflownet, sampler, env, preprocessor


def main():
    parser = argparse.ArgumentParser(description="Run generalization tests on trained models")

    # Model loading arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--config_path", type=str, default="config.yaml",
                       help="Path to the configuration file")
    parser.add_argument("--conditional", action="store_true",
                       help="Whether the model is conditional")

    # Generalization testing arguments
    parser.add_argument("--n_samples", type=int, default=50,
                       help="Number of samples per weight configuration")
    parser.add_argument("--output_dir", type=str, default="generalization_results",
                       help="Output directory for results")
    parser.add_argument("--custom_weights", type=str, default=None,
                       help="Path to custom weight configurations JSON file")

    # Model architecture arguments (needed for loading)
    parser.add_argument("--arch", type=str, default="MLP",
                       choices=["MLP", "MLP_EHH", "Transformer"],
                       help="Model architecture")
    parser.add_argument("--embedding_dim", type=int, default=32,
                       help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden dimension")
    parser.add_argument("--n_hidden", type=int, default=2,
                       help="Number of hidden layers")
    parser.add_argument("--tied", action="store_true",
                       help="Whether to tie parameters")
    parser.add_argument("--subTB_lambda", type=float, default=0.9,
                       help="SubTB lambda parameter")
    parser.add_argument("--subTB_weighting", type=str, default="geometric_within",
                       help="SubTB weighting scheme")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logging.info(f"Using device: {device}")

    # Load configuration
    config = load_config(args.config_path)

    # Load trained model
    gflownet, sampler, env, preprocessor = load_trained_model(
        args.model_path, config, args, device
    )

    # Determine model type
    model_type = "conditional" if args.conditional else "unconditional"

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = f"{args.output_dir}_{timestamp}"

    # Run generalization tests
    logging.info("Starting generalization tests...")
    tester = run_generalization_tests(
        env=env,
        sampler=sampler,
        device=device,
        model_type=model_type,
        n_samples=args.n_samples,
        output_dir=output_dir
    )

    # Generate and display summary
    report_path = tester.generate_report(output_dir)
    logging.info(f"Generalization tests completed!")
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"Report generated: {report_path}")

    # Print summary statistics
    if tester.results:
        all_pareto_effs = [result['stats']['pareto_efficiency'] for result in tester.results.values()]
        all_diversities = [result['stats'].get('diversity_mean', 0) for result in tester.results.values()]

        print("\n" + "="*60)
        print("GENERALIZATION TEST SUMMARY")
        print("="*60)
        print(f"Model: {args.model_path}")
        print(f"Model Type: {model_type}")
        print(f"Configurations Tested: {len(tester.results)}")
        print(f"Samples per Configuration: {args.n_samples}")
        print(f"\nAverage Pareto Efficiency: {sum(all_pareto_effs)/len(all_pareto_effs):.3f}")
        print(f"Average Sequence Diversity: {sum(all_diversities)/len(all_diversities):.3f}")
        print(f"Best Pareto Efficiency: {max(all_pareto_effs):.3f}")
        print(f"Worst Pareto Efficiency: {min(all_pareto_effs):.3f}")
        print("="*60)


if __name__ == "__main__":
    main()

