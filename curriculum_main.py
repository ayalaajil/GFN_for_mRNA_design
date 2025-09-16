# curriculum_mrna_trainer.py
import sys
import os
import argparse
from datetime import datetime
import time
sys.path.insert(0, os.path.dirname(__file__))
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import scipy
from collections import deque
import wandb
from utils import set_seed
from reward import compute_simple_reward
from automatic_curriculum.auto_curri import make_dist_computer
from env import CodonDesignEnv
from main_conditional import build_subTB_gflownet, load_config
from preprocessor import CodonSequencePreprocessor2
from gfn.samplers import Sampler
from plots import plot_training_time_analysis, plot_task_distribution_analysis

class MRNDesignCurriculum:
    """Curriculum learning wrapper for mRNA design"""

    def __init__(self, tasks, curriculum_config=None):
        self.tasks = tasks
        self.device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_tasks = len(tasks)

        self.config = curriculum_config or {
            'lpe': 'Linreg',        # Learning Progress Estimator (Linreg works better than EMA in noisy rewards)
            'acp': 'MR',            # Attention Computer: Mastering Rate
            'a2d': 'Boltzmann',     # Attention-to-Distribution mapping
            'a2d_tau': 0.2,         # Temperature (slightly higher than 0.1 → encourages more exploration)
            'lpe_K': 20,            # Window size for progress estimation (larger → smoother updates)
            'acp_MR_K': 20,         # Window size for MR averaging
            'acp_MR_power': 4,      # Lower power than 6 (avoids over-sharpening task focus)
            'acp_MR_pot_prop': 0.6, # Slightly higher → emphasizes potential progress tasks
            'acp_MR_att_pred': 0.3, # Increase predecessor attention (stabilizes step-up transitions)
            'acp_MR_att_succ': 0.1, # Increase successor attention (gradual forward push)
        }


        self.protein_loader = ProteinDatasetLoader()

        self.G = self._create_curriculum_graph()

        self.init_min_perfs = [0.0] * self.num_tasks
        self.init_max_perfs = [1.0] * self.num_tasks

        self.compute_dist = make_dist_computer(
            self.num_tasks,
            lpe=self.config['lpe'],
            lpe_K=self.config['lpe_K'],
            acp=self.config['acp'],
            acp_MR_G=self.G,
            acp_MR_init_min_perfs=self.init_min_perfs,
            acp_MR_init_max_perfs=self.init_max_perfs,
            acp_MR_K=self.config['acp_MR_K'],
            acp_MR_power=self.config['acp_MR_power'],
            acp_MR_pot_prop=self.config['acp_MR_pot_prop'],
            acp_MR_att_pred=self.config['acp_MR_att_pred'],
            acp_MR_att_succ=self.config['acp_MR_att_succ'],
            a2d=self.config['a2d'],
            a2d_tau=self.config['a2d_tau']
        )

        self.dist = self.compute_dist({})

    def _create_curriculum_graph(self):
        """Create a linear curriculum graph for protein length progression"""
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_tasks))

        for i in range(self.num_tasks - 1):
            G.add_edge(i, i + 1)

        return G

    def sample_task(self):
        """Sample a task based on current curriculum distribution"""

        task_idx = np.random.choice(self.num_tasks, p=self.dist)
        return self.tasks[task_idx], task_idx

    def update_curriculum(self, task_performances):
        """Update curriculum based on task performances"""
        current_performances = {}
        for task_idx, perf_list in task_performances.items():
            if perf_list:
                current_performances[task_idx] = perf_list[-1][1]
            else:
                current_performances[task_idx] = 0.0

        self.dist = self.compute_dist(current_performances)

class ProteinDatasetLoader:
    """Loads and manages protein sequences from datasets for curriculum learning"""

    def __init__(self):
        self.protein_cache = {}
        self.length_index = {}
        self._load_datasets()

    def _load_datasets(self):
        """Load protein sequences from available training datasets"""
        import pandas as pd

        dataset_files = [
            'training_dataset_very_short.csv',
            'training_dataset_short.csv',
            'training_dataset_medium.csv',
            'training_dataset_long.csv',
            'training_dataset_very_long.csv',
            # 'training_dataset_all.csv'
        ]

        all_proteins = []

        for dataset_file in dataset_files:
            try:
                df = pd.read_csv(dataset_file)
                if 'protein_sequence' in df.columns and 'protein_length' in df.columns:

                    valid_proteins = df[
                        (df['protein_sequence'].str.len() >= 10) &
                        (df['protein_sequence'].str.match(r'^[ACDEFGHIKLMNPQRSTVWY\*]+$'))
                    ]

                    for _, row in valid_proteins.iterrows():
                        protein_seq = row['protein_sequence']
                        length = len(protein_seq)

                        if length not in self.length_index:
                            self.length_index[length] = []
                        self.length_index[length].append(protein_seq)
                        all_proteins.append(protein_seq)

                    print(f"Loaded {len(valid_proteins)} proteins from {dataset_file}")

            except FileNotFoundError:
                print(f"Warning: {dataset_file} not found, skipping...")
                continue
            except Exception as e:
                print(f"Warning: Error loading {dataset_file}: {e}")
                continue

        self.protein_cache = {seq: len(seq) for seq in all_proteins}

        print(f"Total proteins loaded: {len(all_proteins)}")
        print(f"Length range: {min(self.length_index.keys()) if self.length_index else 0} - {max(self.length_index.keys()) if self.length_index else 0}")

    def _is_valid_protein_sequence(self, protein_seq):
        """Check if protein sequence is valid for the environment"""
        # Check if all amino acids are in the codon table
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY*")
        return all(aa in valid_aas for aa in protein_seq) and len(protein_seq) > 0

    def sample_protein_by_length_range(self, min_length, max_length):
        """Sample a real protein sequence within the specified length range (inclusive)"""

        # Find all proteins within the length range
        valid_proteins = []
        for length in range(min_length, max_length + 1):
            if length in self.length_index:
                proteins_at_length = [p for p in self.length_index[length] if self._is_valid_protein_sequence(p)]
                valid_proteins.extend(proteins_at_length)

        if valid_proteins:
            return np.random.choice(valid_proteins)

        # If no proteins in range, find the closest available length
        available_lengths = sorted(self.length_index.keys())
        if not available_lengths:
            raise ValueError("No protein data available in the dataset")

        # Find closest length to the range
        closest_length = None
        min_distance = float('inf')

        for length in available_lengths:
            if length < min_length:
                distance = min_length - length
            elif length > max_length:
                distance = length - max_length
            else:
                distance = 0  # Within range

            if distance < min_distance:
                min_distance = distance
                closest_length = length

        if closest_length is not None:
            proteins_at_closest = [p for p in self.length_index[closest_length] if self._is_valid_protein_sequence(p)]
            if proteins_at_closest:
                print(f"Warning: No proteins in range [{min_length}, {max_length}], using closest length {closest_length}")
                return np.random.choice(proteins_at_closest)

        raise ValueError(f"No valid proteins found in range [{min_length}, {max_length}] or nearby lengths")


    def sample_protein_by_length(self, target_length, tolerance=5):
        """Sample a protein sequence of approximately the target length (legacy method)"""
        return self.sample_protein_by_length_range(
            max(1, target_length - tolerance),
            target_length + tolerance
        )

    def get_available_lengths(self):
        """Get list of available protein lengths in the dataset"""
        return sorted(self.length_index.keys())

    def get_length_distribution(self):
        """Get distribution of protein lengths"""
        return {length: len(proteins) for length, proteins in self.length_index.items()}

    def generate_protein_sequence(self, task):
        """Generate a protein sequence for the given task using real protein datasets"""
        if isinstance(task, dict):
            # Complexity-based task
            return self._generate_complexity_based_protein(task)
        elif isinstance(task, (list, tuple)) and len(task) == 2:
            # Length range task [min_length, max_length]
            min_length, max_length = task

            print(f"Generating protein sequence for length range {min_length}-{max_length}")
            return self.sample_protein_by_length_range(min_length, max_length)
        else:
            # Single length task (legacy support)
            return self.sample_protein_by_length(task)


def train_with_curriculum(args, config, curriculum_tasks):
    """Main training function with curriculum learning"""

    curriculum = MRNDesignCurriculum(curriculum_tasks)
    preprocessor = CodonSequencePreprocessor2(250, embedding_dim=args.embedding_dim, device=curriculum.device)
    dummy_env = CodonDesignEnv("A*", device=curriculum.device)
    gflownet, pf_estimator, pb_estimator = build_subTB_gflownet(dummy_env, preprocessor, args, lamda=args.subTB_lambda)

    gflownet = gflownet.to(curriculum.device)
    pf_estimator = pf_estimator.to(curriculum.device)
    pb_estimator = pb_estimator.to(curriculum.device)

    optimizer = torch.optim.Adam(gflownet.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )
    sampler = Sampler(estimator=gflownet.pf)

    total_steps = args.n_iterations
    eval_every = args.eval_every
    train_steps_per_task = args.train_steps_per_task

    global_step = 0
    task_performances = {}

    curriculum_tasks_tuples = [tuple(task) if isinstance(task, list) else task for task in curriculum_tasks]

    HISTORY_LEN = 200
    EMA_ALPHA = 0.05

    training_stats = {
        'total_loss': 0.0,
        'valid_steps': 0,
        'skipped_steps': 0,
        'task_counts': {task: 0 for task in curriculum_tasks_tuples},
        'best_performance': {task: 0.0 for task in curriculum_tasks_tuples},
        'current_lr': args.lr
    }

    training_stats['recent_losses'] = {task: deque(maxlen=HISTORY_LEN) for task in curriculum_tasks_tuples}
    training_stats['per_task_ema'] = {task: None for task in curriculum_tasks_tuples}
    training_stats['global_ema'] = None


    run_name = args.run_name or f"curriculum_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.wandb_project:

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **vars(args),
                **vars(config),
                'curriculum_tasks': curriculum_tasks,
                'total_steps': total_steps,
                'eval_every': eval_every,
                'train_steps_per_task': train_steps_per_task,
                'device': str(curriculum.device),
                'protein_loader_stats': curriculum.protein_loader.get_length_distribution()
            },
            tags=['curriculum_learning', 'mRNA_design', 'GFlowNet']
        )

    print("=" * 80)
    print("CURRICULUM LEARNING TRAINING STARTED")
    print("=" * 80)
    print(f"Total iterations: {total_steps}")
    print(f"Evaluation every: {eval_every} steps")
    print(f"Training steps per task: {train_steps_per_task}")
    print(f"Curriculum tasks: {curriculum_tasks}")
    print(f"Device: {curriculum.device}")
    print(f"Initial learning rate: {args.lr}")
    print("=" * 80)

    # Global time tracking
    time_start = time.time()
    training_phase_times = {
        'initialization': 0.0,
        'training': 0.0,
        'evaluation': 0.0,
        'total': 0.0
    }
    task_times = {task: 0.0 for task in curriculum_tasks_tuples}
    task_distribution_history = []
    time_per_iteration = []

    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    while global_step < total_steps:
        iteration_start_time = time.time()

        # Teacher samples a task
        current_task, current_task_idx = curriculum.sample_task()

        print(f"We are training on task: {current_task}")
        protein_seq = curriculum.protein_loader.generate_protein_sequence(current_task)
        print(f"Protein sequence: {protein_seq}")

        task_env = CodonDesignEnv(protein_seq=protein_seq, device=curriculum.device)

        current_task_key = tuple(current_task) if isinstance(current_task, list) else current_task
        training_stats['task_counts'][current_task_key] += 1

        # Record task distribution
        task_distribution_history.append({
            'step': global_step,
            'task': current_task_key,
            'task_idx': current_task_idx,
            'distribution': curriculum.dist.copy()
        })

        if isinstance(current_task, (list, tuple)) and len(current_task) == 2:
            task_display = f"Range [{current_task[0]}-{current_task[1]}]"
        else:
            task_display = f"Length {current_task}"

        print(f"\n{'='*60}")
        print(f"STEP {global_step + 1}/{total_steps} | TASK: {task_display} (Actual Length: {len(protein_seq)})")
        print(f"{'='*60}")
        print(f"Protein sequence: {protein_seq}")
        print(f"Task distribution: {curriculum.dist}")
        print(f"Task counts: {training_stats['task_counts']}")
        print(f"Current learning rate: {training_stats['current_lr']:.3f}")

        if args.wandb_project:

            task_counts_str = {str(k): v for k, v in training_stats['task_counts'].items()}

            wandb.log({
                'training/protein_length': len(protein_seq),
                'curriculum/task_counts': task_counts_str,
            })


        task_losses = []
        task_start_time = time.time()

        for step_in_task in  (pbar := tqdm(range(train_steps_per_task), dynamic_ncols=True)):

            try:
                print(f"Starting training GFlowNet on task {current_task} at step {step_in_task + 1}/{train_steps_per_task}....")

                # Sample weights
                weights = np.random.dirichlet([1, 1, 1])
                task_env.set_weights(weights)
                weights_tensor = torch.tensor(weights, dtype=torch.get_default_dtype(), device=curriculum.device)
                conditioning = weights_tensor.unsqueeze(0).expand(args.batch_size, *weights_tensor.shape)

                # Sample trajectories with conditioning
                trajectories = gflownet.sample_trajectories(
                    task_env,
                    n=args.batch_size,
                    conditioning=conditioning,
                    save_logprobs=True,
                    save_estimator_outputs=True,
                    epsilon=args.epsilon,
                )

                optimizer.zero_grad()
                loss = gflownet.loss_from_trajectories(
                    task_env, trajectories, recalculate_all_logprobs=False
                )
                loss.backward()
                optimizer.step()

                scheduler.step(loss)
                task_losses.append(loss.item())

                training_stats['total_loss'] += loss.item()
                training_stats['valid_steps'] += 1
                training_stats['current_lr'] = optimizer.param_groups[0]['lr']

                task_key = current_task_key
                training_stats['recent_losses'][task_key].append(loss.item())

                prev = training_stats['per_task_ema'][task_key]
                training_stats['per_task_ema'][task_key] = (
                    loss.item() if prev is None else EMA_ALPHA * loss.item() + (1 - EMA_ALPHA) * prev
                )

                gprev = training_stats.get('global_ema')
                training_stats['global_ema'] = (
                    loss.item() if gprev is None else EMA_ALPHA * loss.item() + (1 - EMA_ALPHA) * gprev
                )

                per_task_means = {
                    t: (np.mean(list(q)) if len(q) > 0 else None)
                    for t, q in training_stats['recent_losses'].items()
                }


                valid_means = [m for m in per_task_means.values() if m is not None]
                global_unweighted_mean = float(np.mean(valid_means)) if valid_means else None


                if 'prev_per_task_ema_snapshot' not in training_stats:
                    training_stats['prev_per_task_ema_snapshot'] = training_stats['per_task_ema'].copy()


                per_task_lp = {}

                for t in per_task_means:
                    cur = training_stats['per_task_ema'][t]
                    prev = training_stats['prev_per_task_ema_snapshot'].get(t)
                    if cur is None or prev is None:
                        per_task_lp[t] = None
                    else:
                        per_task_lp[t] = prev - cur

                print(f" Step {step_in_task + 1}/{train_steps_per_task}: Loss = {loss.item():.6f}")

            except Exception as e:
                print(f"Error during training step {global_step}: {e}")
                training_stats['skipped_steps'] += 1
                continue

            # Track task completion time
            task_end_time = time.time()
            task_duration = task_end_time - task_start_time
            task_times[current_task_key] += task_duration
            training_phase_times['training'] += task_duration

            if args.wandb_project:

                wandb.log({
                    'training/global_unweighted_mean_loss': global_unweighted_mean,
                    'training/global_ema_loss': training_stats['global_ema'],
                    'training/step': global_step,
                    'training/elapsed_time': time.time() - time_start,

                    **{f'task/{t}_mean_loss': (m if m is not None else -1) for t, m in per_task_means.items()},
                    **{f'task/{t}_ema_loss': (training_stats['per_task_ema'][t] if training_stats['per_task_ema'][t] is not None else -1) for t in per_task_means},
                    **{f'task/{t}_lp': (per_task_lp[t] if per_task_lp[t] is not None else 0) for t in per_task_means}
                })


        if global_step % eval_every == 0:
                eval_start_time = time.time()
                eval_log_data = {'global_step': global_step}

                print(f"\n EVALUATION at step {global_step+1}")
                print("-" * 40)

                # Evaluate ALL tasks
                for task_idx, task in enumerate(curriculum.tasks):

                    eval_protein = curriculum.protein_loader.generate_protein_sequence(task)
                    eval_env = CodonDesignEnv(protein_seq=eval_protein, device=curriculum.device)

                    performance = evaluate_task_performance(gflownet, eval_env, sampler, n_samples=16)
                    task_performances.setdefault(task_idx, []).append((global_step, performance))

                    task_key = curriculum_tasks_tuples[task_idx]
                    if performance > training_stats['best_performance'][task_key]:
                        training_stats['best_performance'][task_key] = performance

                    eval_log_data[f'eval/perf_task_{task_idx}'] = performance
                    eval_log_data[f'eval/best_perf_task_{task_idx}'] = training_stats['best_performance'][task_key]

                    print(f"Task {task_idx} ({task}): Performance = {performance:.4f} (Best: {training_stats['best_performance'][task_key]:.4f})")

                curriculum.update_curriculum(task_performances)
                new_dist = curriculum.dist

                entropy = scipy.stats.entropy(new_dist, base=2)
                eval_log_data['curriculum/distribution_entropy'] = entropy

                for i, prob in enumerate(new_dist):
                    eval_log_data[f'curriculum/dist_task_{i}'] = prob

                avg_task_loss = sum(task_losses) / len(task_losses) if task_losses else 0.0
                eval_log_data['eval/task_avg_loss'] = avg_task_loss
                eval_log_data['training/learning_rate'] = optimizer.param_groups[0]['lr']

                # Track evaluation time
                eval_end_time = time.time()
                eval_duration = eval_end_time - eval_start_time
                training_phase_times['evaluation'] += eval_duration
                eval_log_data['eval/evaluation_duration'] = eval_duration

                if args.wandb_project:
                    wandb.log(eval_log_data)

        if task_losses:
            avg_loss = sum(task_losses) / len(task_losses)
            print(f"Task {current_task} completed - Avg loss: {avg_loss:.6f}")
        else:
            print(f"Task {current_task} completed - No valid steps")

        # Track iteration time
        iteration_end_time = time.time()
        iteration_duration = iteration_end_time - iteration_start_time
        time_per_iteration.append(iteration_duration)

        global_step += 1

    total_time = time.time() - time_start
    training_phase_times['total'] = total_time
    training_phase_times['initialization'] = time_start - time_start

    print(f"Training completed in {total_time:.2f} seconds.")
    print(f"Training phase breakdown:")
    print(f"  - Training: {training_phase_times['training']:.2f}s ({training_phase_times['training']/total_time*100:.1f}%)")
    print(f"  - Evaluation: {training_phase_times['evaluation']:.2f}s ({training_phase_times['evaluation']/total_time*100:.1f}%)")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED - FINAL SUMMARY")
    print("=" * 80)
    print(f"Total steps completed: {global_step}/{total_steps}")
    print("\nTask Statistics:")

    for i, task in enumerate(curriculum_tasks):

        task_key = curriculum_tasks_tuples[i]
        count = training_stats['task_counts'][task_key]
        best_perf = training_stats['best_performance'][task_key]
        task_time = task_times[task_key]
        avg_time_per_task = task_time / count if count > 0 else 0
        if isinstance(task, (list, tuple)) and len(task) == 2:
            task_display = f"Range [{task[0]}-{task[1]}]"
        else:
            task_display = f"Length {task}"
        print(f"  Task {task_display}: {count} times, Best performance: {best_perf:.4f}, Total time: {task_time:.2f}s, Avg time: {avg_time_per_task:.2f}s")

    print("\nFinal Curriculum Distribution:")
    for i, prob in enumerate(curriculum.dist):
        task = curriculum_tasks[i]
        if isinstance(task, (list, tuple)) and len(task) == 2:
            task_display = f"Range [{task[0]}-{task[1]}]"
        else:
            task_display = f"Length {task}"
        print(f"  Task {task_display}: {prob:.4f}")
    print("=" * 80)

    print("\nCreating time and task distribution plots...")
    plot_training_time_analysis(task_distribution_history, training_phase_times, task_times, time_per_iteration, output_dir)
    plot_task_distribution_analysis(task_distribution_history, curriculum_tasks, output_dir)

    if args.wandb_project:

        task_counts_str = {str(k): v for k, v in training_stats['task_counts'].items()}
        best_performances_str = {str(k): v for k, v in training_stats['best_performance'].items()}

        final_metrics = {
            'final/total_steps': global_step,
            'final/success_rate': training_stats['valid_steps']/(training_stats['valid_steps'] + training_stats['skipped_steps']) if (training_stats['valid_steps'] + training_stats['skipped_steps']) > 0 else 0,
            'final/learning_rate': training_stats['current_lr'],
            'final/overall_avg_loss': training_stats['total_loss']/training_stats['valid_steps'] if training_stats['valid_steps'] > 0 else 0,
            'final/task_counts': task_counts_str,
            'final/best_performances': best_performances_str,
            'final/total_training_time': total_time,
            'final/training_phase_time': training_phase_times['training'],
            'final/evaluation_phase_time': training_phase_times['evaluation'],
            'final/avg_time_per_iteration': np.mean(time_per_iteration) if time_per_iteration else 0,
            'final/std_time_per_iteration': np.std(time_per_iteration) if time_per_iteration else 0
        }

        for i, task in enumerate(curriculum_tasks):
            task_key = curriculum_tasks_tuples[i]
            final_metrics[f'final/task_{i}_count'] = training_stats['task_counts'][task_key]
            final_metrics[f'final/task_{i}_best_performance'] = training_stats['best_performance'][task_key]
            final_metrics[f'final/task_{i}_total_time'] = task_times[task_key]
            final_metrics[f'final/task_{i}_avg_time'] = task_times[task_key] / training_stats['task_counts'][task_key] if training_stats['task_counts'][task_key] > 0 else 0

        wandb.log({**final_metrics,
        'final/distribution_entropy': float(scipy.stats.entropy(curriculum.dist, base=2)),
        'final/training_time_analysis': wandb.Image(f"{output_dir}/training_time_analysis.png"),
        'final/task_distribution_analysis': wandb.Image(f"{output_dir}/task_distribution_analysis.png"),
        })

        wandb.summary['best_overall_performance'] = max(training_stats['best_performance'].values())
        wandb.finish()


    print("Saving final curriculum-trained model...")
    output_dir = f"curriculum_model/{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "final_curriculum_gflownet.pth")

    torch.save({
        'model_state_dict': gflownet.state_dict(),
        'args': args,
    }, model_save_path)

    print(f"Model saved to {model_save_path}")


def evaluate_task_performance(gflownet, env, sampler, n_samples=32):
    """Evaluate GFlowNet performance on a specific task"""

    with torch.no_grad():

        total_reward = 0
        device = env.device

        weights = torch.tensor([0.3, 0.3, 0.4], device=device)
        conditioning = weights.unsqueeze(0).expand(n_samples, *weights.shape)

        trajectories = sampler.sample_trajectories(
            env,
            n=n_samples,
            conditioning=conditioning
        )

        final_states = trajectories.terminating_states.tensor

        rewards = []
        components_list  = []
        valid_samples = 0

        for state in final_states:
                    reward, components = compute_simple_reward(state, env.codon_gc_counts, weights)
                    total_reward += reward
                    rewards.append(reward)
                    components_list .append(components)
                    valid_samples += 1

        avg_reward = total_reward / valid_samples if valid_samples > 0 else 0.0
        avg_gc = sum(components[0] for components in components_list) / valid_samples if valid_samples > 0 else 0.0
        avg_mfe = sum(components[1] for components in components_list) / valid_samples if valid_samples > 0 else 0.0
        avg_cai = sum(components[2] for components in components_list) / valid_samples if valid_samples > 0 else 0.0

        if rewards:
            min_reward = min(rewards)
            max_reward = max(rewards)
            std_reward = (sum((r - avg_reward)**2 for r in rewards) / len(rewards))**0.5
            print(f" Reward stats: avg={avg_reward:.4f}, min={min_reward:.4f}, max={max_reward:.4f}, std={std_reward:.4f}")
            print(f" GC stats: avg={avg_gc:.4f}")
            print(f" MFE stats: avg={avg_mfe:.4f}")
            print(f" CAI stats: avg={avg_cai:.4f}")

            print(f" Valid samples: {valid_samples}/{n_samples} ({valid_samples/n_samples*100:.1f}%)")

        return avg_reward


def parse_curriculum_tasks(task_strings):
    """Parse curriculum task strings into appropriate format"""
    parsed_tasks = []
    for task_str in task_strings:
        if task_str.startswith('[') and task_str.endswith(']'):

            try:
                range_str = task_str[1:-1]
                min_val, max_val = map(int, range_str.split(','))
                parsed_tasks.append([min_val, max_val])
            except ValueError:
                raise ValueError(f"Invalid range format: {task_str}. Expected format: [min,max]")
        else:

            try:
                parsed_tasks.append(int(task_str))
            except ValueError:
                raise ValueError(f"Invalid task format: {task_str}. Expected integer or [min,max] range")
    return parsed_tasks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument('--curriculum_tasks', nargs='+', type=str,
                       default=['[25,40]', '[45,60]', '[65,80]', '[85,120]', '[125,180]'],
                       help='Protein length ranges for curriculum learning (format: [min,max])')

    parser.add_argument('--curriculum_type', type=str, default='length',
                       choices=['length', 'complexity'],
                       help='Type of curriculum: length-based or complexity-based')

    parser.add_argument('--n_iterations', type=int, default=10)    #50
    parser.add_argument('--eval_every', type=int, default=5)    #5
    parser.add_argument('--train_steps_per_task', type=int, default=200)  #200   # total training steps are n_iterations * train_steps_per_task = 100 * 100 = 10000

    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=4)
    parser.add_argument('--subTB_lambda', type=float, default=0.9)
    parser.add_argument('--arch', type=str, default='Transformer')
    parser.add_argument('--tied', action='store_true')

    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--lr_logz', type=float, default=1e-1)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--subTB_weighting', type=str, default="geometric_within")
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--top_n', type=int, default=50)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epsilon', type=float, default=0.25)

    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--save_progress', action='store_true', default=True, help='Save training progress to files')

    parser.add_argument("--config_path", type=str, default="config.yaml")

    parser.add_argument('--wandb_project', type=str, default='mRNA_GFN_Experiments_CL_FINAL')


    args = parser.parse_args()
    config = load_config(args.config_path)
    set_seed(args.seed)

    if args.curriculum_type == 'length':
        curriculum_tasks = parse_curriculum_tasks(args.curriculum_tasks)
        print(len(curriculum_tasks))

    print(f"Parsed curriculum tasks: {curriculum_tasks}")
    train_with_curriculum(args, config, curriculum_tasks)

if __name__ == "__main__":
    main()