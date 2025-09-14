# training_with_teacher.py
from teacher_student import TeacherStudent
from utils import evaluate_task
import torch
import time
from env import CodonDesignEnv
import random


def env_factory(protein_seq, device):
    return CodonDesignEnv(protein_seq=protein_seq, device=device)

def generate_random_protein(length):
    # standard 20 AAs
    AAS = list("ACDEFGHIKLMNPQRSTVWY")
    return ''.join(random.choice(AAS) for _ in range(length))


def evaluate_task(env_factory, gflownet, aa_length, n_rollouts=128, batch_size=32, device=torch.device('cpu')):
    """
    env_factory: callable(protein_seq, device) -> CodonDesignEnv
    gflownet: your GFlowNet object which must support a sampling function that
              returns actions/state sequences or direct samples of codon indices.
              Replace `gflownet.sample_batch(env, batch_size)` with your API.
    """
    # create env for this task
    protein = generate_random_protein(aa_length)
    env = env_factory(protein, device)
    rewards = []

    # We'll assume gflownet has a method `sample_trajectories(env, n)` that returns states
    # If not, replace below with the function that produces completed states (codon indices)
    produced = 0
    while produced < n_rollouts:
        cur_batch = min(batch_size, n_rollouts - produced)

        # -----------------------
        # --- ADAPT THIS BLOCK ---
        # You must produce `cur_batch` finished states (DiscreteStates) or
        # directly codon-index tensors shaped (cur_batch, seq_length).
        # e.g. samples = gflownet.sample_batch(env, n=cur_batch)
        # where samples can be a torch tensor of codon indices or a States object.
        # -----------------------
        samples = gflownet.sample_batch(env, n=cur_batch)  # <-- replace with your method
        # samples should be either:
        #   - torch.LongTensor (cur_batch, seq_length) with -1 for empty; OR
        #   - a States-like object with `.tensor`
        if hasattr(samples, 'tensor'):
            states_tensor = samples.tensor
        else:
            states_tensor = samples  # assume tensor

        # compute reward via your env.reward
        r = env.reward(type('S', (), {'tensor': states_tensor}))
        rewards.append(r.detach().cpu().numpy())
        produced += cur_batch

    rewards = np.concatenate(rewards, axis=0)[:n_rollouts]
    return float(rewards.mean())

    

# tasks: protein lengths
tasks = [10, 30, 50, 70, 100, 150, 200, 250]
teacher = TeacherStudent(tasks, window_size=40, min_prob=1e-3, eps=1e-6, seed=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparams
steps = 20000
train_steps_per_pick = 50   # how many gfn update steps per selected task
eval_every = 200            # update teacher every `eval_every` training steps with perf calc
n_eval_rollouts = 64

# main loop
global_step = 0
while global_step < steps:
    # teacher picks a task
    aa_length = teacher.sample_task()
    # create a protein instantiation (can be random or from dataset)
    protein = generate_random_protein(aa_length)
    env = env_factory(protein, device)

    # train gflownet on this env for a number of internal updates / batches
    for _ in range(train_steps_per_pick):
        loss = gflownet.train_step(env)   # <- adapt this to your train_step
        global_step += 1

        # do periodic evaluation + teacher update
        if global_step % eval_every == 0:
            perf = evaluate_task(env_factory, gflownet, aa_length, n_rollouts=n_eval_rollouts, batch_size=32, device=device)
            teacher.update(aa_length, perf)

    # optional logging
    if global_step % (10 * eval_every) == 0:
        print(f"[{time.strftime('%H:%M:%S')}] step {global_step} task {aa_length} loss {loss:.4f}")
        print("Teacher state:", teacher.state())
