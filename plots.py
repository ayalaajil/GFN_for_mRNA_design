import matplotlib.pyplot as plt
import torch

def plot_training_curves(loss_history, reward_components, out_path="training_curves.png"):
    reward_tensor = torch.tensor(reward_components)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 4, 1)
    plt.plot(loss_history)
    plt.title("Loss")

    for i, (label, color) in enumerate(zip(["GC", "MFE", "CAI"], ['green', 'blue', 'orange'])):
        plt.subplot(1, 4, i + 2)
        plt.plot(reward_tensor[:, i], label=label, color=color)
        plt.title(f"{label} Evolution")
        plt.xlabel("Iteration")
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)

