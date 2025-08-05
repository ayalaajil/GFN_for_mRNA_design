import optuna
import argparse
from main import main, load_config


def objective(trial):

    args = argparse.Namespace(
        # Tunable hyperparameters
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        lr_logz=trial.suggest_float("lr_logz", 1e-3, 1.0, log=True),
        batch_size=trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32]),
        epsilon=trial.suggest_float("epsilon", 0.0, 0.5),
        embedding_dim=trial.suggest_categorical("embedding_dim", [16, 32, 64]),
        hidden_dim=trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        n_hidden=trial.suggest_int("n_hidden", 1, 3),
        tied=trial.suggest_categorical("tied", [True, False]),
        # Fixed arguments
        no_cuda=False,
        seed=42,
        clip_grad_norm=1.0,
        lr_patience=10,
        config_path="config.yaml",
        n_iterations=200,
        n_samples=100,
    )

    config = load_config(args.config_path)
    eval_avg_reward = main(args, config)

    if hasattr(eval_avg_reward, "item"):
        return float(eval_avg_reward.item())

    return float(eval_avg_reward)


if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best value:", study.best_value)
    print("Best hyperparameters:", study.best_params)




# Best value: 68.4481430053711
# Best hyperparameters: {'lr': 0.0005876783399995771, 'lr_logz': 0.010288695676294389, 'batch_size': 2, 'epsilon': 0.3346584155876452, 'embedding_dim': 64, 'hidden_dim': 256, 'n_hidden': 3, 'tied': False}
