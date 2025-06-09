import sys
import os

# Add the project root directory (demo) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import os
import wandb
import shutil
from datetime import datetime

# Assuming 'on_policy' is a package accessible in the Python path,
# and 'GymEnv.py' (containing GymEnv class) is in the same directory as this train.py.
from on_policy import OnPolicyRunner
from GymEnv import GymVecEnv # Relative import for GymEnv

def get_hopper_v5_train_cfg():
    train_cfg = {
        "runner": {
            "experiment_name": "HopperPPO_Run", # Specific to this run
            "num_steps_per_env": 2048,      # Standard for MuJoCo, good rollout length
            "save_interval": 200,           # Save model every N learning iterations
        },
        "algorithm": {
            "num_learning_epochs": 8,      # Number of epochs to update policy per rollout
            "num_mini_batches": 32,         # num_envs * num_steps_per_env / num_mini_batches
                                            # 16 * 2048 / 32 = 1024 (mini-batch size)
                                            # Or if num_envs is 8: 8 * 2048 / 32 = 512
            "clip_param": 0.2,              # PPO clipping parameter
            "gamma": 0.99,                  # Discount factor
            "lam": 0.95,                    # GAE lambda parameter
            "value_loss_coef": 0.5,
            "entropy_coef": 0.0,          # Small entropy bonus to encourage exploration
            "learning_rate": 3e-4,          # Common starting learning rate
            "max_grad_norm": 0.5,
            "use_clipped_value_loss": True,
            "schedule": 'fixed',           # Linear decay of learning rate is often good
                                            # (ensure your PPO impl. supports this)
                                            # If not, 'fixed' or 'adaptive' are alternatives.
            "desired_kl": 0.01,             # Target KL for adaptive LR schedule (if used)
        },
        "policy": {
            # "policy_name" can be added if your runner uses it for logging
            "actor_hidden_dims": [256, 256], # Good size for Hopper
            "critic_hidden_dims": [256, 256],# Good size for Hopper
            "init_noise_std": 0.5, # Initial standard deviation for action distribution
                                   # Adjust based on how ActorCritic uses this.
                                   # If actor outputs log_std, this might be an initial value for log_std.
            "activation": nn.Tanh(),       # Tanh is often preferred for MuJoCo stability
        },
        "env_name": "Hopper-v5",
        "num_envs": 16, # Number of parallel environments (adjust based on CPU cores)
                        # 8 to 16 is common
        "seed": 1017,                     # Or any other seed
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_learning_iterations": 1500, # Total PPO learning iterations
                                         # Total timesteps: 1500 * 16 * 2048 = ~49 million
                                         # Hopper typically needs millions of timesteps
        "log_dir_base": "./hopper_ppo_results"
    }
    return train_cfg

# --- Main test function ---
def main_test_runner():
    env_name = "Hopper-v5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Create wrapped environment
    vec_env = GymVecEnv(env_name, device=device)

    # 2. Get configuration
    train_cfg = get_hopper_v5_train_cfg()

    # 3. Create a temporary log directory
    # Use a fixed path for easier inspection if needed, or tempfile for auto-cleanup
    # log_dir = tempfile.mkdtemp()
    log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"{env_name}_test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to directory: {log_dir}")

    try:
        # 4. Instantiate OnPolicyRunner
        # Disable wandb for this test by setting mode if possible, or ensure it runs offline
        os.environ["WANDB_MODE"] = "disabled" # Prevent actual wandb runs

        runner = OnPolicyRunner(
            env=vec_env,
            cfg=train_cfg,
            log_dir=log_dir, # Provide log_dir to enable logging and saving
            device=device
        )

        # 5. Start learning
        num_learning_iterations = 1000
        print(f"Starting learning for {num_learning_iterations} iterations...")
        runner.train(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=False)
        print("Learning finished.")

        # 6. Test saving
        final_model_path = os.path.join(log_dir, "model_final_test.pt")
        runner.save(final_model_path)
        assert os.path.exists(final_model_path), "Model was not saved."
        print(f"Model saved to {final_model_path}")

        # 7. Test loading
        # Create a new runner instance for loading
        runner_loaded = OnPolicyRunner(
            env=vec_env, # Can re-use or re-create; re-using for simplicity
            cfg=train_cfg,
            log_dir=None, # Or a different log_dir if continuing logging
            device=device
        )
        runner_loaded.load(final_model_path, load_optimizer=True)
        assert runner_loaded.current_learning_iteration == runner.current_learning_iteration, "Loaded iteration mismatch"
        print(f"Model loaded from {final_model_path}, iteration: {runner_loaded.current_learning_iteration}")

        # 8. Test inference
        inference_policy = runner.get_inference_policy(device=device)
        obs, _ = vec_env.reset() # Get initial observation
        for _ in range(5): # Run a few inference steps
            with torch.no_grad():
                action = inference_policy(obs.to(device)) # Ensure obs is on correct device
            obs, _, _, _, _ = vec_env.step(action)
        print("Inference policy test completed.")

    finally:
        # Clean up
        if os.path.exists(log_dir) and log_dir == "temp_hopper_test_log": # Basic safety for fixed path
             shutil.rmtree(log_dir)
             print(f"Cleaned up log directory: {log_dir}")
        vec_env.close()
        if "WANDB_MODE" in os.environ:
            del os.environ["WANDB_MODE"]

    print(f"{env_name} OnPolicyRunner test case finished successfully.")

if __name__ == "__main__":
    main_test_runner()
