import os
import argparse
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from suika_env import SuikaEnv

def main():
    parser = argparse.ArgumentParser(description="Train DQN agent")
    parser.add_argument("--model", type=str, help="Path to existing model to continue training (optional)")
    args = parser.parse_args()

    # Create log dir
    log_dir = "logs_dqn/"
    os.makedirs(log_dir, exist_ok=True)

    # Instantiate the env
    env_kwargs = {
        'render_mode': 'rgb_array', # Keeps backend fast (no window)
        'action_type': 'discrete',
        'discrete_bins': 15,
        'max_fruits': 50 # Limit features to 50 fruits
    }
    
    # Create the environment
    vec_env = make_vec_env(lambda: SuikaEnv(**env_kwargs), n_envs=1)

    model = None
    if args.model:
        if os.path.exists(args.model) or os.path.exists(args.model + ".zip"):
            print(f"Loading existing model from {args.model}...")
            model = DQN.load(args.model, env=vec_env, tensorboard_log=log_dir)
        else:
            print(f"Error: Model path '{args.model}' not found. Starting fresh.")
    
    if model is None:
        print("Starting fresh model...")
        # Initialize the agent
        # We switch from "CnnPolicy" to "MlpPolicy" because inputs are now features (numbers), not images
        model = DQN(
            "MlpPolicy", 
            vec_env, 
            verbose=1, 
            tensorboard_log=log_dir,
            buffer_size=100000, 
            learning_starts=5000,
            target_update_interval=2000,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.3, 
            exploration_final_eps=0.10,
            learning_rate=1e-4, 
        )

    # Train the agent
    print("Starting/Continuing training with DQN (MlpPolicy - Features)...")
    
    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path='./models_dqn/',
        name_prefix='suika_dqn_mlp'
    )

    # Train for some number of timesteps
    model.learn(total_timesteps=5000000, callback=checkpoint_callback)

    # Save the final model
    model.save("suika_dqn_mlp_final")
    print("Training finished and model saved.")

    # Close the environment
    vec_env.close()

if __name__ == "__main__":
    main()
