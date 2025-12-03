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

    log_dir = "logs_dqn/"
    os.makedirs(log_dir, exist_ok=True)

    env_kwargs = {
        'render_mode': 'rgb_array',
        'action_type': 'discrete',
        'discrete_bins': 128,
        'max_fruits': 50
    }
    
    vec_env = make_vec_env(lambda: SuikaEnv(**env_kwargs), n_envs=1)

    model = None
    if args.model:
        if os.path.exists(args.model) or os.path.exists(args.model + ".zip"):
            print(f"Loading existing model from {args.model}...")
            custom_objects = {
                "learning_rate": 5e-5,
                "exploration_initial_eps": 0.1,
                "exploration_final_eps": 0.02,
                "exploration_fraction": 0.05
            }
            model = DQN.load(args.model, env=vec_env, tensorboard_log=log_dir, custom_objects=custom_objects)
            
            model.learning_rate = 5e-5
            model.exploration_initial_eps = 0.1
            model.exploration_final_eps = 0.02
            model.exploration_fraction = 0.05
        else:
            print(f"Error: Model path '{args.model}' not found. Starting fresh.")
    
    if model is None:
        print("Starting fresh model...")
        model = DQN(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        tensorboard_log=log_dir,
        buffer_size=1000000, 
        learning_starts=50000,
        target_update_interval=10000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1, 
        exploration_final_eps=0.02,
        learning_rate=1e-4,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    print("Starting/Continuing training with DQN (MlpPolicy - Features)...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path='./models_dqn/',
        name_prefix='suika_dqn_mlp'
    )

    model.learn(total_timesteps=10000000, callback=checkpoint_callback)

    model.save("suika_dqn_mlp_final")
    print("Training finished and model saved.")

    vec_env.close()

if __name__ == "__main__":
    main()
