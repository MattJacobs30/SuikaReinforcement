import os
import argparse
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import time

from suika_env import SuikaEnv

def main():
    parser = argparse.ArgumentParser(description="Test a trained DQN model")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model zip file")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS for viewing")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic (random) actions instead of deterministic")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    print(f"Loading model from {model_path}...")

    # Environment settings MUST match training
    env_kwargs = {
        'render_mode': 'human', # We want to watch it
        'action_type': 'discrete',
        'discrete_bins': 15, 
        'max_fruits': 50
    }
    
    # Create single environment for testing
    env = SuikaEnv(**env_kwargs)
    
    # Load the agent
    model = DQN.load(model_path, env=env)

    print(f"Starting testing... (Deterministic: {not args.stochastic})")
    
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        print(f"Episode {ep+1} started.")
        
        while not (done or truncated):
            # Predict action
            action, _states = model.predict(obs, deterministic=not args.stochastic)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Simple FPS limit
            time.sleep(1.0 / args.fps)
            
            # Handle manual exit
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    truncated = True
        
        print(f"Episode {ep+1} finished. Score: {info['score']}, Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()
