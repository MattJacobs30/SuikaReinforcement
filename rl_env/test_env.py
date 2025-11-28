import gymnasium as gym
import numpy as np
from suika_env import SuikaEnv
import time

def main():
    # Initialize environment
    # render_mode="human" to see the game playing
    env = SuikaEnv(render_mode="human")
    
    obs, info = env.reset()
    print("Environment reset. Initial score:", info['score'])
    
    done = False
    step_count = 0
    
    try:
        while not done:
            # Sample random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step_count}: Action={action}, Reward={reward}, Score={info['score']}, Game Over={terminated}")
            
            step_count += 1
            
            if terminated or truncated:
                print("Game Over or Truncated")
                done = True
                
            # Optional: slowdown to watch
            # time.sleep(0.1) 
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        env.close()

if __name__ == "__main__":
    main()

