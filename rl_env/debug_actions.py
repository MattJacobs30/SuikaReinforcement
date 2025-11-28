import time
import numpy as np
from suika_env import SuikaEnv

def main():
    print("Initializing environment...")
    # Discrete mode with 50 bins
    env = SuikaEnv(render_mode="human", action_type="discrete", discrete_bins=50)
    env.reset()
    
    # Test actions across the range
    test_actions = [0, 12, 25, 37, 49]
    
    print("\nTesting specific actions to verify cloud movement...")
    
    for action in test_actions:
        print(f"\n--- Testing Action: {action} ---")
        
        # We need to reset or just wait for the next drop.
        # Let's manually set the cloud phase to allow a drop immediately if we want,
        # but stepping naturally is better.
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Access internal cloud x to verify
        current_x = env.cloud.curr.x
        print(f"Action {action} -> Cloud X position: {current_x}")
        
        # Verify against expected bounds
        # We need to import config to check bounds, but let's just look at the numbers
        # Left is ~415, Right is ~863
        
        time.sleep(1.0) # Pause to see it
        
        if terminated or truncated:
            env.reset()

    env.close()
    print("\nTest finished.")

if __name__ == "__main__":
    main()

