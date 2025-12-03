import gymnasium as gym
import numpy as np
import pygame
from suika_env import SuikaEnv
from suika.part2.config import config
import time

def main():
    env = SuikaEnv(render_mode="human", action_type="continuous")
    
    running = True
    
    try:
        while running:
            obs, info = env.reset()
            print("\nEnvironment reset. Initial score:", info['score'])
            print("Click on the screen to drop the fruit.")
            print("Press ESC or close window to quit.")
            
            done = False
            
            while not done:
                action = None
                waiting_for_input = True
                
                while waiting_for_input:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True
                            running = False
                            waiting_for_input = False
                        
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                done = True
                                running = False
                                waiting_for_input = False
                        
                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            mouse_x, _ = pygame.mouse.get_pos()
                            
                            pad_width = config.pad.right - config.pad.left
                            clamped_x = np.clip(mouse_x, config.pad.left, config.pad.right)
                            
                            action_val = ((clamped_x - config.pad.left) / pad_width) * 2.0 - 1.0
                            action = np.array([action_val], dtype=np.float32)
                            
                            waiting_for_input = False

                    if waiting_for_input and running:
                        mouse_x, _ = pygame.mouse.get_pos()
                        env.cloud.curr.set_x(mouse_x)
                        env._draw_frame(wait_val=0)
                        env.clock.tick(60)

                if action is not None:
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    print(f"Action={action[0]:.2f}, Reward={reward}, Score={info['score']}, Game Over={terminated}")
                    
                    if terminated or truncated:
                        print("Game Over!")
                        time.sleep(2)
                        done = True
                
                if not running:
                    break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        env.close()

if __name__ == "__main__":
    main()
