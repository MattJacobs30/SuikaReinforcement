import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import pymunk

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from suika.part2.config import config, CollisionTypes
    from suika.part2.cloud import Cloud
    from suika.part2.wall import Wall
    from suika.part2.particle import Particle
    from suika.part2.collision import collide
    from suika.part2.text import score as draw_score
    from suika.part2.text import gameover as draw_gameover
except ImportError as e:
    raise ImportError(f"Could not import game modules. Make sure you are running from the project root or have set PYTHONPATH correctly. Error: {e}")

class SuikaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": config.screen.fps}

    def __init__(self, render_mode=None, action_type="continuous", discrete_bins=128, max_fruits=50):
        self.render_mode = render_mode
        self.action_type = action_type
        self.discrete_bins = discrete_bins
        self.max_fruits = max_fruits 
        
        self.last_action = None
        self.repeat_count = 0
        
        pygame.init()
        pygame.display.init()
        
        self.screen_width = config.screen.width
        self.screen_height = config.screen.height
        
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Suika RL Environment")
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

        self.clock = pygame.time.Clock()

        if self.action_type == "discrete":
            self.action_space = spaces.Discrete(self.discrete_bins)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        obs_len = 9 + (self.max_fruits * 4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32
        )

        self.space = None
        self.walls = None
        self.cloud = None
        self.handler = None
        self.game_over = False
        self.game_over_timer = 0
        self.game_over_threshold = 3.0
        
    def _normalize(self, val, max_val):
        return val / max_val

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        W = float(self.screen_width)
        H = float(self.screen_height)
        MAX_TYPE = 11.0 
        MAX_RADIUS = 150.0 
        
        obs[0] = self.cloud.curr.n / MAX_TYPE
        obs[1] = self.cloud.curr.radius / MAX_RADIUS
        
        obs[2] = self.cloud.next.n / MAX_TYPE
        obs[3] = self.cloud.next.radius / MAX_RADIUS
        
        obs[4] = config.pad.left / W
        obs[5] = config.pad.right / W
        obs[6] = config.pad.bot / H 
        obs[7] = config.pad.killy / H 
        
        fruits = []
        min_y = H 
        
        for p in self.space.shapes:
            if isinstance(p, Particle) and p.alive:
                fruits.append(p)
                if (p.pos[1] - p.radius) < min_y:
                    min_y = p.pos[1] - p.radius
        
        obs[8] = min_y / H
        
        fruits.sort(key=lambda p: (p.pos[1], p.pos[0]))
        
        idx = 9
        fruit_count = 0
        
        for p in fruits:
            if fruit_count >= self.max_fruits:
                break
                
            obs[idx] = p.n / MAX_TYPE
            obs[idx+1] = p.pos[0] / W
            obs[idx+2] = p.pos[1] / H
            obs[idx+3] = p.radius / MAX_RADIUS
            
            idx += 4
            fruit_count += 1
                
        return obs

    def _get_info(self):
        return {
            "score": self.handler.data["score"] if self.handler else 0,
            "game_over": self.game_over
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.last_action = None
        self.repeat_count = 0

        self.space = pymunk.Space()
        self.space.gravity = (0, config.physics.gravity)
        self.space.damping = config.physics.damping
        self.space.collision_bias = config.physics.bias

        left = Wall(config.top_left, config.bot_left, self.space)
        bottom = Wall(config.bot_left, config.bot_right, self.space)
        right = Wall(config.bot_right, config.top_right, self.space)
        self.walls = [left, bottom, right]

        self.cloud = Cloud()
        
        do_random_start = True
        if options and "random_start" in options:
            do_random_start = options["random_start"]
            
        if do_random_start:
            rng = np.random.default_rng(seed)
            num_random = rng.integers(3, 9) 
            
            for _ in range(num_random):
                x_pos = rng.uniform(config.pad.left + 20, config.pad.right - 20)
                n_type = rng.integers(0, 6)
                
                p = Particle((x_pos, config.pad.top), n_type, self.space)
                
                for _ in range(30):
                    self.space.step(1/config.screen.fps)

        self.handler = self.space.add_collision_handler(CollisionTypes.PARTICLE, CollisionTypes.PARTICLE)
        self.handler.begin = collide
        self.handler.data["score"] = 0

        self.game_over = False
        self.game_over_timer = 0
        
        if self.render_mode == "human":
            self._draw_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_obs(), 0, True, False, self._get_info()

        act_val = 0.0
        if self.action_type == "discrete":
            bin_idx = action 
            act_val = -1.0 + (bin_idx / (self.discrete_bins - 1)) * 2.0
        else:
            act_val = np.clip(action[0], -1.0, 1.0)
        
        pad_width = config.pad.right - config.pad.left
        target_x = config.pad.left + (act_val + 1.0) * 0.5 * pad_width
        
        self.cloud.curr.set_x(int(target_x))
        
        self.cloud.release(self.space)
        
        steps_to_sim = 120 
        
        reward = 0
        initial_score = self.handler.data["score"]

        for i in range(steps_to_sim):
            if self.render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        return self._get_obs(), 0, True, False, self._get_info()

            if i == steps_to_sim - 1:
                self.cloud.step()

            self.space.step(1/config.screen.fps)
            
            any_over = False
            for p in self.space.shapes:
                if isinstance(p, Particle):
                    if p.has_collided:
                         bottom_y = p.pos[1] + p.radius
                         if bottom_y < config.pad.killy:
                             any_over = True
                             break
            
            if any_over:
                self.game_over_timer += (1/config.screen.fps)
                if self.game_over_timer > self.game_over_threshold:
                    self.game_over = True
            else:
                self.game_over_timer = 0
            
            if self.game_over:
                break
                
            if self.render_mode == "human":
                self._draw_frame(wait_val=steps_to_sim - i)
                self.clock.tick(config.screen.fps)

        final_score = self.handler.data["score"]
        step_reward = final_score - initial_score
        
        reward = step_reward + 0.25
        
        if self.action_type == "discrete":
            if self.last_action is not None and action == self.last_action:
                self.repeat_count += 1
            else:
                self.repeat_count = 0
                self.last_action = action
            
            if self.repeat_count > 2:
                reward -= 1 
        
        terminated = self.game_over
        
        if terminated:
            reward -= 100.0 
            
        truncated = False 
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _draw_frame(self, wait_val=0):
        self.screen.blit(config.background_blit, (0, 0))
        
        self.cloud.draw(self.screen, wait_val)
        
        for p in self.space.shapes:
            if isinstance(p, Particle):
                p.draw(self.screen)
        
        draw_score(self.handler.data['score'], self.screen)
        
        if self.game_over:
            draw_gameover(self.screen)

        pygame.display.update()

    def close(self):
        pygame.quit()
