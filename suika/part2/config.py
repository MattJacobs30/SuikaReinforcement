import pygame
import yaml
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

class CollisionTypes:
    PARTICLE = 1
    WALL = 2


class ConfigNode:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)


class Config:
    def __init__(self):
        with open(os.path.join(BASE_DIR, "config.yaml"), "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        self.screen = ConfigNode(**self.config["screen"])
        self.pad = ConfigNode(**self.config["pad"])
        self.physics = ConfigNode(**self.config["physics"])

        self.fruit_names = ["cherry", "strawberry", "grapes", "orange",
                            "persimmon", "apple", "pear", "peach", "pineapple",
                            "melon", "watermelon"]

        blits_dir = os.path.join(PARENT_DIR, "blits")
        
        try:
            self.background_blit = pygame.image.load(os.path.join(blits_dir, "background.png"))
            self.cloud_blit = pygame.image.load(os.path.join(blits_dir, "cloud.png"))
    
            for name in self.fruit_names:
                self.config[name]["blit"] = pygame.transform.scale(
                    pygame.image.load(os.path.join(blits_dir, f"{name}.png")),
                    size=self.config[name]["size"],
                )
        except pygame.error:
            self.background_blit = pygame.Surface((self.screen.width, self.screen.height))
            self.cloud_blit = pygame.Surface((50, 50)) 
            for name in self.fruit_names:
                self.config[name]["blit"] = pygame.Surface(self.config[name]["size"])

        self.screen_center = (self.screen.width // 2, self.screen.height // 2)

    def __getitem__(self, key):
        index, field = key
        fruit = self.fruit_names[index]
        return self.config[fruit][field]

    @property
    def top_left(self):
        return self.pad.left, self.pad.top

    @property
    def bot_left(self):
        return self.pad.left, self.pad.bot

    @property
    def top_right(self):
        return self.pad.right, self.pad.top

    @property
    def bot_right(self):
        return self.pad.right, self.pad.bot


config = Config()
