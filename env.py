from gymnasium import Env
from gymnasium.spaces import MultiBinary, Box
import retro
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Creating custom enviroments
class StreetFighter(Env):
    def __init__(self):
        super.__init__
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis", use_restricted_actions = retro.Actions.FILTERED)
        # use_restricted_actions prevent to actions like up and down at the same time to be executed. 
    
    def step(self, action):
        obs, reward, done, truncated, info = self.game.step(action)
        obs = self.pre_process(obs)
        frame_delta = obs - self.previous_frame 
        self.previous_frame = obs
        reward = info["score"] - self.score
        self.score = info["score"]
        return frame_delta, reward, done, truncated, info
        
    def render(self):
        self.game.render()
        
    def reset(self, **kwargs):
        obs, info = self.game.reset()
        obs  = self.pre_process(obs)
        self.previous_frame = obs
        self.score = 0
        return obs, info
        
    def pre_process(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        channels = np.reshape(resize, (84, 84, 1))
        return channels
        
    def close(self):
        self.game.close()

