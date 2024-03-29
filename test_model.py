# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace

# Import the game
import gym_super_mario_bros
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# Import Gym
import gym

from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt
# Import os for file path management

import os 
# Import PPO for algos
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# 3. Load model & Test model
# 1. Changing render mode None to 'human'
env = gym.make('SuperMarioBros-v0',apply_api_compatibility=True, render_mode='human')
# 2. Simplify the controls 
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])

#model = PPO.load('DIR', env)
model = PPO.load('./train/best_model_1000000', env)
state = env.reset()

# Start the game 
state = env.reset()
# Loop through the game
while True: 
    
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
