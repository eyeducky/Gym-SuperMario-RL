# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace

# Import the game
import gym_super_mario_bros
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# Import Gym
import gym
# Import Torch (I think this for linux & 이거 안하면 cpu로 돌아감 다만 Cuda환경 구축 안되있으면 의미 없음)
import torch

# Setup game
env = gym.make('SuperMarioBros-v0',apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

"""
# This is just render test, not necessery to code - made by ducky

done = True
# Loop through each frame in the game (done = frame)
for step in range(1000): 
    if done: 
        # Start the gamee
        env.reset()
    # Do random actions
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    # Show the game on the screen
# Close the game
env.close()
"""

# 1. Preprocess Environment 
#  (gym(gym_ver-v0.26)이랑 stable-baselines3는 호환이 되는데, gym-super-mario-bros(gym_ver-v0.21)랑 
#  stable-baselines3가 서로 호환이 안되서 데이터를 가공해야됨)
# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt
# 1. Create the base environment
env = gym.make('SuperMarioBros-v0',apply_api_compatibility=True, render_mode=None)
# 2. Simplify the controls 
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

state = env.reset()


# 2. Train the RL Model
# Import os for file path management
import os 
# Import PPO for algos
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
# This is the AI model started
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, 
            n_steps=512) 
# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=1000000, callback=callback, progress_bar=True)
model.save('thisisatestmodel')



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
model = PPO.load('thisisatestmodel', env)
state = env.reset()

# Start the game 
state = env.reset()
# Loop through the game
while True: 
    
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()

