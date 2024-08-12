# RL algorithm
from typing import Dict
from ray.rllib.algorithms.ppo import PPOConfig

import ray
import csv

import gymnasium as gym


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy

ray.init()

from gymnasium.envs.registration import register

register(
     id="case2env",
     entry_point="case_2_env:Case2Env",
     max_episode_steps=100,
)

env = gym.make('case2env')

# csv file to store average network utilizations
with open("utils_ppo_case2.csv", mode='w', newline='') as file:
  writer = csv.writer(file)
  writer.writerow(['Episode ID', 'Network Utilization'])

# callback to get total network utilizations after every training iteration for PPO
class MyPPOCallbacks(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):

        episode_info = episode.last_info_for()
        if episode_info is not None:
            network_utilization = episode_info.get('total_average_network_utilizations', 0)
            with open("utils_ppo_case2.csv", mode='a', newline='') as file:
              writer = csv.writer(file)
              writer.writerow([episode.episode_id, network_utilization])

config = (PPOConfig()
          .training(gamma=0.999, lr=0.008)
          .environment(env='case2env')
          .callbacks(MyPPOCallbacks)
          .resources(num_gpus=0)
          .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
        )

algo = config.build()

blconfig = (PPOConfig()
          .training(gamma=0.999, lr=0.000)
          .environment(env='case2env')
          .resources(num_gpus=0)
          .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
        )

baseline = blconfig.build()

for _ in range(10):
    algo.train()
    baseline.train()

env.close()