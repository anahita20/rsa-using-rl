# RL algorithm
from typing import Dict
from ray.rllib.algorithms.impala import ImpalaConfig

import ray
import csv

import gymnasium as gym

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from ray.rllib.algorithms.callbacks import DefaultCallbacks
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
with open("utils_impala_case2.csv", mode='w', newline='') as file:
  writer = csv.writer(file)
  writer.writerow(['Episode ID', 'Network Utilization'])

# callback to get total network utilizations after every training iteration for IMPALA
class MyIMPALACallbacks(DefaultCallbacks):
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
            with open("utils_impala_case2.csv", mode='a', newline='') as file:
              writer = csv.writer(file)
              writer.writerow([episode.episode_id, network_utilization])

config = (ImpalaConfig()
          .training(lr=0.003, gamma=0.999)
          .resources(num_gpus=0)
          .callbacks(MyIMPALACallbacks)
          .environment(env='case2env')
          .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
          
)

algo = config.build()

blconfig = (ImpalaConfig()
          .training(lr=0.0, gamma=0.999)
          .resources(num_gpus=0)
          .environment(env='case2env')
          .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
          
)
baseline = blconfig.build()

for _ in range(10):
    algo.train()
    baseline.train()

env.close()