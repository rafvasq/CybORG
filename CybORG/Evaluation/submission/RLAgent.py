from inspect import signature
from typing import Union

from gym import Space

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent

from CybORG import CybORG
from CybORG.Simulator.Scenarios import DroneSwarmScenarioGenerator
from CybORG.Shared import Results
from ray.rllib.agents.dqn import DQNTrainer

from CybORG.Agents.Wrappers.CommsPettingZooParallelWrapper import ObsCommsPettingZooParallelWrapper
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env

def env_creator_CC3(env_config: dict):
    sg = DroneSwarmScenarioGenerator()
    cyborg = CybORG(scenario_generator=sg, environment='sim')
    env = ParallelPettingZooEnv(ObsCommsPettingZooParallelWrapper(env=cyborg))
    return env

register_env(name="CC3", env_creator=env_creator_CC3)

class RLAgent(BaseAgent):
    def __init__(self):
        self.trainer = DQNTrainer(env="CC3", config={'learning_starts': 1000, 'replay_buffer_config': {'capacity': 10000}, 'train_batch_size': 32})
        self.trainer.restore("/cage/CybORG/Evaluation/dqn_obscomms2/checkpoint_029881/checkpoint-29881")

    def train(self, results: Results):
        """allows an agent to learn a policy"""
        pass

    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        return self.trainer.compute_single_action(observation)

    def end_episode(self):
        """Allows an agent to update its internal state"""
        pass

    def set_initial_values(self, action_space, observation):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return f"{self.__class__.__name__}"
