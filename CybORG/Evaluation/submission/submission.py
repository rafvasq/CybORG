from CybORG.Agents.Wrappers.CommsPettingZooParallelWrapper import ObsCommsPettingZooParallelWrapper
from .RLAgent import RLAgent

agents = {f"blue_agent_{agent}": RLAgent() for agent in range(18)}

def wrap(env):
    return ObsCommsPettingZooParallelWrapper(env=env)

submission_name = 'Observational DQN'
submission_team = 'Experimental Novice'
submission_technique = 'DQN'
