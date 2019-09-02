import gym
from gym import error


class b_colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def is_goal_env(env, suppress_errors=True):
    msg = 'GoalEnv requires the "{}" key to be part of the observation dictionary.'
    # Enforce that each GoalEnv uses a Goal-compatible observation space.
    if not isinstance(env.observation_space, gym.spaces.Dict):
        if not suppress_errors:
            raise error.Error('GoalEnv requires an observation space of type gym.spaces.Dict')
        else:
            return False
    for key in ['observation', 'achieved_goal', 'desired_goal']:
        if key not in env.observation_space.spaces:
            if not suppress_errors:
                raise error.Error(msg.format(key))
            else:
                return False
    return True
