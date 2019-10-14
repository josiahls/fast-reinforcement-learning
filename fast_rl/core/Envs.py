from functools import partial

import gym
# noinspection PyUnresolvedReferences
import pybulletgym.envs
# noinspection PyUnresolvedReferences
import gym_maze.envs
import numpy as np


class Envs:
    @staticmethod
    def _error_out(env, ban_list):
        for key in ban_list:
            if env.__contains__(key):
                print(ban_list[key] % env)
                return False
        return True

    @staticmethod
    def ban(envs: list):
        banned_envs = {
            'Defender': 'Defender (%s) seems to load for an extremely long time. Skipping for now. Determine cause.',
            'Fetch': 'Fetch (%s) envs are not ready yet.',
            'InvertedPendulumMuJoCoEnv': 'Mujoco Inverted Pendulum (%s) has a bug.',
            'HopperMuJoCoEnv': '(%s) Does not pass azure pipeline tests',
            'InvertedDoublePendulumMuJoCoEnv': '(%s) Does not pass azure pipeline tests',
            'HalfCheetahMuJoCoEnv': '(%s) Does not pass azure pipeline tests',
            'HumanoidMuJoCoEnv': '(%s) Does not pass azure pipeline tests',
            'Walker2DMuJoCoEnv': '(%s) Does not pass azure pipeline tests',
            'AntMuJoCoEnv': '(%s) Does not pass azure pipeline tests',
            'AtlasPyBulletEnv': 'AtlasPyBulletEnv (%s) seems to load very slowly. Skipping for now.',
            'MazeEnv': '(%s) Having a maze view issue.',
        }
        envs = np.array(envs)[list(map(partial(Envs._error_out, ban_list=banned_envs), envs))]

        return envs

    @staticmethod
    def get_all_envs(key=None, exclude_key=None):
        filter_env_names = [env.id for env in gym.envs.registry.all()
                            if (key is None or env.id.lower().__contains__(key)) and \
                            (exclude_key is None or not env.id.lower().__contains__(exclude_key))]
        return Envs.ban(filter_env_names)

    @staticmethod
    def get_all_latest_envs(key=None, exclude_key=None):
        all_envs = Envs.get_all_envs(key, exclude_key)
        roots = list(set(map(lambda x: str(x).split('-v')[0], all_envs)))
        return list(set([sorted([env for env in all_envs if env.__contains__(root)])[-1] for root in roots]))
