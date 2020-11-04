import numpy as np
import joblib
from copy import deepcopy


class EnvBatch:
    def __init__(self, make_env_func, n_envs=10):
        """ Creates n_envs environments and babysits them for ya' """
        self.n_envs = n_envs
        self.envs = [make_env_func() for _ in range(n_envs)]


    def reset(self):
        """ Reset all games and return [n_envs, *obs_shape] observations """
        return np.array([env.reset() for env in self.envs])

    def step(self, actions):
        """
        Send a vector[batch_size] of actions into respective environments
        :returns: observations[n_envs, *obs_shape], rewards[n_envs], done[n_envs,], info[n_envs]
        """
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        new_obs, rewards, done, infos = map(np.array, zip(*results))

        # reset environments automatically
        for i in range(len(self.envs)):
            if done[i]:
                new_obs[i] = self.envs[i].reset()

        return new_obs, rewards, done, infos

