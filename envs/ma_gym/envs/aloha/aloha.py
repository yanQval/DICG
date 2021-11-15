import copy
import logging
import random

import gym
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text

logger = logging.getLogger(__name__)


class AlohaEnv(gym.Env):
    
    def __init__(
            self,
            n_agents=10,
            episode_limit=20,
            max_list_length=5,
            seed=None
    ):
        # Map arguments
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        self.n_agents = n_agents


        # Rewards args
        self.max_list_length = max_list_length

        # Other
        self._seed = seed

        # Actions
        self.n_actions = 2
        self.reward_scale = 10.

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0
        self.episode_limit = episode_limit

        # Initialize backlogs
        self.backlogs = np.ones(self.n_agents)
        self.transmitted = 0
        self.adj = np.array([[0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
                             [1., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                             [0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
                             [0., 0., 1., 0., 1., 0., 0., 0., 1., 0.],
                             [0., 0., 0., 1., 0., 1., 0., 0., 0., 1.],
                             [1., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
                             [0., 1., 0., 0., 0., 1., 0., 1., 0., 0.],
                             [0., 0., 1., 0., 0., 0., 1., 0., 1., 0.],
                             [0., 0., 0., 1., 0., 0., 0., 1., 0., 1.],
                             [0., 0., 0., 0., 1., 0., 0., 0., 1., 0.]])

        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.n_actions) for _ in range(self.n_agents)])

        self._obs_high = np.array([float(self.max_list_length + 1)])
        self._obs_low = np.array([0.])
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            _obs.append([self.backlogs[agent_i]])

        return _obs

    def reset(self):
        self._episode_steps = 0

        self.transmitted = 0
        self.backlogs = np.ones(self.n_agents)

        return self.get_agent_obs()

    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        actions_numpy = actions
        reward = 0

        for agent_i, action in enumerate(actions):
            if action == 1 and self.backlogs[agent_i] > 0:
                if (self.adj[agent_i] * actions_numpy).sum() < 0.01:
                    self.backlogs[agent_i] = self.backlogs[agent_i] - 1
                    self.transmitted += 1
                    reward += 0.1
                else:
                    reward -= 10

        terminated = False
        info['trans'] = self.transmitted
        info['left'] = self.backlogs.sum()
        info['battle_won'] = False

        # Add new packages
        self.backlogs += np.random.choice([0., 1.], p=[0.4, 0.6], size=[self.n_agents])
        self.backlogs = np.clip(self.backlogs, a_min=0, a_max=self.max_list_length)

        if self._episode_steps >= self.episode_limit:
            terminated = True

        if terminated:
            self._episode_count += 1
            self.battles_game += 1

            if self.transmitted > self.n_agents / 2 * self.episode_limit * 0.9:
                info['battle_won'] = True
                self.battles_won += 1

        return self.get_agent_obs(), [reward] * self.n_agents, [terminated] * self.n_agents, info

    def render(self, mode='human'):
        pass

    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def close(self):
        pass

