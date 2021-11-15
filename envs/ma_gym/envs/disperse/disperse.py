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


class DisperseEnv(gym.Env):
    
    def __init__(
            self,
            n_agents=12,
            n_actions=4,
            initial_need=[0, 0, 0, 0],
            episode_limit=10,
            seed=None
    ):
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        self.n_agents = n_agents

        # Observations and state

        # Other
        self._seed = seed

        # Actions
        self.n_actions = n_actions
        self.initial_need = initial_need

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self.episode_limit = episode_limit
        self.needs = initial_need
        self.actions = np.random.randint(0, n_actions, n_agents)
        self._match = 0


        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.n_actions) for _ in range(self.n_agents)])

        self._obs_high = np.array([float(self.n_agents + 1)] * (self.n_actions + 1 + self.n_agents))
        self._obs_low = np.array([0.] * (self.n_actions + 1 + self.n_agents))
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        agent_action = self.actions[agent_id]
        # print([agent_action, self.needs[agent_action]])
        # print([float(x) for x in (self.actions == agent_action)])
        action_one_hot = np.zeros(self.n_actions)
        action_one_hot[agent_action] = 1.
        return action_one_hot.tolist() + [self.needs[agent_action]] + [float(x) for x in (self.actions == agent_action)]
        # return np.array([agent_action, self.needs[agent_action], (self.actions == agent_action).sum()])


    def get_agent_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def reset(self):
        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self.needs = self.initial_need
        self.actions = np.random.randint(0, self.n_actions, self.n_agents)
        self._match = 0

        return self.get_agent_obs()
    
    def _split_x(self, x, n):
        result = np.zeros(n)
        p = np.random.randint(low=0, high=n)
        low = x // 2
        result[p] = np.random.randint(low=low, high=x+1)
        return result

    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}
        self.actions = actions

        terminated = False
        info['battle_won'] = False
        # actions_numpy = actions.detach().cpu().numpy()

        delta = []
        for action_i in range(self.n_actions):
            supply = float((actions == action_i).sum())
            need = float(self.needs[action_i])

            if supply >= need:
                self._match += 1

            delta.append(min(supply - need, 0))
        reward = float(np.array(delta).sum()) / self.n_agents

        # print('step', self._episode_steps, ':')
        # print(self.needs)
        # print(self.actions)
        # print(reward)

        self.needs = self._split_x(self.n_agents, self.n_actions)
        info['match'] = self._match

        if self._episode_steps >= self.episode_limit:
            # print(self._match)
            # print(reward)
            terminated = True
            self._episode_count += 1
            self.battles_game += 1

            if self._match == self.n_actions * self.episode_limit:
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

