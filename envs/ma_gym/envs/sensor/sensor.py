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


class SensorEnv(gym.Env):
    
    def __init__(
            self,
            n_preys=3,
            episode_limit=10,
            array_height=3,
            array_width=5,
            catch_reward=2,
            scan_cost=1,
            seed=None
    ):
        # Map arguments
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        self.n_agents = array_height * array_width
        self.n_preys = n_preys
        self.episode_limit = episode_limit
        self.array_width = array_width
        self.array_height = array_height
        self.map_height = 2 * array_height - 1
        self.map_width = 2 * array_width - 1
        self.catch_reward = catch_reward
        self.scan_cost = scan_cost

        # Other
        self._seed = seed

        # Actions
        self.n_actions = 9

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0
        self.neighbors = [(1, 1), (1, -1), (1, 0), (-1, 0), (-1, 1), (-1, -1), (0, 1), (0, -1)]

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        # Configuration initialization
        self.prey_positions = np.zeros((self.map_height, self.map_width))
        self.occ = np.zeros((self.map_height, self.map_width)).astype(int)
        self.occ[0:self.map_height:2, 0:self.map_width:2] = 1
        self.prey_positions_idx = np.zeros((self.n_preys, 2)).astype(int)
        self.agent_positions_idx = np.zeros((self.n_agents, 2)).astype(int)

        for prey_i in range(self.n_preys):
            prey_h = np.random.randint(low=0, high=self.map_height)
            prey_w = np.random.randint(low=0, high=self.map_width)

            while self.occ[prey_h, prey_w]:
                prey_h = np.random.randint(low=0, high=self.map_height)
                prey_w = np.random.randint(low=0, high=self.map_width)

            self.prey_positions[prey_h, prey_w] = prey_i + 1
            self.occ[prey_h, prey_w] = 1
            self.prey_positions_idx[prey_i, 0] = prey_h
            self.prey_positions_idx[prey_i, 1] = prey_w

        for agent_y in range(self.array_height):
            for agent_x in range(self.array_width):
                self.agent_positions_idx[agent_y * array_width + agent_x, 0] = agent_y * 2
                self.agent_positions_idx[agent_y * array_width + agent_x, 1] = agent_x * 2

        self.obs_size = 11
        self.avail_actions = []
        for agent_i in range(self.n_agents):
            agent_y = self.agent_positions_idx[agent_i, 0]
            agent_x = self.agent_positions_idx[agent_i, 1]
            _avail_actions = [] # size 9

            for delta in self.neighbors:
                if 0 <= agent_x + delta[0] < self.map_width and 0 <= agent_y + delta[1] < self.map_height:
                    _avail_actions.append(1)
                else:
                    _avail_actions.append(0)
            _avail_actions.append(1)
            self.avail_actions.append(_avail_actions)

        self._episode_scaned = 0


        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.n_actions) for _ in range(self.n_agents)])

        self._obs_high = np.array([float(self.n_agents + 1)] * self.obs_size)
        self._obs_low = np.array([0.] * self.obs_size)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        agent_h = self.agent_positions_idx[agent_id, 0]
        agent_w = self.agent_positions_idx[agent_id, 1]
        occ_temp = np.pad(self.occ, ((1,1),(1,1)), 'constant', constant_values=(-1,-1))
        agent_h = agent_h + 1
        agent_w = agent_w + 1
        obs = occ_temp[agent_h - 1: agent_h + 2, agent_w - 1: agent_w + 2]
        obs[1, 1] = 0
        return obs.flatten().tolist() +  self.agent_positions_idx[agent_id].tolist()

    def get_agent_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def reset(self):
        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self.prey_positions = np.zeros((self.map_height, self.map_width))
        self.occ = np.zeros((self.map_height, self.map_width)).astype(int)
        self.occ[0:self.map_height:2, 0:self.map_width:2] = 1
        self.prey_positions_idx = np.zeros((self.n_preys, 2)).astype(int)
        self.agent_positions_idx = np.zeros((self.n_agents, 2)).astype(int)

        for prey_i in range(self.n_preys):
            prey_h = np.random.randint(low=0, high=self.map_height)
            prey_w = np.random.randint(low=0, high=self.map_width)

            while self.occ[prey_h, prey_w]:
                prey_h = np.random.randint(low=0, high=self.map_height)
                prey_w = np.random.randint(low=0, high=self.map_width)

            self.prey_positions[prey_h, prey_w] = prey_i + 1
            self.occ[prey_h, prey_w] = 1
            self.prey_positions_idx[prey_i, 0] = prey_h
            self.prey_positions_idx[prey_i, 1] = prey_w

        for agent_y in range(self.array_height):
            for agent_x in range(self.array_width):
                self.agent_positions_idx[agent_y * self.array_width + agent_x, 0] = agent_y * 2
                self.agent_positions_idx[agent_y * self.array_width + agent_x, 1] = agent_x * 2

        self._episode_scaned = 0

        return self.get_agent_obs()
    
    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        reward = 0
        terminated = False
        info['battle_won'] = False

        prey_scaned = np.array([0 for _ in range(self.n_preys)])

        # map = np.zeros((self.map_height, self.map_width))
        # map[0:self.map_height:2, 0:self.map_width:2] = 1
        # for prey_i in range(self.n_preys):
        #     map[self.prey_positions_idx[prey_i, 0], self.prey_positions_idx[prey_i, 1]] = 2

        for agent_i, action in enumerate(actions):
            if action < 8:
                reward -= self.scan_cost

                agent_y = self.agent_positions_idx[agent_i, 0]
                agent_x = self.agent_positions_idx[agent_i, 1]

                scan_x = agent_x + self.neighbors[action][0]
                scan_y = agent_y + self.neighbors[action][1]

                # map[scan_y, scan_x] += 10

                if 0 <= scan_y < self.map_height and 0 <= scan_x < self.map_width:
                    for prey_i in range(self.n_preys):
                        if scan_x == self.prey_positions_idx[prey_i, 1] and scan_y == self.prey_positions_idx[prey_i, 0]:
                            prey_scaned[prey_i] += 1

        # print(map)

        for _prey_scaned in prey_scaned:
            if _prey_scaned == 2:
                reward += self.catch_reward
                self._episode_scaned += 1
            elif _prey_scaned == 3:
                reward += self.catch_reward * 1.5
                self._episode_scaned += 1
            elif _prey_scaned == 4:
                reward += self.catch_reward * 2
                self._episode_scaned += 1

        info['scaned'] = self._episode_scaned

        # Prey move
        for prey_i in range(self.n_preys):
            h, w = self.prey_positions_idx[prey_i, 0], self.prey_positions_idx[prey_i, 1]

            delta_h = np.random.randint(low=-2, high=3)
            delta_w = np.random.randint(low=-2, high=3)

            target_w = min(max(w + delta_w, 0), self.map_width - 1)
            target_h = min(max(h + delta_h, 0), self.map_height - 1)

            while self.occ[target_h, target_w]:
                delta_h = np.random.randint(low=-2, high=3)
                delta_w = np.random.randint(low=-2, high=3)

                target_w = min(max(w + delta_w, 0), self.map_width - 1)
                target_h = min(max(h + delta_h, 0), self.map_height - 1)

            self.occ[h, w] = 0
            self.occ[target_h, target_w] = 1
            self.prey_positions_idx[prey_i, 0] = target_h
            self.prey_positions_idx[prey_i, 1] = target_w

        if self._episode_steps >= self.episode_limit:
            terminated = True
            self._episode_count += 1
            self.battles_game += 1

        return self.get_agent_obs(), [reward] * self.n_agents, [terminated] * self.n_agents, info

    def render(self, mode='human'):
        pass

    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def close(self):
        pass

