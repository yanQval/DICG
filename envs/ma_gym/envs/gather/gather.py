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


class GatherEnv(gym.Env):
    
    def __init__(
            self,
            n_agents=7,
            episode_limit=20,
            map_height=3,
            map_width=7,
            catch_reward=10,
            catch_fail_reward=-5,
            target_reward=0.000,
            other_reward=5,
            seed=None
    ):
        # Map arguments
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        self.n_agents = n_agents
        self.episode_limit = episode_limit
        self.map_height = map_height
        self.map_width = map_width
        self.catch_reward = catch_reward
        self.catch_fail_reward = catch_fail_reward
        self.other_reward = other_reward
        self.target_reward = target_reward


        # Other
        self._seed = seed

        # Actions
        self.n_actions = 5

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        # Configuration initialization
        self.agent_positions_idx = np.zeros((self.n_agents, 2)).astype(int)
        target_count = [0, 0, 0]
        self.agent_target = [np.zeros(2) for _ in range(self.n_agents)]

        for agent_i in range(self.n_agents):
            agent_x = np.random.randint(low=0, high=self.map_width)
            agent_y = np.random.randint(low=0, high=self.map_height)

            self.agent_positions_idx[agent_i, 0] = agent_y
            self.agent_positions_idx[agent_i, 1] = agent_x

            if self._distance(agent_x, 0, agent_y, 1) < self._distance(agent_x, 2, agent_y, 1):
                self.agent_target[agent_i] = np.array([1, 0])
                target_count[0] += 1
            else:
                if self._distance(agent_x, 4, agent_y, 1) <= self._distance(agent_x, 2, agent_y, 1):
                    self.agent_target[agent_i] = np.array([1, 4])
                    target_count[2] += 1
                else:
                    self.agent_target[agent_i] = np.array([1, 2])
                    target_count[1] += 1

        if target_count[0] >= target_count[1] and target_count[0] >= target_count[2]:
            self.target = np.array([1, 0])
            self.n_target = np.array([1, 2])
            self.n2_target = np.array([1, 4])
        else:
            if target_count[1] >= target_count[0] and target_count[1] >= target_count[2]:
                self.target = np.array([1, 2])
                self.n_target = np.array([1, 0])
                self.n2_target = np.array([1, 4])
            else:
                self.target = np.array([1, 4])
                self.n_target = np.array([1, 2])
                self.n2_target = np.array([1, 0])

        for agent_i in range(self.n_agents):
            if self.agent_target[agent_i][1] != self.target[1] or self.agent_target[agent_i][0] != self.target[0]:
                self.agent_target[agent_i] = np.array([-1, -1])


        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.n_actions) for _ in range(self.n_agents)])

        self._obs_high = np.array([float(self.n_agents + 1)] * 4)
        self._obs_low = np.array([0.] * 4)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.agent_positions_idx[agent_id].tolist() + self.agent_target[agent_id].tolist()


    def get_agent_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def reset(self):
        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        # Configuration initialization
        self.agent_positions_idx = np.zeros((self.n_agents, 2)).astype(int)
        target_count = [0, 0, 0]
        self.agent_target = [np.zeros(2) for _ in range(self.n_agents)]

        for agent_i in range(self.n_agents):
            agent_x = np.random.randint(low=0, high=self.map_width)
            agent_y = np.random.randint(low=0, high=self.map_height)

            self.agent_positions_idx[agent_i, 0] = agent_y
            self.agent_positions_idx[agent_i, 1] = agent_x

            if self._distance(agent_x, 0, agent_y, 1) < self._distance(agent_x, 2, agent_y, 1):
                self.agent_target[agent_i] = np.array([1, 0])
                target_count[0] += 1
            else:
                if self._distance(agent_x, 4, agent_y, 1) <= self._distance(agent_x, 2, agent_y, 1):
                    self.agent_target[agent_i] = np.array([1, 4])
                    target_count[2] += 1
                else:
                    self.agent_target[agent_i] = np.array([1, 2])
                    target_count[1] += 1

        if target_count[0] >= target_count[1] and target_count[0] >= target_count[2]:
            self.target = np.array([1, 0])
            self.n_target = np.array([1, 2])
            self.n2_target = np.array([1, 4])
        else:
            if target_count[1] >= target_count[0] and target_count[1] >= target_count[2]:
                self.target = np.array([1, 2])
                self.n_target = np.array([1, 0])
                self.n2_target = np.array([1, 4])
            else:
                self.target = np.array([1, 4])
                self.n_target = np.array([1, 2])
                self.n2_target = np.array([1, 0])

        for agent_i in range(self.n_agents):
            if self.agent_target[agent_i][1] != self.target[1] or self.agent_target[agent_i][0] != self.target[0]:
                self.agent_target[agent_i] = np.array([-1, -1])

        return self.get_agent_obs()

    def _distance(self, x1, x2, y1, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    
    def step(self, actions):
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        reward = 0
        terminated = False
        info['battle_won'] = False

        occ_count = 0
        n_occ_count = 0
        n2_occ_count = 0

        # map = np.zeros((self.map_height, self.map_width))

        for agent_i, action in enumerate(actions):
            y, x = self.agent_positions_idx[agent_i, 0], self.agent_positions_idx[agent_i, 1]

            target_x = x
            target_y = y

            if action == 0:
                target_x, target_y = x, min(self.map_height - 1, y + 1)
            elif action == 1:
                target_x, target_y = min(x + 1, self.map_width - 1), y
            elif action == 2:
                target_x, target_y = x, max(0, y - 1)
            elif action == 3:
                target_x, target_y = max(0, x - 1), y

            self.agent_positions_idx[agent_i, 0], self.agent_positions_idx[agent_i, 1] = target_y, target_x
            # map[target_y, target_x] += 1

            if target_x == self.target[1] and target_y == self.target[0]:
                occ_count += 1
            elif target_x == self.n_target[1] and target_y == self.n_target[0]:
                n_occ_count += 1
            elif target_x == self.n2_target[1] and target_y == self.n2_target[0]:
                n2_occ_count += 1

        # print(map)

        if occ_count == self.n_agents:
            terminated = True
            info['battle_won'] = True
            self.battles_won += 1
            reward += self.catch_reward

        if self._episode_steps >= self.episode_limit:
            terminated = True

            if occ_count + n_occ_count+ n2_occ_count == self.n_agents:
                if occ_count == 0:
                    reward += self.other_reward
                elif occ_count < self.n_agents:
                    reward += self.catch_fail_reward

        if terminated:
            #print("terminated")
            #print(reward)
            #print(occ_count)
            #print(n_occ_count)
            #print(n2_occ_count)
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

