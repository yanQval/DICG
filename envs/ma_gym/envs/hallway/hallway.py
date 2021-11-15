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


class HallwayEnv(gym.Env):
    
    def __init__(
            self,
            n_agents=5,
            n_groups=2,
            state_numbers=[4,4,4,4,4],
            group_ids=[0,0,1,1,1],
            reward_win=10,
            seed=None
    ):
        # Map arguments
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        self.n_agents = n_agents
        self.n_states = np.array(state_numbers,
                                 dtype=np.int)

        # Rewards args
        self.reward_win = reward_win

        # Other
        self._seed = seed

        # Actions
        self.n_actions = 3
        self.n_groups = n_groups
        self.group_ids = np.array(group_ids)

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self.episode_limit = max(state_numbers) + 10

        # initialize agents
        self.state_n = np.array([np.random.randint(low=1, high=self.n_states[i]+1) for i in range(self.n_agents)],
                                dtype=np.int)

        # self.group_members = [[] for _ in range(self.n_groups)]
        # self.status_by_group = [[] for _ in range(self.n_groups)]
        # for agent_i, group_i in enumerate(self.group_ids):
        #     self.group_members[group_i].append(agent_i)
        #     self.status_by_group[group_i].append(0)

        self.active_group = [True for _ in range(self.n_groups)]
        self.active_agent = np.array([True for _ in range(self.n_agents)])
        self._win_group = 0
        self._fail_group = 0


        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.n_actions) for _ in range(self.n_agents)])

        self._obs_high = np.array([float(self.n_agents + 1)] * 2)
        self._obs_low = np.array([0.] * 2)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return [self.state_n[agent_id], float(self.active_agent[agent_id])]


    def get_agent_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def reset(self):
        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.state_n = np.array([np.random.randint(low=1, high=self.n_states[i]+1) for i in range(self.n_agents)],
                                dtype=np.int)
        self.active_group = [True for _ in range(self.n_groups)]
        self.active_agent = np.array([True for _ in range(self.n_agents)])
        self._win_group = 0
        self._fail_group = 0

        return self.get_agent_obs()
    
    def step(self, actions):
        """Returns reward, terminated, info."""
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        last_state = self.state_n

        for agent_i, action in enumerate(actions):
            if self.active_agent[agent_i]:
                if action == 0:
                    pass
                elif action == 1:
                    self.state_n[agent_i] = max(0, self.state_n[agent_i] - 1)
                elif action == 2:
                    self.state_n[agent_i] = min(self.n_states[agent_i], self.state_n[agent_i] + 1)

        reward = 0
        terminated = False
        info['battle_won'] = False

        win_in_this_round = 0
        win_agents = np.array([False for _ in range(self.n_agents)])

        for group_i in range(self.n_groups):
            if self.active_group[group_i]:
                id = self.state_n[self.group_ids == group_i]

                if (id == 0).all():
                    reward += self.reward_win
                    self._win_group += 1
                    self.active_group[group_i] = False
                    self.active_agent[self.group_ids == group_i] = False
                    win_agents[self.group_ids == group_i] = True
                    win_in_this_round += 1
                elif (id == 0).any():
                    self.active_group[group_i] = False
                    self.active_agent[self.group_ids == group_i] = False
                    self._fail_group += 1

        info['win_group'] = self._win_group

        if win_in_this_round > 1:
            self._win_group -= win_in_this_round
            reward -= self.reward_win * 1.5 * win_in_this_round
            self.active_agent[win_agents] = True
            self.state_n[win_agents] = last_state[win_agents]

        if self._win_group == self.n_groups:
            terminated = True
            self.battles_won += 1
            info['battle_won'] = True
        elif self._fail_group == self.n_groups:
            terminated = True

        if self._episode_steps >= self.episode_limit:
            terminated = True

        if terminated:
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

