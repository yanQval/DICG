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


class PursuitEnv(gym.Env):
    
    def __init__(
            self,
            n_agents=8,
            n_preys=10,
            episode_limit=10,
            map_size=5,
            catch_reward=10,
            catch_fail_reward=-10,
            sight_range=2,
            seed=None
    ):
        # Map arguments
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        self.n_agents = n_agents
        self.n_preys = n_preys
        self.episode_limit = episode_limit
        self.map_size = map_size
        self.catch_reward = catch_reward
        self.catch_fail_reward = catch_fail_reward
        self.sight_range = sight_range

        # Other
        self._seed = seed

        # Actions
        self.n_actions = 9
        # 0,1,2,3:catch 4,5,6,7:move 8:stop

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0
        self.rest_prey = self.n_preys
        self.neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        # Configuration initialization
        self.prey_positions = np.zeros((self.map_size, self.map_size))
        self.agent_positions = np.zeros((self.map_size, self.map_size))
        self.occ = np.zeros((self.map_size, self.map_size)).astype(int)
        self.prey_positions_idx = np.zeros((self.n_preys, 2)).astype(int)
        self.agent_positions_idx = np.zeros((self.n_agents, 2)).astype(int)

        for prey_i in range(self.n_preys):
            prey_x = np.random.randint(low=0, high=self.map_size)
            prey_y = np.random.randint(low=0, high=self.map_size)

            while self.occ[prey_x, prey_y]:
                prey_x = np.random.randint(low=0, high=self.map_size)
                prey_y = np.random.randint(low=0, high=self.map_size)

            self.prey_positions[prey_x, prey_y] = prey_i + 1
            self.occ[prey_x, prey_y] = 1
            self.prey_positions_idx[prey_i, 0] = prey_x
            self.prey_positions_idx[prey_i, 1] = prey_y

        self.alive_preys = np.ones(self.n_preys)
        self.alive_agents = np.ones(self.n_agents)

        for agent_i in range(self.n_agents):
            agent_x = np.random.randint(low=0, high=self.map_size)
            agent_y = np.random.randint(low=0, high=self.map_size)

            while self.occ[agent_x, agent_y]:
                agent_x = np.random.randint(low=0, high=self.map_size)
                agent_y = np.random.randint(low=0, high=self.map_size)

            self.agent_positions[agent_x, agent_y] = agent_i + 1
            self.occ[agent_x, agent_y] = 1
            self.agent_positions_idx[agent_i, 0] = agent_x
            self.agent_positions_idx[agent_i, 1] = agent_y

        self._obs = -np.ones([self.map_size + 2 * self.sight_range, self.map_size + 2 * self.sight_range])
        self._obs[self.sight_range: -self.sight_range,
        self.sight_range: -self.sight_range] = self.agent_positions + 10 * self.prey_positions
        self.obs_size = (self.sight_range * 2 + 1) ** 2 * 2 + self.map_size * 2
        self.map_eye = np.eye(self.map_size)


        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.n_actions) for _ in range(self.n_agents)])

        self._obs_high = np.array([float(self.n_agents + 1)] * self.obs_size)
        self._obs_low = np.array([0.] * self.obs_size)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        if  self.alive_agents[agent_id] == 0:
            return [0 for _ in range(self.obs_size)]

        self_x = self.agent_positions_idx[agent_id, 0]
        self_y = self.agent_positions_idx[agent_id, 1]

        grid_agent = np.zeros((self.sight_range * 2 + 1, self.sight_range * 2 + 1))
        for agent_i in range(self.n_agents):
            if self.alive_agents[agent_i]:
                delta_x = self.agent_positions_idx[agent_i, 0] - self_x
                delta_y = self.agent_positions_idx[agent_i, 1] - self_y
                if abs(delta_x) <= self.sight_range and abs(delta_y) <= self.sight_range:
                    grid_agent[delta_x + self.sight_range, delta_y + self.sight_range] = agent_i + 1

        grid_prey = np.zeros((self.sight_range * 2 + 1, self.sight_range * 2 + 1))
        for prey_i in range(self.n_preys):
            delta_x = self.prey_positions_idx[prey_i, 0] - self_x
            delta_y = self.prey_positions_idx[prey_i, 1] - self_y
            if abs(delta_x) <= self.sight_range and abs(delta_y) <= self.sight_range:
                grid_prey[delta_x + self.sight_range, delta_y + self.sight_range] = 1

        return grid_agent.flatten().tolist() + grid_prey.flatten().tolist() +  self.map_eye[self_x].tolist() + self.map_eye[self_y].tolist()


    def get_agent_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]
    
    def _get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [self._get_avail_agent_actions(i) for i in range(self.n_agents)]

    def _get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        agent_x, agent_y = self.agent_positions_idx[agent_id, 0], self.agent_positions_idx[agent_id, 1]

        move = np.zeros(4)
        for move_i in range(4):
            target_x = agent_x + self.neighbors[move_i][0]
            target_y = agent_y + self.neighbors[move_i][1]
            if 0 <= target_x < self.map_size and 0 <= target_y < self.map_size:
                move[move_i] = 1

        catch = np.zeros(4)
        for catch_i in range(4):
            target_x = agent_x + self.neighbors[catch_i][0]
            target_y = agent_y + self.neighbors[catch_i][1]
            if 0 <= target_x < self.map_size and 0 <= target_y < self.map_size and self.prey_positions[target_x, target_y] > 0:
                catch[catch_i] = 1

        move *= self.alive_agents[agent_id]
        catch *= self.alive_agents[agent_id]
        return catch.tolist() + move.tolist() + [1]

    def reset(self):
        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self.prey_positions = np.zeros((self.map_size, self.map_size))
        self.agent_positions = np.zeros((self.map_size, self.map_size))
        self.occ = np.zeros((self.map_size, self.map_size)).astype(int)
        self.prey_positions_idx = np.zeros((self.n_preys, 2)).astype(int)
        self.agent_positions_idx = np.zeros((self.n_agents, 2)).astype(int)

        for prey_i in range(self.n_preys):
            prey_x = np.random.randint(low=0, high=self.map_size)
            prey_y = np.random.randint(low=0, high=self.map_size)

            while self.occ[prey_x, prey_y]:
                prey_x = np.random.randint(low=0, high=self.map_size)
                prey_y = np.random.randint(low=0, high=self.map_size)

            self.prey_positions[prey_x, prey_y] = prey_i + 1
            self.occ[prey_x, prey_y] = 1
            self.prey_positions_idx[prey_i, 0] = prey_x
            self.prey_positions_idx[prey_i, 1] = prey_y

        self.alive_preys = np.ones(self.n_preys)
        self.alive_agents = np.ones(self.n_agents)

        for agent_i in range(self.n_agents):
            agent_x = np.random.randint(low=0, high=self.map_size)
            agent_y = np.random.randint(low=0, high=self.map_size)

            while self.occ[agent_x, agent_y]:
                agent_x = np.random.randint(low=0, high=self.map_size)
                agent_y = np.random.randint(low=0, high=self.map_size)

            self.agent_positions[agent_x, agent_y] = agent_i + 1
            self.occ[agent_x, agent_y] = 1
            self.agent_positions_idx[agent_i, 0] = agent_x
            self.agent_positions_idx[agent_i, 1] = agent_y

        self.rest_prey = self.n_preys

        self._obs = -np.ones([self.map_size + 2 * self.sight_range, self.map_size + 2 * self.sight_range])
        self._obs[self.sight_range: -self.sight_range,
        self.sight_range: -self.sight_range] = self.agent_positions + 10 * self.prey_positions


        return self.get_agent_obs()
    

    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        reward = 0
        terminated = False
        info['battle_won'] = False

        def is_catch_neighbor(_agent_x, _agent_y, _prey_x, _prey_y):
            if abs(_agent_x - _prey_x) + abs(_agent_y - _prey_y) <= 1:
                return True
            return False

        # print('step', self._episode_steps)
        # grid = np.zeros((self.map_size, self.map_size + 1))
        # for i in range(self.n_agents):
        #     if self.alive_agents[i] == 1:
        #         grid[self.agent_positions_idx[i][0], self.agent_positions_idx[i][1]] = i + 1
        # for i in range(self.n_preys):
        #     if self.alive_preys[i] == 1:
        #         grid[self.prey_positions_idx[i][0], self.prey_positions_idx[i][1]] = -1
        # print(actions)
        # for i in range(self.map_size):
        #     grid[i, self.map_size] = -9
        #     print(grid[i])
        # print(self.alive_agents)

        for prey_i in range(self.n_preys):
            if self.alive_preys[prey_i]:
                catch_number = 0
                prey_x, prey_y = self.prey_positions_idx[prey_i, 0], self.prey_positions_idx[prey_i, 1]

                for agent_i, action in enumerate(actions):
                    agent_x, agent_y = self.agent_positions_idx[agent_i, 0], self.agent_positions_idx[agent_i, 1]
                    if is_catch_neighbor(agent_x, agent_y, prey_x, prey_y) and action < 4 \
                            and agent_x + self.neighbors[action][0] == prey_x and agent_y + self.neighbors[action][1] == prey_y:
                        catch_number += 1

                if catch_number == 1:
                   reward += self.catch_fail_reward
                if catch_number >= 2:
                    reward += self.catch_reward
                    self.alive_preys[prey_i] = 0
                    self.rest_prey -= 1
                    self.prey_positions[prey_x, prey_y] = 0
                    self.occ[prey_x, prey_y] = 0

                    self.prey_positions_idx[prey_i, 0] = self.map_size + 1
                    self.prey_positions_idx[prey_i, 1] = self.map_size + 1

                    predator_count = 2
                    for agent_i, action in enumerate(actions):
                        agent_x, agent_y = self.agent_positions_idx[agent_i, 0], self.agent_positions_idx[agent_i, 1]
                        if is_catch_neighbor(agent_x, agent_y, prey_x, prey_y) and action < 4 \
                                and agent_x + self.neighbors[action][0] == prey_x and agent_y + self.neighbors[action][1] == prey_y:
                            self.alive_agents[agent_i] = 0
                            self.agent_positions[agent_x, agent_y] = 0
                            self.agent_positions_idx[agent_i, 0] = self.map_size + 1
                            self.agent_positions_idx[agent_i, 1] = self.map_size + 1
                            self.occ[agent_x, agent_y] = 0
                            predator_count -= 1
                            if predator_count == 0:
                                break
                if self.rest_prey == 0:
                    break

        info['prey_left'] = self.rest_prey

        for agent_i, action in enumerate(actions):
            if 4 <= action <= 7:
                x, y = self.agent_positions_idx[agent_i, 0], self.agent_positions_idx[agent_i, 1]

                target_x = 100
                target_y = 100

                if action == 4:
                    target_x, target_y = x, min(self.map_size - 1, y + 1)
                elif action == 5:
                    target_x, target_y = min(x + 1, self.map_size - 1), y
                elif action == 6:
                    target_x, target_y = x, max(0, y - 1)
                elif action == 7:
                    target_x, target_y = max(0, x - 1), y

                if not self.occ[target_x, target_y]:
                    self.agent_positions[x, y] = 0
                    self.agent_positions[target_x, target_y] = agent_i + 1
                    self.occ[x, y] = 0
                    self.occ[target_x, target_y] = 1
                    self.agent_positions_idx[agent_i, 0], self.agent_positions_idx[agent_i, 1] = target_x, target_y

        # Prey move

        for prey_i in range(self.n_preys):
            if self.alive_preys[prey_i]:
                x, y = self.prey_positions_idx[prey_i, 0], self.prey_positions_idx[prey_i, 1]
                action = np.random.randint(low=0, high=5)

                if action <= 3:
                    target_x = 100
                    target_y = 100

                    if action == 0:
                        target_x, target_y = x, min(self.map_size - 1, y + 1)
                    elif action == 1:
                        target_x, target_y = min(x + 1, self.map_size - 1), y
                    elif action == 2:
                        target_x, target_y = x, max(0, y - 1)
                    elif action == 3:
                        target_x, target_y = max(0, x - 1), y

                    if not self.occ[target_x, target_y]:
                        self.prey_positions[x, y] = 0
                        self.prey_positions[target_x, target_y] = prey_i + 1
                        self.occ[x, y] = 0
                        self.occ[target_x, target_y] = 1
                        self.prey_positions_idx[prey_i, 0], self.prey_positions_idx[prey_i, 1] = target_x, target_y

        self._obs = -np.ones([self.map_size + 2 * self.sight_range, self.map_size + 2 * self.sight_range])
        self._obs[self.sight_range: -self.sight_range,
        self.sight_range: -self.sight_range] = self.agent_positions + 10 * self.prey_positions

        if self.rest_prey == 0:
            terminated = True
            info['battle_won'] = True
            self.battles_won += 1
        elif self._episode_steps >= self.episode_limit:
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

