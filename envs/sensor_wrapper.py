# Using local gym
import sys
import os
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../../')


from envs.ma_gym.envs.sensor import SensorEnv
import gym
import torch
import numpy as np

import dowel
from dowel import logger, tabular
from garage.misc.prog_bar_counter import ProgBarCounter



class SensorWrapper(SensorEnv):

    def __init__(self, centralized, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = self.action_space[0]
        self.observation_space = self.observation_space[0]
        self.centralized = centralized
        if centralized:
            self.observation_space = gym.spaces.Box(
                low=np.array(list(self.observation_space.low) * self.n_agents),
                high=np.array(list(self.observation_space.high) * self.n_agents)
            )
        
        self.pickleable = False

    def get_avail_actions(self):
        avail_actions = self.avail_actions
        if not self.centralized:
            return avail_actions
        else:
            return np.concatenate(avail_actions)

    def get_agent_obs(self):
        obs = super().get_agent_obs()
        return obs

    def step(self, actions):
        obses, rewards, dones, infos = super().step(actions)
        if not self.centralized:
            return obses, rewards, dones, infos
        else:
            return np.concatenate(obses), np.mean(rewards), np.all(dones), infos

    def reset(self):
        obses = super().reset()
        if not self.centralized:
            return obses
        else:
            return np.concatenate(obses)

    def eval(self, policy, n_episodes=20, greedy=True, load_from_file=False, 
             render=False):
        
        if load_from_file:
            logger.add_output(dowel.StdOutput())
        logger.log('Evaluating policy, {} episodes, greedy = {} ...'.format(
            n_episodes, greedy))
        scanned = 0
        episode_rewards = []
        pbar = ProgBarCounter(n_episodes)
        for e in range(n_episodes):
            obs = self.reset()
            policy.reset([True])
            info = {'scanned': 0}
            terminated = False
            episode_rewards.append(0)

            while not terminated:
                obs = np.array([obs]) # add [.] for vec_env
                avail_actions = np.array([self.get_avail_actions()])
                actions, agent_infos = policy.get_actions(obs, 
                    avail_actions, greedy=greedy)
                obs, reward, terminated, info = self.step(actions[0])
                if not self.centralized:
                    terminated = all(terminated)
                episode_rewards[-1] += np.mean(reward)
            pbar.inc(1)

            # If case SC2 restarts during eval, KeyError: 'battle_won' can happen
            # Take precaution
            if type(info) == dict: 
                if 'scaned' in info.keys():
                    scanned += info["scaned"]
        
        pbar.stop()
        policy.reset([True])
        scanned_mean = scanned / n_episodes
        avg_return = np.mean(episode_rewards)

        logger.log('EvalScannedMean: {}'.format(scanned_mean))
        logger.log('EvalAvgReturn: {}'.format(avg_return))
        if not load_from_file:
            tabular.record('EvalScannedMeans', scanned_mean)
            tabular.record('EvalAvgReturn', avg_return)
