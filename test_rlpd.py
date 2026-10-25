from RLPD import *
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor


def make_env():
    env = gym.make("supertuxkart/flattened_continuous_actions-v0", use_subprocess=False, render_mode="human")
    obs_space = env.observation_space["continuous"]
    env = gym.wrappers.TransformObservation(env, lambda o: o["continuous"])
    env.observation_space = obs_space
    return Monitor(env)

env = DummyVecEnv([make_env])
policy = SAC.load("data/sac_rlpd", env=env)
rewards, lengths = evaluate_policy(policy, env, 3, return_episode_rewards=True)
print("Reward moyen :", np.mean(rewards))
print("Temps de tour (en pas) :", np.mean(lengths))
