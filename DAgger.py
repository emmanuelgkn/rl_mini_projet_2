import os, sys, tempfile
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from stable_baselines3.common.monitor import Monitor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../pystk_project/src")))
import pystk2_gymnasium
from agents.team3.agent3 import Agent3
from stable_baselines3.common.policies import ActorCriticPolicy

class LayerNormReLU(nn.Sequential):
    def __init__(self):
        super().__init__(nn.LayerNorm(64), nn.ReLU())

def make_env():
    env = gym.make("supertuxkart/flattened_continuous_actions-v0", use_subprocess=False)
    obs_space = env.observation_space["continuous"]
    env = gym.wrappers.TransformObservation(env, lambda o: o["continuous"])
    env.observation_space = obs_space
    return Monitor(env)


class ExpertPolicy(BasePolicy):
    def __init__(self, venv):
        super().__init__(venv.observation_space, venv.action_space)
        self.raw_env = venv.envs[0].unwrapped
        self.agent = Agent3(self.raw_env)

    def _predict(self, observation, deterministic=False):
        raw_obs = self.raw_env.get_observation(self.raw_env.kart_ix, False)
        act = self.agent.choose_action(raw_obs)
        return torch.tensor([[float(act["acceleration"]), float(act["steer"])]])


if __name__ == "__main__":

    # Pour la repoductibilité
    torch.manual_seed(0)
    np.random.seed(0)

    rng = np.random.default_rng(0)
    env = DummyVecEnv([make_env])
    expert = ExpertPolicy(env)

    policy = ActorCriticPolicy(
    env.observation_space,
    env.action_space,
    lr_schedule=lambda _: 1e-3,
    net_arch=[64, 64],
    activation_fn=LayerNormReLU,
    
)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=rng,
        policy=policy,
    )

    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        dagger_trainer = SimpleDAggerTrainer(
            venv=env,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
        )
        dagger_trainer.train(20_000)

    dagger_trainer.policy.save("data/dagger_policy.pt")
    rewards, lengths = evaluate_policy(dagger_trainer.policy, env, 1, return_episode_rewards=True)
    print("Reward moyen :", np.mean(rewards))
    print("Temps de tour (en pas) :", np.mean(lengths))
    env.close()
# Reward moyen : 314.085036
# Temps de tour (en pas) : 374.0
