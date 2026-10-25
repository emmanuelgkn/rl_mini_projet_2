import torch as th
import numpy as np
import torch.nn as nn
from DAgger import make_env
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.save_util import load_from_pkl
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import ReplayBufferSamples

class LayerNormReLU(nn.Sequential):
    def __init__(self):
        super().__init__(nn.LayerNorm(256), nn.ReLU())

if __name__ == "__main__":
    env = DummyVecEnv([make_env])
    expert_buffer = load_from_pkl("expert_buffer.pkl")

    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=1000,
        gradient_steps=4,
        batch_size=256,
        policy_kwargs=dict(
            net_arch=[256,256],
            activation_fn=LayerNormReLU,
            n_critics=5,
        ),
        verbose=1,
    )

    online_sample = model.replay_buffer.sample

    def rlpd_sample(batch_size, env=None):
        half = batch_size//2
        b_on = online_sample(half,env)
        b_ex = expert_buffer.sample(batch_size-half,env)
        return ReplayBufferSamples(*[th.cat([x,y],dim=0) for x,y in zip(b_on,b_ex)])

    model.replay_buffer.sample = rlpd_sample

    model.learn(total_timesteps=20_000, log_interval=1)
    model.save("sac_rlpd")

    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=1, return_episode_rewards=True)
    print("Reward_moyen :", np.mean(rewards), "| Pas:", np.mean(lengths))
    env.close()
