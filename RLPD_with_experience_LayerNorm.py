import torch as th
import numpy as np
import torch.nn as nn
from DAgger import make_env
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.save_util import load_from_pkl
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# On importe les données offline générées par l'agent 3 
expert_buffer = load_from_pkl("data/expert_buffer.pkl")


obs_data = expert_buffer.observations[:expert_buffer.size(), 0, :]
obs_mean = th.tensor(obs_data.mean(axis=0), dtype=th.float32)
obs_std = th.tensor(obs_data.std(axis=0) + 1e-5, dtype=th.float32)

#On adapte les valeurs des entrées
class StdExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, observation_space.shape[0])
        self.register_buffer("mean", obs_mean)
        self.register_buffer("std", obs_std)

    def forward(self, x):
        return th.clamp((x - self.mean) / self.std, -10.0, 10.0)

class LayerNormReLU(nn.Sequential):
    def __init__(self):
        super().__init__(nn.LayerNorm(256), nn.ReLU())


# Modification du critique de SAC de manière à incorporer des fonctions d'activations différentiées avec l'acteur 
class SACPolicyCriticAct(SACPolicy):
    def __init__(self, *args, critic_activation_fn=None, **kwargs):
        self._critic_activation_fn = critic_activation_fn
        super().__init__(*args, **kwargs)
 
    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        if self._critic_activation_fn is not None:
            critic_kwargs["activation_fn"] = self._critic_activation_fn
        return ContinuousCritic(**critic_kwargs).to(self.device)
 
    def _get_constructor_parameters(self):
        # pour que save/load conserve l'option
        data = super()._get_constructor_parameters()
        data.update(critic_activation_fn=self._critic_activation_fn)
        return data


if __name__ == "__main__":
    env = DummyVecEnv([make_env])

    for seed in range(3):
        env.envs[0].episode_returns.clear()
        env.envs[0].episode_lengths.clear()

        model = SAC(
            SACPolicyCriticAct,
            env,
            seed=seed,
            learning_starts=100,
            ent_coef="auto_0.01",
            gradient_steps=10,
            batch_size=256,
            learning_rate= 1e-3,
            policy_kwargs=dict(
                net_arch=[256, 256],
                activation_fn=LayerNormReLU,
                critic_activation_fn = LayerNormReLU, #A remplacer avec nn.ReLU si on ne veut pas normaliser la sortie
                n_critics=2,
                features_extractor_class=StdExtractor,
            ),
            verbose=1,
        )
        

        online_sample = model.replay_buffer.sample

        #Redefinition de la manière dont on récupère les batch de transitions pour inclure des données offline
        def rlpd_sample(batch_size, env=None):
            half = batch_size//2
            b_on = online_sample(half,env)
            b_ex = expert_buffer.sample(batch_size-half,env)
            return ReplayBufferSamples(*[th.cat([x,y],dim=0) for x,y in zip(b_on,b_ex)])

        model.replay_buffer.sample = rlpd_sample

        model.learn(total_timesteps=20_000, log_interval=1)

        np.savez(
        f"logs/metrics_rlpd_{seed}_with_LayerNorm.npz",
        rewards=env.envs[0].get_episode_rewards(),
        lengths=env.envs[0].get_episode_lengths(),
        )
        

    model.save("data/sac_rlpd")
    env.close()
