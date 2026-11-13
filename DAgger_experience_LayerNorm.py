import os, sys, tempfile
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.save_util import load_from_pkl
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data.types import Transitions
from stable_baselines3.common.monitor import Monitor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../pystk_project/src")))
import pystk2_gymnasium
from agents.team3.agent3 import Agent3
from stable_baselines3.common.policies import ActorCriticPolicy

class LayerNormReLU(nn.Sequential):
    def __init__(self):
        super().__init__(nn.LayerNorm(256), nn.ReLU())

def make_env():
    env = gym.make(
        "supertuxkart/flattened_continuous_actions-v0", 
        track="snowmountain",
        use_subprocess=False, 
        max_episode_steps=700
        )
    obs_space = env.observation_space["continuous"]
    env = gym.wrappers.TransformObservation(env, lambda o: o["continuous"])
    env.observation_space = obs_space
    return Monitor(env)

# Wrapper pour l'agent 3 qui sert à générer la politique experte
class ExpertPolicy(BasePolicy):
    def __init__(self, venv):
        super().__init__(venv.observation_space, venv.action_space)
        self.raw_env = venv.envs[0].unwrapped
        self.agent = Agent3(self.raw_env)

    def _predict(self, observation, deterministic=False):
        raw_obs = self.raw_env.get_observation(self.raw_env.kart_ix, False)
        act = self.agent.choose_action(raw_obs)
        return torch.tensor([[float(act["acceleration"]), float(act["steer"])]])


# On change le fonctionnement de l'acteur de SAC pour l'adapter au Behavior Cloning
class SACPolicyBC(SACPolicy):
    def _scale_action_th(self, actions: torch.Tensor) -> torch.Tensor:
        """env units -> [-1, 1] (torch version of SACPolicy.scale_action)."""
        low = torch.as_tensor(self.action_space.low, device=actions.device, dtype=actions.dtype)
        high = torch.as_tensor(self.action_space.high, device=actions.device, dtype=actions.dtype)
        return 2.0 * ((actions - low) / (high - low)) - 1.0
 
    def evaluate_actions(self, obs, actions):
        # imitation demonstrations are in ENV units -> scale to the tanh range
        actions = self._scale_action_th(actions.float())
        mean, log_std,kwargs = self.actor.get_action_dist_params(obs)
        dist = self.actor.action_dist.proba_distribution(mean, log_std,**kwargs)
        # log_prob does atanh (clamped) + the tanh Jacobian correction internally
        log_prob = dist.log_prob(actions)
        # no closed-form entropy for a squashed Gaussian -> dummy zeros, use ent_weight=0
        entropy = torch.zeros_like(log_prob)
        return None, log_prob, entropy
 


# On récupère les données de l'expert 
expert_buffer = load_from_pkl("data/expert_buffer.pkl")
n = expert_buffer.size()
    
obs_data = expert_buffer.observations[:n, 0, :]


obs_mean = torch.tensor(obs_data.mean(axis=0), dtype=torch.float32)
obs_std = torch.tensor(obs_data.std(axis=0) + 1e-5, dtype=torch.float32)

#Comme pour RLPD, on adapte les valeurs des entrées 
class StdExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, observation_space.shape[0])
        self.register_buffer("mean", obs_mean)
        self.register_buffer("std", obs_std)

    def forward(self, x):
        return torch.clamp((x - self.mean) / self.std, -10.0, 10.0)





if __name__ == "__main__":

    # Pour la repoductibilité
    env = DummyVecEnv([make_env])
    expert = ExpertPolicy(env)
    
    
    

    for seed in range(3):
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)


        env.envs[0].episode_returns.clear()
        env.envs[0].episode_lengths.clear()
        
        
        
        
        #Initialisation de SAC avec la politique adaptée à BC
        model_actor_SAC = SAC(
        SACPolicyBC,                      
        env,
        seed=seed,
        learning_rate=1e-3,               
        learning_starts=100,
        ent_coef="auto_0.01",
        gradient_steps=10,
        batch_size=256,
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=LayerNormReLU,                 
            n_critics=2,
            features_extractor_class=StdExtractor,
        ),
        verbose=0,
        )
        
        

        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            rng=rng,
            batch_size=256,
            policy=model_actor_SAC.policy,
            ent_weight=0.0,
            l2_weight=0.0,
        )
        
        
        next_obs = expert_buffer.next_observations[:n, 0, :]
        acts_env = model_actor_SAC.policy.unscale_action(expert_buffer.actions[:n, 0, :])
        dones = expert_buffer.dones[:n, 0].astype(bool)
        
        demos = Transitions(obs=obs_data, acts=acts_env, infos=np.array([{}] * n),
                    next_obs=next_obs, dones=dones)
        
        bc_trainer.set_demonstrations(demos)
        
        print("-------- Burn-out ----------- ")

        # On échauffe l'acteur en reproduisant les actions de la politique experte de manière à ce que les deux modèles SAC utilisant les algorithmes RLPD et DAgger aient vu les mêmes données initiales.
        bc_trainer.train(n_epochs=4)
        
        print("----end of burn-out --------")
        
        
        print("-----------Begin DAgger -------------")

        # On améliore le Behavior Cloning en utilisant la méthode DAgger        
        with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
            dagger_trainer = SimpleDAggerTrainer(
                venv=env,
                scratch_dir=tmpdir,
                expert_policy=expert,
                bc_trainer=bc_trainer,
                rng=rng,
            )
            dagger_trainer.train(20_000,bc_train_kwargs=dict(log_rollouts_venv=None))
        
        print("---------- End of DAgger ----------------")
        
        
        os.makedirs("logs", exist_ok=True)
        
        np.savez(
            f"logs/metrics_dagger_{seed}.npz",
            rewards=env.envs[0].get_episode_rewards(),
            lengths=env.envs[0].get_episode_lengths(),
        )

    os.makedirs("data", exist_ok=True)
    dagger_trainer.policy.save("data/dagger_policy.pt")
    env.close()
