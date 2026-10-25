from DAgger import *
from stable_baselines3.common.policies import ActorCriticPolicy

def make_env():
    env = gym.make("supertuxkart/flattened_continuous_actions-v0",
                   track="snowmountain", 
                   use_subprocess=False, 
                   render_mode="human")
    obs_space = env.observation_space["continuous"]
    env = gym.wrappers.TransformObservation(env, lambda o: o["continuous"])
    env.observation_space = obs_space
    return Monitor(env)

env = DummyVecEnv([make_env])

data = torch.load("data/dagger_policy.pt", weights_only=False)
policy = ActorCriticPolicy(**data["data"])
policy.load_state_dict(data["state_dict"])

rewards, lengths = evaluate_policy(policy, env, 3, return_episode_rewards=True)
print("Reward moyen :", np.mean(rewards))
print("Temps de tour (en pas) :", np.mean(lengths))
# Reward moyen : 314.085036
# Temps de tour (en pas) : 374.0
