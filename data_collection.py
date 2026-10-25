from DAgger import make_env, ExpertPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC


if __name__ == "__main__":
    env = DummyVecEnv([make_env])
    expert = ExpertPolicy(env)
    model = SAC("MlpPolicy", env=env,buffer_size=50_000)

    obs = env.reset()
    for _ in range(10_000):
        action, _ = expert.predict(obs, deterministic=True)
        next_obs, reward, done, info = env.step(action)
        scaled_action = model.policy.scale_action(action)
        model.replay_buffer.add(obs, next_obs, scaled_action, reward, done, info)
        obs = next_obs

    model.save_replay_buffer("data/expert_buffer.pkl")
    env.close()
