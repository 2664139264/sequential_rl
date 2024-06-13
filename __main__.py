import gymnasium as gym

from experiment.dqn import run_dqn


if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode="rgb_array")

    result = run_dqn(env)

    print(result)