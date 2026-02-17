from env import StreetFighter
import time

env = StreetFighter()
obs, _ = env.reset()
done = False
n_episodes = 1
for game in range(n_episodes):
    while not done:
        env.render()
        time.sleep(0.01)
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        if reward > 0:
            print(reward)
        