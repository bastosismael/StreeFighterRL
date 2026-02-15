import retro
import time

env = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis")
env.metadata["render_fps"]=10
obs = env.reset()
done = False
n_episodes = 1
for game in range(n_episodes):
    while not done:
        env.render()
        time.sleep(0.01)
        obs, reward, done, info, s = env.step(env.action_space.sample())
        print(reward)
        
