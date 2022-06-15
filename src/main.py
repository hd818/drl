from environment import *


if __name__ == "__main__":
    env = VehicleEnvironment()

    episodes = 5
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print(f"Episode {episode}: Score = {score}")
    env.close()


