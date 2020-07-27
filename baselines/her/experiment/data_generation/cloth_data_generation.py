import gym
import numpy as np


"""Data generation for the case of a single block pick and place in Fetch Env"""

actions = []
observations = []
infos = []

def main():
    env = gym.make('ClothReach-v1')
    numItr = 100
    initStateSpace = "random"
    env.reset()
    print("Reset!")
    while len(actions) < numItr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)


    fileName = "data_cloth"
    fileName += "_" + initStateSpace
    fileName += "_" + str(numItr)
    fileName += ".npz"
    np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file

def goToGoal(env, initial_observation):

    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    timeStep = 0 
    episodeObs.append(initial_observation)

    while True: 
        env.render()
        action = np.array([0.0, 0.0, 0.0]) + np.random.normal(0, 0.1, 3)

        if timeStep < 12:
            action[0] = 0.2
            action[1] = 0.2
            action[2] = 0.45
        elif timeStep < 40:
            action[0] = 0.2
            action[1] = 0.2
            action[2] = -0.2
        else:
            action[2] = -1

        obsDataNew, reward, done, info = env.step(action)
        if reward == 0:
            print("suc", reward, info)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        if timeStep >= env._max_episode_steps: break

    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)
    print("Lens", len(episodeAcs), len(episodeInfo), len(episodeObs))


if __name__ == "__main__":
    main()
