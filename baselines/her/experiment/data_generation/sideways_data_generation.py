import gym
import numpy as np
import math
import time
import glfw


"""Data generation for the case of a single block pick and place in Fetch Env"""

actions = []
observations = []
infos = []

from pynput import mouse

class Actor(object):
    def __init__(self, env):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.init_x = 0.0
        self.init_y = 0.0
        self.record = False
        self.env = env
        self.rec_loc_inited = False
        self.t = 0
        self.grasp = 1
        self.gain = 0.05
        self.momentum = 0.5

    def set_values(self,value):
        print("value", value)
        if value == 80:
            self.y += self.gain
            self.x *= self.momentum
            self.z *= self.momentum
        elif value == 59:
            self.y -= self.gain
            self.x *= self.momentum
            self.z *= self.momentum
        elif value == 39:
            self.x += self.gain
            self.y *= self.momentum
            self.z *= self.momentum
        elif value == 76:
            self.x -= self.gain
            self.y *= self.momentum
            self.z *= self.momentum
        elif value == 81:
            self.z += self.gain
            self.x *= self.momentum
            self.y *= self.momentum
        elif value == 87:
            self.z -= self.gain
            self.x *= self.momentum
            self.y *= self.momentum
        elif value == 91:
            self.record = True
            self.rec_loc_inited = False
            print("Recording turned on")
        elif value == 79:
            self.env.reset()
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.init_x = 0.0
            self.init_y = 0.0
            self.t = 0.0
            self.grasp = 1
            self.record = False
        elif value == 65:
            self.grasp = -1
        

    def get_action(self):
        return [self.x, self.y, self.z, self.grasp]


    def on_move(self,x, y):
        if not self.rec_loc_inited:
            self.init_x = x
            self.init_y = y
            self.rec_loc_inited = True
        if self.record:
            self.x = (-self.init_x +x)/100
            self.y = (-y + self.init_y)/100

    def on_click(self,x, y, button, pressed):
        self.record = False
        self.z = 0.0

    def on_scroll(self,x, y, dx, dy):
        pass


def main():
    env = gym.make('ClothSideways-v1')
    actor = Actor(env)
    listener = mouse.Listener(
        on_move=actor.on_move,
        on_click=actor.on_click,
        on_scroll=actor.on_scroll)
    listener.start()
    env.set_key_callback_function(actor.set_values)
    num_itr = 100
    initStateSpace = "random"
    env.reset()

    successes = 0
    try_n = 0
    while successes < num_itr:
        try_n += 1
        obs = env.reset()
        print("ITERATION NUMBER ", try_n, "Success so far", successes)
        success = goToGoal(env, obs, actor)
        if success:
            successes += 1



    fileName = "data_cloth_sideways_as3"
    fileName += "_" + initStateSpace
    fileName += "_" + str(num_itr)
    fileName += ".npz"

    np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file
    print("Saved, success rate:", successes)

def goToGoal(env, initial_observation, actor):

    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    episodeObs.append(initial_observation)
    items = [[0., 0., 0., 1.,] ,
        [0., 0., 0., 1.] ,
        [0., 0., 0., 1.] ,
        [0., 0., 0., 1.] ,
        [0., 0., 0., 1.] ,
        [0., 0., 0., 1.] ,
        [0.,  0.,  0.1, 1. ] ,
        [0.,  0.,  0.1, 1. ] ,
        [0.,         0.01929688, 0.1 ,       1.,        ] ,
        [0.03867188, 0.11238281, 0.1 ,       1.        ] ,
        [0.0653125,  0.17683594, 0.1 ,       1.        ] ,
        [0.07914063, 0.21265625, 0.1,        1.        ] ,
        [0.08890625, 0.24820312, 0.1 ,       1.        ] ,
        [0.10023438, 0.27324219, 0.1,        1.        ] ,
        [0.11695313, 0.30570313, 0.1 ,       1.        ] ,
        [0.13265625, 0.33046875, 0.1,        1.        ] ,
        [0.14800781, 0.34851562, 0.1 ,       1.        ] ,
        [0.17625 ,   0.35800781, 0.1  ,      1.        ] ,
        [0.34578125, 0.37238281, 0.1  ,      1.        ] ,
        [0.40609375, 0.37621094, 0.1,        1.        ] ,
        [0.49902344, 0.38003906, 0.1 ,       1.        ] ,
        [0.65976563, 0.38003906, 0.1 ,       1.        ] ,
        [0.74980469, 0.36867187, 0.1  ,      1.        ] ,
        [0.80523437, 0.1715625,  0.1,        1.        ] ,
        [ 0.73570313, -0.26335938,  0.1 ,        1.        ] ,
        [ 0.34089844, -0.42804687,  0.1,         1.        ] ,
        [ 0.33136719, -0.41445312,  0.1  ,       1.        ] ,
        [ 0.33136719, -0.39242187,  0.1 ,        1.        ] ,
        [ 0.33136719, -0.39242187 , 0.1,         1.        ] ,
        [ 0.33136719, -0.39242187,  0.1 ,        1.        ] ,
        [ 0.33136719, -0.39242187,  0.1  ,       1.        ] ,
        [ 0.33136719 ,-0.39242187 , 0.1 ,        1.        ] ,
        [ 0.33136719, -0.39242187 , 0.1 ,       -1.        ] ,
        [ 0.33136719, -0.39242187 , 0.1   ,     -1.        ] ,
        [ 0.33136719, -0.39242187 , 0.1,        -1.        ] ,
        [ 0.33136719 ,-0.39242187 , 0.1  ,      -1.        ] ,
        [ 0.33136719, -0.39242187,  0.1  ,      -1.        ] ,
        [ 0.33136719 ,-0.39242187  ,0.1  ,      -1.        ] ,
        [ 0.33136719, -0.39242187 , 0.1 ,       -1.        ] ,
        [ 0.33136719, -0.39242187 , 0.1 ,       -1.        ] ,
        [ 0.92496094, -0.25386719 , 0.1 ,       -1.        ] ,
        [ 1.58167969, -0.57402344,  0.1 ,       -1.        ] ,
        [ 1.67710937, -1.20914063 , 0.1  ,      -1.        ] ,]
    t = 0
    episode_success = False

    for i, action in enumerate(items):
        ac = np.array([0.0,0.0,0.0])
        ac[0] = action[0] *0.85 + np.random.normal(0, 0.1)
        ac[1] = action[1] + np.random.normal(0, 0.1)
        ac[2] = action[2] * 1.5 + np.random.normal(0, 0.1)
        if i > 32:
            ac[0] = 0
            ac[1] = 0
            ac[2] = -1
        action = ac
        env.render()
        action = np.array(action)
        obs, reward, done, info = env.step(action)
        if reward == 0:
            print("suc", reward, info)
            episode_success = True
        t += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obs)
    while t < env._max_episode_steps:
        env.render()
        t += 1
        action = np.array([0.0,0.0,0.0])
        env.step(action)
        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obs)
    print("Lens", len(episodeAcs), len(episodeInfo), len(episodeObs))
    print("Episode success", episode_success)

    while False:
        env.render()
        action = actor.get_action()
        action = np.array(action)
        obsDataNew, reward, done, info = env.step(action)
        if actor.t == 0:
            print("Timestep", actor.t, action)
        else:
            print([act for act in action],  ",")
        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        actor.t += 1

        #if timeStep >= env._max_episode_steps: break
    if episode_success:
        actions.append(episodeAcs)
        observations.append(episodeObs)
        infos.append(episodeInfo)
    
    return episode_success


if __name__ == "__main__":
    main()






#python -m baselines.run --alg=her --env=ClothSideways-v1 --demo_file=/Users/juliushietala/Desktop/Robotics/baselines/baselines/her/experiment/data_generation/data_cloth_sideways_as3_random_100.npz --num_demo=100 --num_timesteps=50000 --demo_batch_size=128 --bc_loss=1 --q_filter=1 --prm_loss_weight=0.001 --aux_loss_weight=0.0078 --n_cycles=20 --batch_size=1024 --random_eps=0.1 --noise_eps=0.1 --play --save_path=/Users/juliushietala/Desktop/Robotics/policies/her/sideways50k 
# OPENAI_LOG_FORMAT=stdout,csv,tensorboard python -m baselines.run --alg=her --env=ClothReach-v1 --demo_file=/Users/juliushietala/Desktop/Robotics/baselines/baselines/her/experiment/data_generation/data_cloth_random_100.npz --num_demo=100 --num_timesteps=1500 --demo_batch_size=128 --bc_loss=1 --q_filter=1 --prm_loss_weight=0.001 --aux_loss_weight=0.0078 --n_cycles=20 --batch_size=1024 --random_eps=0.1 --noise_eps=0.1 --play --save_path=/Users/juliushietala/Desktop/Robotics/policies/her/diag30k --log_path=/Users/juliushietala/Desktop/Robotics/logs/cloth_diagonal --num_train=2
