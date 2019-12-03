import argparse
import gym
import gym_gvgai
from random import randint
from tensorboardX import SummaryWriter
import numpy as np

tbwriter = SummaryWriter()

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
args = parser.parse_args()


games = []
for i in range(5):
    name = args.env_name + '-lvl' + str(i) + '-v0'
    games.append(name)


for j in range(5):
    print(games[j])

    env = gym.make(games[j])
    actions = env.unwrapped.get_action_meanings()

    rewards = []
    for k in range(100):
        episode_rewards = 0
        obs = env.reset()
        while True:
            obs, reward, done, info = env.step(randint(0,len(actions)-1))    
            episode_rewards +=  reward
            if done:
                break
        
        env.close()
        rewards.append(episode_rewards)

    print(np.mean(rewards), j)
    name = 'random'
    tbwriter.add_scalar(name, np.mean(rewards), j)