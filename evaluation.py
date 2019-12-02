import argparse
import os
import sys
import copy
import numpy as np
import torch
import gym
import gym_gvgai
from ppo.envs.atari import VecPyTorch, make_vec_envs
from ppo.utils import get_render_func, get_vec_normalize
from tensorboardX import SummaryWriter

sys.path.append('ppo')
tbwriter = SummaryWriter()

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--training-levels',
    type=int,
    default=2,
    help='amount of training levels')
args = parser.parse_args()

args.det = not args.non_det
load_dir = args.load_dir + args.env_name


games = []
for i in range(5):
    name = args.env_name + '-lvl' + str(i) + '-v0'
    games.append(name)

for i in range(3):
    env_name = args.env_name + '-' + str(i+1) + 'TL'
    actor_critic, ob_rms = \
                    torch.load(os.path.join(load_dir, env_name + ".pt"))

    print('Evaluating for ' + str(i+1) + ' training level')

    for j in range(5):
        print(games[j])

        env = make_vec_envs(games[j], args.seed, 1, None, None, 'cpu', False)
        render_func = get_render_func(env)
        recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
        masks = torch.zeros(1, 1, device='cpu')

        rewards = []
        for k in range(100):
            episode_rewards = 0
            obs = env.reset()
            while True:
                # env.render()

                with torch.no_grad():
                    value, action, _, recurrent_hidden_states = actor_critic.act(
                        obs, recurrent_hidden_states, masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, info = env.step(action.cpu())    
                episode_rewards +=  reward.data.cpu().numpy()[0] 

                if done:
                    break
            
            env.close()
            rewards.append(episode_rewards)

        name = 'reward ' + str(i) + 'training level'
        tbwriter.add_scalar(name, np.mean(rewards), j)