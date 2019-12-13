import os
import sys
import copy
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
import gym
import gym_gvgai

from ppo import algo, utils
from ppo.envs.atari import VecPyTorch, make_vec_envs
from ppo.utils import get_render_func, get_vec_normalize
from baselines.common.vec_env.vec_normalize import VecNormalize
from ppo.storage import RolloutStorage
from collections import deque

from tensorboardX import SummaryWriter



parser = argparse.ArgumentParser(description='PPO')
parser.add_argument('--num-evals', type=int, default=10)
parser.add_argument('--num-processes', type=int, default=4)
parser.add_argument('--load-dir', type=str, default='trained_models/')
parser.add_argument('--env-name', type=str, default='PongNoFrameskip-v4')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num-steps', type=int, default=2048)
parser.add_argument('--ppo-epochs', type=int, default=10)
parser.add_argument('--num-mini-batch', type=int, default=32)
parser.add_argument('--pi-lr', type=float, default=1e-4)
parser.add_argument('--v-lr', type=float, default=1e-3)
parser.add_argument('--dyn-lr', type=float, default=1e-3)
parser.add_argument('--hidden-size', type=int, default=128)
parser.add_argument('--clip-param', type=float, default=0.3)
parser.add_argument('--value-coef', type=float, default=0.5)
parser.add_argument('--entropy-coef', type=float, default=0.01)
parser.add_argument('--grad-norm-max', type=float, default=5.0)
parser.add_argument('--dyn-grad-norm-max', type=float, default=5)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--use-gae', action='store_true')
parser.add_argument('--gae-lambda', type=float, default=0.95)
parser.add_argument('--share-optim', action='store_true')
parser.add_argument('--predict-delta-obs', action='store_true')
parser.add_argument('--use-linear-lr-decay', action='store_true')
parser.add_argument('--use-clipped-value-loss', action='store_true')
parser.add_argument('--use-tensorboard', action='store_true')
parser.add_argument('--save-frames', action='store_true', default=False)

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    load_dir = args.load_dir + args.env_id

    if args.save_frames:
        frame = 0

    # set device and random seeds
    device = torch.device("cpu")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    for i in range(3):
        # create agent
        load_dir = args.load_dir + args.env_id
        save = args.env_id + '-' + str(i+1) +'TL.pt'
        actor_critic, ob_rms = \
                        torch.load(os.path.join(load_dir,  args.env_id + '-' + str(i+1) +'TL.pt'))

        actor_critic.to(device)
        print('Model ', i+1)

        for j in range(5):
            print('Game level ', j+1)
            
            # setup environment
            name = args.env_id + '-lvl'+ str(j) + '-v0'
            eval_envs = make_vec_envs(env_name=name,
                                seed=args.seed,
                                num_processes=args.num_processes,
                                gamma=args.gamma,
                                log_dir='/tmp/gym/',
                                device=device,
                                allow_early_resets=True)

            print('Evaluating...')            
            if eval_envs.venv.__class__.__name__ == "VecNormalize":
                eval_envs.venv.ob_rms = envs.venv.ob_rms

                def _obfilt(self, obs):
                    if self.ob_rms:
                        obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
                        return obs
                    else:
                        return obs

                eval_envs.venv._obfilt = types.MethodType(_obfilt, envs.venv)

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 1:

                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                if render:
                    plt.imshow(eval_envs.render(mode='rgb_array'))
                    plt.axis('off')
                    plt.savefig('frames/' + name + '-frame' + str(frame) + '.png', bbox_inches='tight')
                    frame = frame + 1
                
                obs, reward, done, infos = eval_envs.step(action)
                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        print(info['episode']['r'])
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()
            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards), np.mean(eval_episode_rewards)))