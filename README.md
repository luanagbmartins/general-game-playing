# General Game Playing

In major problems, reinforcement learning systems should use parameterized function approximators, such as neural networks, to generalize between similar situations and actions. Although proven to be an effective method for specific complex problems, they still fail in the generalization aspect, and the most common reinforcement learning benchmarks still use the same environments for both training and testing. To be able to evaluate the ability of an algorithm to generalize across tasks requires benchmarks that measure its performance on a set of tests that are distinct from those used in training. Therefore, this work aims to evaluate the performance of the Proximal Policy Optimization (PPO) algorithm in the General Video Game Artificial Intelligence (GVGAI) benchmark that provides a subdivision of a virtual world of a game in different stages or levels. Although PPO generally reports great results, it can be noted that the algorithm suffers from overfitting to the training set.

## Requirements

* Python 3 
* [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
* [OpenAI baselines](https://github.com/openai/baselines)
* [GVGAI-GYM](https://github.com/rubenrtorrado/GVGAI_GYM)

In order to install requirements, follow:

```bash
# Create conda environment
conda env create -f environment.yml
conda activate RL

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# GVGAI-GYM framework
git clone https://github.com/rubenrtorrado/GVGAI_GYM.git
cd GVGAI_GYM
pip install -e .

```

## Usage
To run the code, simply execute `python main.py` after installing all the requirements. There are many customizable hyperparemeters and configurations. You can see them with `python main.py --help`.

## Experiments

The experiments were carried out using three games from GVGAI-GYM framework:
- gvgai-aliens
- gvgai-boulderdash
- gvgai-missilecommand

To see other games available in GVGAI-GYM framework click [here](https://github.com/rubenrtorrado/GVGAI_GYM/tree/master/gym_gvgai/envs/games)

## Credits
This repository contains [Ilya Kostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) Pytorch implementation of Proximal Policy Optimization (PPO). The implementation was used to evaluate generalization in reinforcement learning using the GVGAI-GYM framework.
