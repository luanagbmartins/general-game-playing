# General Game Playing

This repository contains [Ilya Kostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) Pytorch implementation of Proximal Policy Optimization (PPO). The implementation was used to evaluate generalization in reinforcement learning using the GVGAI-GYM framework.

## Requirements

* Python 3 
* [Conda] (https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
* [OpenAI baselines](https://github.com/openai/baselines)
* [GVGAI-GYM](https://github.com/rubenrtorrado/GVGAI_GYM)

In order to install requirements, follow:

```bash
# Create conda environment
conda env create -f environment.yml

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# GVGAI-GYM framework
git clone https://github.com/rubenrtorrado/GVGAI_GYM.git
cd GVGAI_GYM
pip install -e .

```
