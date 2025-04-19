# Design Notes

Light weight package for bam gym environments

- Control remote gym environments running rosbridge_server using roslibpy

- Idea is that anyone can download this package and rapidly start developing with the familar gym api

- These environments can quickly tested on a ros free workspace, or also be wrapped by a ros server node 

### Environments

This repository hosts the examples that are shown [on the environment creation documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).

- `GridWorldEnv`: Simplistic implementation of gridworld environment


## Installation

Tested on Ubuntu Noble

To install your new environment, run the following commands:

1. First download repo
- If Using ROS its reccomeded you do in different directory than bam_ws, as cd-autocomplete becomes a pain! Either in bam_ws/src/ or in other_bam_packages/


2. Install package into your virtual environment

- This assumes you already have a virtual environment for your development, please activate it.
- If you don't have one and want to standlone test, create a new virtual env

```{shell}
cd bam_gym_env
python3 -m venv --copies venv
source venv/bin/activate

```

Make sure your virtual env is active for the next steps 

Upgrade packages, then install

```{shell}
python3 -m pip install --upgrade pip setuptools wheel build hatchling
python3 -m pip install -e .
```

Check its installed correctly into your virtual env

```{shell}
python3 -m pip list
```
