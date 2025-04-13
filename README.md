# Gymnasium Examples

Some simple examples of Gymnasium environments and wrappers.
For some explanations of these examples, see the [Gymnasium documentation](https://gymnasium.farama.org).

### Environments

This repository hosts the examples that are shown [on the environment creation documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).

- `GridWorldEnv`: Simplistic implementation of gridworld environment

### Wrappers

This repository hosts the examples that are shown [on wrapper documentation](https://gymnasium.farama.org/api/wrappers/).

- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range
- `DiscreteActions`: An `ActionWrapper` that restricts the action space to a finite subset
- `RelativePosition`: An `ObservationWrapper` that computes the relative position between an agent and a target
- `ReacherRewardWrapper`: Allow us to weight the reward terms for the reacher environment

### Contributing

If you would like to contribute, follow these steps:

- Fork this repository
- Clone your fork
- Set up pre-commit via `pre-commit install`

PRs may require accompanying PRs in [the documentation repo](https://github.com/Farama-Foundation/Gymnasium/tree/main/docs).

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
