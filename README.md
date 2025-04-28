# Design Notes

Light weight package for bam gym environments

- Control remote gym environments running rosbridge_server using roslibpy

- Idea is that anyone can download this package and rapidly start developing with the familar gym api

- These environments can quickly tested on a ros free workspace, or also be wrapped by a ros server node 

### Environments

This repository hosts the examples that are shown [on the environment creation documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).

- `GridWorldEnv`: Simplistic implementation of gridworld environment

These are local gym libraries

local bam environments
- classic_bandit
- mnist

Transport bam environments
running on remote computers

## Installation

Tested on Ubuntu Noble

To install your new environment, run the following commands:

1. First download repo
- If Using ROS its reccomeded you do in different directory than bam_ws, as cd-autocomplete becomes a pain! Either in bam_ws/src/ or in other_bam_packages/

2. Install depencies

```
sudo apt update

sudo apt install python3-pip python3-venv

```

2. Install package into your virtual environment

- This assumes you already have a virtual environment for your development, please activate it.
- If you don't have one and want to standlone test, create a new virtual env

```{shell}
cd bam_gym
python3 -m venv --copies .venv
source .venv/bin/activate

```

Make sure your virtual env is active for the next steps 

Upgrade packages, then install

--

Determine your CUDA version and install correct torch. 
```
nvidia-smi

python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

```

```{shell}
python3 -m pip install --upgrade pip setuptools wheel build hatchling
cd /bam_gym
python3 -m pip install -e .
```

```
python3 -m pip install gymnasium pygame

```

Check its installed correctly into your virtual env

```{shell}
python3 -m pip list
```

### Considerations for ROS development

Reccomended
- Install the package twice. Once inside ROS workspace (/bam_ws) and once in a ROS free standalone env (/other_bam_packages)
- Do all the development inside the standalone env, to make sure your not dependening on any ROS this
- Update and pull changes to the ROS workspace as needed

# Design Notes

"Reward is all you need" - [Sutton & Silver](https://www.sciencedirect.com/science/article/pii/S0004370221000862)

The goal of this package is to provide a light weight inerface for the BAM hardware.

The MDP interface really captures the essence of what we want to do here.

It's a super powerful abstraction, with a common set of language and open source tools (gym, rllib, etc.)


- It depends only on the same things needed to run gym envionrments (like cart_pole)
- gynamisum, pygame, numpy, etc.
- Install open cv and cv bridge for parsing images

Makes it very easy to do experiments on simple environments, and then switch to more complex ones by just changing a single line of code!

## Transport Layer

In BAM envs, generally the actual server is running remotely (simulation, real robot, etc)

By having a modular transport layer, you can implement any communication method (rclpy, roslibpy, custom tcp, etc.)

observation, reward, terminated, truncated, info = transport.step(action)


### roslibpy
https://roslibpy.readthedocs.io/en/latest/

Requires for rosbridge to be running on server computer

```ros2 launch rosbridge_server rosbridge_websocket_launch.xml```

- default port is 9090

#### Installing rosbridge

```sudo apt-get install ros-jazzy-rosbridge-suite```

https://wiki.ros.org/rosbridge_suite/Tutorials/RunningRosbridge

```ModuleNotFoundError: No module named 'bson'```

I belive caused by an issue of virtual env overriding system package. For now launching it outside of bam_ws works

Update: You cannot launch outside of bam_ws as then it won't have access to the bam_msgs package

```all_service InvalidModuleException: Unable to import bam_msgs.srv from package bam_msgs. Caused by: No module named 'bam_msgs'```

Install bson, which is a submodule of pymongo

``` python3 -m pip install pymongo ```

ros2 launch rosbridge_server rosbridge_websocket_launch.xml

#### Bugs



## How It can be used

1. Deployment

On a deployed BAM robot there is always live RL agent interacting with an indentical gym interface.

When you start controlling a real robot, you pause the deployed agent, and take over. 

If you are happy with your performance locally, you can freeze the weights and deploy to the real robot. 

As the deployed interface and api are indentical, you can simply copy paste. This makes CI a breeze.


2. Training

When locally training you can do online RL 

You can also run a script to collect a dataset and then do offline RL


3. Testing

Large scale training is done on a large remote GPU cluster, using an accumlated dataset that is stored on AWS. 
- Leveraging LLM infastructure for this

To verify the performance of the algorithim on completely new data, this interface can be used


3. Play/Experimentation

    Easy to poke around, and better understand the dynamics/action/observation spaces


# example agents

Folder is at top level so it imports it the same as an external