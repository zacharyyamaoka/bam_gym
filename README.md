# Bam Gym

<img src="docs/system_design_v1_june_17_2025.jpg" alt="alt text" width="800"/>

## Goals

- Light weight, ROS free package, that provides gym api to BAM products
- Show how Waste sorting -> RL -> Supervised learning

- Control remote gym environments running rosbridge_server using roslibpy
- Idea is that anyone can download this package and rapidly start developing with the familar gym api
- These environments can quickly tested on a ros free workspace, or also be wrapped by a ros server node 
- This is the key API from which top level requirements flow down, then you can go into the weeds in Bam Core to implement the functionality
- We should always have a baseline which is a randomly moving robot
- Should enable ML/AI specialistic, with limited experince with ROS/hardware/low level control/C++ to rapidly develop algortihims and control the product
- Quickly test model on new data, collect datasets, get a feel for the problem
- Can connect to any Bam robot in the world/lab if you know the ip+port, and gym env that is running
- The top level example should be super simple, like 50 lines of code, similar to the gym examples

What is the agent? (aka lego, in a [How Big Things Get Done](https://timharford.com/2023/02/what-lego-can-teach-us-about-saving-the-planet/) sense)
- A robot rack. It's a self contained unit, with sensors, actuators, computer, needed to accomplish goals.

## Functional Requirements



Future Functinality
- [ ] Control world via Roslibpy 
- [ ] Reconfigure robot/env via api

## Design Parameters

### What goes inside the agent?

- Neural nets, Software 3.0 Code, Feedforward networks that map from observation to action
- High level coordination strategies for determining which object to pick

- Helper functions for:
    - Saving transitions to database
    - Dealing with async reward futures
    - Vizualisation of observations (I can likely do this in a way that simplifies if it goes to pygame or foxglove)

### Todo 

June 2025 - Working towards first deployment of learning robot
- You have the robot setup, now just play the gym game!
- [ ] Cartpole example 
- [ ] Bam API Render mode human
- [ ] Random actions 
- [ ] Baseline solution for reference 
- [ ] Conv solution for ref. (no I will leave that to someone else)
- [ ] Python + jupyter cookbook examples 


### Gym Envs

Goal with the local gym enviornments is to provide a illustration of how the BAM remote environments are similar/different to classical gym environments and 
supervised learning. Also fast/repeatable baselines that can be run when prototyping on models. No need to worry about connecting to a real agent. Can potetailly incorperate data from a very wide number of agent to get a better evaluation as well.

Local
- CartPole
    - Classic baseline environment
    - http://joschu.net/docs/nuts-and-bolts.pdf
    - http://joschu.net/blog/opinionated-guide-ml-research.html
    - https://karpathy.github.io/2019/04/25/recipe/

- Mnist
    - baseline image enviornment
    - Link to that page about RL -> supervised learning
- CIFAR-10
    - harder image environment
- Grasp Net 1 billon
    - Full grasping supervised learning problem
- BAM offline dataset

Remote
- PickAndLift
    - Indiscrimenate grasping
    - Similar to classic research: https://research.google/blog/deep-learning-for-robots-learning-from-large-scale-interaction/
    - Robot Classroom: https://rl-at-scale.github.io/


## Risks/Counter Measures

Why will I fail?

- Moving to slowly
    - Try to accomplish in an hour what would take you a day, in a day what would take you a week

## References

- BAM
    - [On Landfills and Bandits](https://www.canva.com/design/DAGfSm-pHFs/FYOadwZK48Q7wnUkWjxOSw/edit?utm_content=DAGfSm-pHFs&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

- Google
    - [2016 Sergey Learning Hand Eye @ Google](https://research.google/blog/deep-learning-for-robots-learning-from-large-scale-interaction/)
    - [RL at Scale](https://rl-at-scale.github.io/)

- [Andy Zeng](https://andyzeng.github.io/)
    - [Cloud folding](https://sites.google.com/berkeley.edu/cloudfolding)

- [Berkley Blue Python API](https://github.com/berkeleyopenarms/blue_interface?tab=readme-ov-file)
- [Ray RLLib External Env](https://docs.ray.io/en/latest/rllib/external-envs.html)
- [Moteus Python Lib](https://github.com/mjbots/moteus/tree/main/lib/python)
- ROS2 Transport layer design
- [Open AI Gym API](https://gymnasium.farama.org/)

## Acceptance Tests

- `test_local_gym.py` ✅
    - Verify you can play local gym envs locally

*require remote env to be running*

```ros2 launch bam_core_bringup gym_env.launch.py env:=CartPole```

- `test_roslibpy.py` ✅
    - Checks that a connection can be made to remote robot

- `test_remote_gym.py` ✅
    - Verify you can play cartpole remotely

- `test_random_agent.py -e GraspXYR` ❌
    - Verify that random actions work
    - See robot move in Foxglove
    - See robot observations in Local pygame

- `test_simple_v1_agent.py -e GraspXYR` ❌
    - Verify that simple heuristic agent works and rewards
    - See robot move in Foxglove
    - See robot observations in Local pygame

## Environments

- Before there was an idea to have a unique environment for each customer/application/etc. 
- Instead I now prefer to have a more general environment
- There are many, many things you would consider in a pick and place env...

- Before I wanted to send discrete variables, but now I want the interface to be richer

#### `PickAndPlace`

These enviornments parametrize a robot that does simple pick and place (robotics 2.0)

- The agent is a robot rack
- To control an entire row of robots, you can parralelize the env
- It picks up and item, and then it throws/places it at another location (Could be a machine, inside a bin, etc)
- This Env type can be used for many applications. No need to make a new Env for each application
- Should be easy to use for Agent. Default case is to just send pose you want it to grasp at. Do I want the ML engineer to think about where to drop of the items?
- Hmmm.. its a bit of added complexity, but I think its important. Otherwise core gets a bit to complicated.

- Parameterize motion with waypoints and params
###### Action Space:

- Grasps are represented by 7 DOF vector
    - Position (X, Y, Z)
    - Orientation (Rx, Ry, Rz)
    - Grasp Width

- Pick List - len (1-8):
    - 1 - 4 Target Grasps for 1-4 arms
    - Grasp Retries
    - Multiple arm grasps and grasp retries are handled in the same way (One Obs Out -> 1-8 Actions In -> 1-8 Rewards + 1 Obs)
    - The grasp approach, and where/how the item is placed is controled by the env/app config.

- Place Length (1 - 4)
 
- Other params
    - Throw, etc.
    - Expected obj class

- Can be extended to add more way points for controlling approach, etc. if desired (that would be a new environment though)


###### Observation Space:

- RGBD + Segmentation Image

###### Reward:

- 1 if grasp was succesful
- 0 otherwise

Speed of execution etc is control be robotics 2.0. Following Andy Leung and using a probability of success seems like a good approach.



###### Env Names

- `GraspXYR`
- `GraspXYRW`
- `GraspXYZRRRW`

###### Psuedo Code





---

Observation Space


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

Updated: June 4 2025
    - Just install it once in the a /python_ws
    - then install into into ros virtual env with
    - installation is with -e (editable), so you can adjust as needed!

```{shell}
python3 -m pip install -e /home/bam/python_ws/bam_gym

```

# Design Notes

![alt text](docs/seperation_of_concerns.png)

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


# Key Idea. 

First we made it look like an RL problem

Now we make it look like a supervised learning problem.

dataset.

We train for a bit, then we update the dataset with more samples. train for a bit, and reapeat!

# Saving Datasets.

- Idea 1
- Image as images in a folder, other data as a csv.

Using the replaybuffer lets you do alot of other things, in terms of having ordered data, etc.
Really this is a supervised leraning problem... How are supervised learning datasets saved? As images, actions and results... basically labels.

- Could look at how replay buffers are saved. there is some work on that, or look at how image datasets are saved... I think its more similar for now!
