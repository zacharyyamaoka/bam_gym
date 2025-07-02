#!/usr/bin/env python3

"""

Launch Server with:
ros2 launch bam_core_bringup gym_env.launch.py env:=MockEnv

or

ros2 launch bam_core_bringup gym_env.launch.py env:=CartPole

Make sure that namespaces match (ie. 'bam_GPU')

Problems I am seeing

1. it says its not been reset?
2. Sim time look up error

I am suprised by how long this takes to communicate. Only like 1 step per second, its far to slow!

But its inline with what I was getting doing teleop over bridge....

Cloud robotics doesn't really work that great for these images! I even had trouble with the teleop, ethernet was good...

I was expected it to be faster on the own computer though... I am sending very large images though, and depth and mask...
vs a lower res RGB image.


I still think that having the GYM API is a good idea, and having it with lists... its a general purpose API for input output of a machine.

Cost for the additoinal sturcutre I think its worth it to have the one type.

You shouldn't need to worry about how its transported. if in the future you use webrtc, then you can transport, and then construct the type on the other side

So lets keep GYM API. Is there gains to be had though internall within core by utilizing image transport better?

Ok so I think it makes sense that there is a single node, where I read in the camera images...

Do object detection if needed, and then send them to gym? 

There needs to be decompression to run object detection. and then the result can be passed as an mask/polgyon?

There needs to be not nessaricly decompression but reading to combine the messages into a gym API response.

I could read the items before hand in the camera API... and then the gym can just read the camera API, advntage is that its bundled to the gym,
Doesn't care about the copying compelxioty. the downside is that you have to read the obejcts into memory I think, you cannot use the message by reference
if you keep them on their own images...
But then then potetially different nodes will need to buffer...

Ok I think the correct thing is to read them into the camera server, run object detection there if needed, and bundle together,
Then you can query it from the camera server....

What about gazebo though?
----

Ok the big bottle neck now is the roslibpy transport, to send 5mb image is to much!!!
Runs super slowly....

"""
import gymnasium as gym
from bam_gym.envs import ObsEnv
from ros_py_types.bam_msgs import ErrorCode, ErrorType
from bam_gym.transport import RoslibpyTransport
from bam_gym.utils import print_step, is_env_success


transport = RoslibpyTransport(namespace="bam_GPU")
env = ObsEnv(transport=transport, render_mode="human")

print(env.action_space)
print(env.observation_space)

observation, info = env.reset(seed=1337)

assert is_env_success(info)

for _ in range(100):
    action = env.action_space.sample(mask=(1,None)) # Mask sequence to len(1)
    new_observation, reward, terminated, truncated, info = env.step(action)

    print_step(_, observation, action, reward, terminated, truncated, info)
    observation = new_observation

    # Handle error - Simulated environments always return observations, but sometimes
    # real environments have issues! need to access full info and check if success
    if info["header"]["error_code"]["value"] != ErrorType.SUCCESS:
        print("Skipping this step due to error.")
        print("Error message:", info["header"].get("error_msg", ""))
        continue

    # No need to reset as env auto resets
    if False and (terminated[0] or truncated[0]):
        print("Reseting")
        observation, info = env.reset()


env.close()