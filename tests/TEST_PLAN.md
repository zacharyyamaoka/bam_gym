
At the top level of the control stack

Is a policy, env -> agent node

The policy and env, have been intentionally sperated from ROS.

I think its ok to use the ros_py_types

You should be able to achieve a fairly high degree of test performance just mocking the env, or using offline envs, like graspnet, etc..

Ok yes that is the idea, perfect....

What would give me confidence the policy is correct?

1. basic syntax etc
2. Performance (the most important thing, and requires everything to be right already....)

Can I use the random policy on grasp net??? now that would be cool...
Bam Offline data...? yup.

---
Lets focus though back on critical path for UR pick up.

What policy am I going to use? Blind policy. what env? Pick env, with mocked..