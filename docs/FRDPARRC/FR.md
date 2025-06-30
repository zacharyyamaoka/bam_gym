# Functional Requirments

---

## CONOPS

BAM agent implements the a policy to solve MDP within a bam environment.

We are used to typically doing something like

```python
import gymnasium as gym

env = gym.make("CartPole", render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

Which is fast/good for scripting, but what if we want to deploy this in a robot that continually works, has fault recovery, obeys business logic, etc.

`bam_agent` package provides the base classes and helpers to do this!



## FRs

1. Pass` observation`, `info` to `policy()`, and pass action to the env client
2. Follow business logic
    - 2.1 Start/Stop/Reset
    - 2.2 Slow Down/Speed Up
    - 2.3 Change which items you are picking up
3. Diagonistic logging
4. Handle faults 
5. Loading new weights for network? tbd...


