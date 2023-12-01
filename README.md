# rl-program

Input：
- guide info (2,)
- camera info (3,)

Output:
- da,de,df

Feature:
- noise

break into :
1. frame buffer, to get the capture state -> flag to judge identifacation
2. noise guidance infomation  -> clear one

# Issuse
强化学习在其中扮演的角色
- 输入guid info
- 输出角度，焦距的增量

## 如何评估一个agent的performance
现有中，评估一个agent的performance来自于计算一个episode的得分，而这个得分，在现有的程序中即为reward，如下面的`total_reward`
```
for episode in range(max_episodes):
    observation = env.reset()
    total_reward = 0
    done = False

    while not done:
        #print(f"{observation}")
        action = agent.select_action(observation)
        #print(f"{action}")
        next_observation, reward, done, _ = env.step(action)
        agent.buffer.push(observation, action, reward, next_observation, done)
        total_reward += reward

        observation = next_observation
        # SAC算法中的训练步骤
        for i in range(3):
            agent.update_parameters(agent.buffer, 1024, updates)
            updates += 1  # 每次更新后递增更新次数

    episode_rewards.append(total_reward)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
```
这种设计是否合理？

目前的情况是，随着我对奖励函数进行调整，total_reward会发生变化，相当于前后模型用的不同的性能评价指标。

其他一些经典的环境，往往奖励函数都是固定的且得到公认的，因此不同论文在这上面做实验，可以把`total_reward`当作性能评价指标。

而在我们这种还处于**环境搭建**阶段，是不是找到另外一种不变量更为合理，比如用了多少步才找到uav

## 性能表现不佳的情况下，首先从哪方面找原因
在我看来，强化学习项目最主要的是agent的**架构设计**和**环境的奖励设计**。
### agent的架构
从原始的policy gradient，到PPO，或者actor critic到SAC，SAC++，还有其他各种算法，通常考虑模型如何更好地优化。因此，他们的性能评价往往是在多个环境上进行测试。

- PG->PPO 增加重要性采样，可以重用过去的trajectory
- PG->AC 增加critic模型，减少方差
- AC->SAC 增加entropy概念，鼓励模型进行探索
- SAC->SAC++ 调整为duel Q架构，自动调整SAC中的temperature 超参数
这些模型的性能评价中，大多数是在atri的50多个游戏环境中进行实验。也并没有说100%超越，会存在个别游戏环境不如前者，比如SAC->SAC++

### 环境的奖励设计
一些新的环境在发表论文时，工程向的论文