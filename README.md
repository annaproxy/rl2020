# On the Verge of Divergence
## Reinforcement Learning 2020

Welcome to therepository for the reproducible research assignment for the course Reinforcement Learning at the University of Amsterdam, by Anna Langedijk, Christiaan van der Vlist, David Černy and Albert Harkema. Here you will find the most recent codebase for the assignment.



# DQN vs MountainCar

Below we have attached a small clip showing a resulting MountainCar run of one of our converged Deep Q-Networks which has learned to solve the MountainCar environment.

![Alt Text](https://github.com/annaproxy/rl2020/blob/master/mtc.gif)


### Usage:
```
python main.py [-h] [--hidden HIDDEN] [--memory MEMORY]
               [--env {MountainCar-v0,MountainCarContinuous-v0,CartPole-v1,Acrobot-v1}]
               [--seed SEED] [--lr LR] [--epsilon EPSILON]
               [--discount DISCOUNT] [--batch_size BATCH_SIZE]
               [--episodes EPISODES] [--epsilon_cap EPSILON_CAP]
               [--max_steps MAX_STEPS] [--target] [--no_replay] [--C C]
               [--save_amt SAVE_AMT]
```

```
  --hidden HIDDEN       Specify hidden size
  --memory MEMORY       Specify memory size for experience replay
  --env {MountainCar-v0,MountainCarContinuous-v0,CartPole-v1,Acrobot-v1}
                        Specify gym environment
  --seed SEED           Random seed
  --lr LR               Learn rate
  --epsilon EPSILON     epsilon for eps-greedy
  --discount DISCOUNT   discount factor
  --batch_size BATCH_SIZE
                        batch size
  --episodes EPISODES   batch size
  --epsilon_cap EPSILON_CAP
                        threhold for epsilon reduction
  --max_steps MAX_STEPS
                        maximum steps per episode
  --target              activate target network
  --no_replay           dont activate replay
  --C C                 how many times to save the target network
  --save_amt SAVE_AMT   save the weights to file
```
