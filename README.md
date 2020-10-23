# Raging Lunatics 2020

Welcome to the Raging Lunatics repository for the reproducible research assignment for the course Reinforcement Learning at the University of Amsterdam, by Anna Langedijk, Christiaan van der Vlist, David Černy and Albert Harkema. Here you will find the most recent codebase, progress and schedules.



# DQN vs MountainCar

![Alt Text](https://github.com/annaproxy/rl2020/blob/master/mtc.gif)

## Schedules

In this section you will find an overview of schedules, agreements and TO-DOs

### Schedule Week 5 (28-09 - 04-10)

* [ ] Find environments
  * [ ] Standard environments (mountain car, gridworld, windy gridworld)
  * [ ] Baird's Counterexample (known to diverge)
* [ ] Tell the story of which environments we have tried, and for which reasons we assumed, based on our experiences, that this one would potentially diverge.

### Schedule Week 6 (05-10 - 11-10)

* [ ] Make DQN code general such that we can work with it
* [ ] Detail our quest for environments and generate results
* [ ] Do a first batch of **experiments** so we ensure we have something ready for the Feedback Session at 12-10:
  * [ ] Do they logically follow? In case we get (unexpected) results, explain our thought process of why this happened and theorize why we decided to do certain follow-up experiments.
  * [ ] Ablation type studies
    * [ ] Do experience replay
    * [ ] Combine experience replay with target networks
* [ ] Make sure we write up a good first draft so we get proper feedback 


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
