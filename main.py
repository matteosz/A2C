from collections import defaultdict
from copy import copy
import torch
from torch import multiprocessing as mp
import gymnasium as gym
from model import A2C
import json

ENV_NAME = 'CartPole-v1'
MAX_STEPS = int(5e5)
EVAL_STEPS = int(2e4)
LOG_STEPS = int(1e3)
EVAL_EPISODES = 10

AGENTS = [2, 42, 242]

DEBUG = False

class Data:
    def __init__(self, agent):
        self.agent = agent
        self.logs = ""
        self.plots = defaultdict(list)

    def append(self, log):  
        self.logs += log
        if len(self.logs) > 1e4:
            with open(f'logs_{self.agent}.txt', 'a+') as f:
                f.write(self.logs)
            self.logs = ""

    def log(self):
        with open(f'logs_{self.agent}.txt', 'a+') as f:
            f.write(self.logs)
        with open(f'plots_{self.agent}.json', 'w') as t:
            json.dump(self.plots, t)

def collect_data(env, model, state, n):
    tot_reward, discounted_reward = .0, .0

    first_action = None
    i = 0
    while i < n:
        action = model.sample_action(state)
        if i == 0:
            first_action = action
        state, reward, terminated, truncated, _ = env.step(action)

        tot_reward += reward
        discounted_reward += model.gamma ** i * reward

        i += 1
        done = terminated or truncated
        if done:
            break
        
    discounted_reward += model.gamma ** i * model.get_reward(state) * (1 - terminated)
      
    return state, first_action, tot_reward, discounted_reward, done

def evaluate(model, data, render=False):
    env = gym.make(ENV_NAME, render_mode='human' if render else None)
    episode_rewards = .0

    states = []
    values = []
    for i in range(EVAL_EPISODES):
        state, _ = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                best_action = model.get_best_action(state)
                state, reward, terminated, truncated, _ = env.step(best_action)
                episode_rewards += reward
                done = terminated or truncated
                if render:
                    env.render()

                if i == EVAL_EPISODES - 1:
                    states.append(state)
                    values.append(model.get_reward(state).item())

    env.close()
    data.plots['3'].append([states, values])

    return episode_rewards / EVAL_EPISODES

def run_episode(env, model, n, iterations, lock, current_reward, n_episodes, data):
    state, _ = env.reset()
    tot_reward = .0

    while iterations.value <= MAX_STEPS:
        with torch.no_grad():
            next_state, action, reward, discounted_reward, done = collect_data(env, copy(model), state, n)
        tot_reward += reward

        with lock:
            actor_loss, critic_loss = model.update_step(state, action, discounted_reward)
        
            if iterations.value % LOG_STEPS == 0:
                append_log = f"Iteration {iterations.value}:\n"
                append_log += f"\tActor loss = {actor_loss}\n\tCritic loss = {critic_loss}\n"
                append_log += f"Current average episodic rewards = {current_reward / n_episodes:.2f}\n"
                
                data.plots['1'].append([n_episodes, current_reward / n_episodes])
                data.plots['4'].append([actor_loss, critic_loss])
                
                if iterations.value % EVAL_STEPS == 0:
                    eval_reward = evaluate(model, data)
                    append_log += f'Average evaluation episodic reward = {eval_reward:.2f}\n'
                    data.plots['2'].append(eval_reward)

                append_log += '--------------\n'
                print(append_log)
                data.append(append_log)
            
            iterations.value += n
        
        if done:
            break
        state = next_state

    return tot_reward

def worker_fn(idx, envs, model, n, iterations, lock, data):
    episodes, tot_reward = 0, .0
    while iterations.value <= MAX_STEPS:
        r = run_episode(envs[idx], model, n, iterations, lock, tot_reward, episodes, data)
        tot_reward += r
        episodes += 1
        if DEBUG:
            print(f'#{episodes:06d} - {r:.2f}')
    
    # Log the data and evaluate the model one last time (only 1 worker performs this)
    with lock:
        if iterations.value < 2 * MAX_STEPS:
            evaluate(model, data, render=True)
            data.log()

            iterations.value = 3 * int(MAX_STEPS)

def a2c(k=1, n=1, agent=0):
    torch.manual_seed(agent)
    data = Data(agent)
    print(f'Running A2C with {k} workers and {n} steps (Agent #{agent})...')

    envs = [gym.make(ENV_NAME)] * k
    num_actions, num_states = envs[0].action_space.n, envs[0].observation_space.shape[0]
    model = A2C(input_size=num_states, output_size=num_actions)

    lock = mp.Lock()
    iterations = mp.Value('i', 1)

    mp.spawn(worker_fn, args=(envs, model, n, iterations, lock, data), nprocs=k)

    print('A2C finished successfully! Closing environments...')
    for env in envs:
        env.close()

if __name__ == '__main__':
    for agent in AGENTS:
        a2c(agent=agent)
        break