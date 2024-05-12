from collections import defaultdict
import torch
from torch import multiprocessing as mp
import gymnasium as gym
from model import A2C
import json

ENV_NAME = 'CartPole-v1'
MAX_STEPS = 5e5
EVAL_STEPS = 2e4
LOG_STEPS = 1e3
EVAL_EPISODES = 10

AGENTS = [2, 42, 82]
CURRENT_AGENT = 0

class Data:
    def __init__(self):
        self.logs = ""
        self.plots = defaultdict(list)

    def append(self, log):  
        self.logs += log
        if len(self.logs) > 1e3:
            with open(f'logs_{CURRENT_AGENT}.txt', 'a+') as f:
                f.write(self.logs)
            self.logs = ""

    def log(self):
        with open(f'logs_{CURRENT_AGENT}.txt', 'a+') as f:
            f.write(self.logs)
        with open(f'plots_{CURRENT_AGENT}.json', 'w') as t:
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
        discounted_reward += model.gamma ** i * float(reward)

        i += 1
        if truncated:
            break
        
    discounted_reward += model.gamma ** i * model.get_reward(state) * (1 - terminated)
      
    return state, first_action, tot_reward, discounted_reward, truncated or terminated

def evaluate(model, data, last=False):
    if last:
        env = gym.make(ENV_NAME, render_mode='human')
    else:
        env = gym.make(ENV_NAME)
    episode_rewards = .0

    states = []
    values = []

    for i in range(EVAL_EPISODES):
        state, _ = env.reset()
        done = False

        while not done:
            best_action = model.get_best_action(state)
            state, reward, terminated, truncated, _ = env.step(best_action)
            episode_rewards += reward
            done = terminated or truncated
            #if last:
            #    env.render()

            if i == EVAL_EPISODES - 1:
                states.append(state)
                values.append(model.get_reward(state).detach().item())

    env.close()

    data.plots['3'].append([states, values])

    return episode_rewards / EVAL_EPISODES

def run_episode(env, model, n, iterations, lock, current_reward, n_episodes, data):
    state, _ = env.reset()
    tot_reward = .0
    done = False

    while not done and iterations.value <= MAX_STEPS:
        freezed_model = model.copy()
        state, action, reward, discounted_reward, done = collect_data(env, freezed_model, state, n)
        tot_reward += reward

        with lock:
            actor_loss, critic_loss = model.update_step(state, action, discounted_reward)
        
            if iterations.value % LOG_STEPS == 0:
                append_log = ""
                append_log += f"Iteration {iterations.value}:\n\tActor loss = {actor_loss}\n\tCritic loss = {critic_loss}\n"
                append_log += f"Average episodic rewards = {current_reward / n_episodes:.2f}\n"
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

    return tot_reward

def worker_fn(idx, envs, model, n, iterations, lock, data):
    episodes, tot_reward = 0, .0
    while iterations.value <= MAX_STEPS:
        tot_reward += run_episode(envs[idx], model, n, iterations, lock, tot_reward, episodes, data)
        episodes += 1
    
    with lock:
        if iterations.value < 2 * MAX_STEPS:
            evaluate(model, data, last=True)
        iterations.value = 2 * int(MAX_STEPS)

    data.log()

def a2c(k=1, n=1, agent=0, data=Data()):
    CURRENT_AGENT = agent
    torch.manual_seed(AGENTS[CURRENT_AGENT])
    print(f'Running A2C with {k} workers and {n} steps (Agent {CURRENT_AGENT})...')

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