# train.py
# Code inspired from this gymnasium tutorial: https://gymnasium.farama.org/tutorials/gymnasium_basics/vector_envs_tutorial/
import numpy as np
import torch
from tqdm import tqdm
import random
import gymnasium as gym
from model import A2C, ENTROPY_CORRECTION, CONTINUOUS
import matplotlib.pyplot as plt
import os
import json
from copy import deepcopy

SEEDS = [2, 39, 242] 

MAX_STEPS = 500000
EVAL_STEPS = 20000
LOG_STEPS = 1000
EVAL_EPISODES = 10

STOCHASTIC = True
ENV_NAME = 'InvertedPendulum-v4' if CONTINUOUS else 'CartPole-v1'
ONLY_PLOT = True

gym.logger.set_level(40) # silence warnings

logged_actor_losses_all_seeds = []
logged_critic_losses_all_seeds = []
logged_entropies_all_seeds = []
logged_episodic_returns_train_all_seeds = []
logged_episodic_returns_eval_all_seeds = []
logged_eval_value_func_all_seeds = []

logged_x_vals_train = []
logged_x_vals_eval = []

logged_actor_losses_curr_seed = []
logged_critic_losses_curr_seed = []
logged_entropies_curr_seed = []
logged_episodic_returns_train_curr_seed = []
logged_episodic_returns_eval_curr_seed = []
logged_eval_value_func_curr_seed = []

eval_info = [0, 0, '']

def set_seed(agent):
    torch.manual_seed(agent)
    np.random.seed(agent)
    random.seed(agent)

def plot_value_function(x_vals, y_vals):
    if not os.path.exists('plots'):
        os.makedirs('plots')

    if not os.path.exists('plots/evaluations'):
        os.makedirs('plots/evaluations')
    
    plt.figure(figsize=(10, 6))

    plt.plot(x_vals, y_vals)

    plt.title("Value function(seed" + str(eval_info[0]) + ", " + "eval nr." + str(eval_info[1]) + ")")
    plt.xlabel('Steps')
    plt.ylabel('Value function')

    path = 'plots/evaluations/plot_value_func_evaluation_seed' + str(eval_info[0]) + "_eval_nr" + str(eval_info[1]) + '.png'
    plt.savefig(path)
    plt.close()

def evaluate(model, step, render=False):
    env = gym.make(ENV_NAME, render_mode='human' if render else None)
    episode_rewards = .0

    values = []
    for i in range(EVAL_EPISODES):
        state, _ = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                if CONTINUOUS:
                    best_action = model.select_best_action_continuous(state[None, :])[0]
                else:
                    best_action = model.select_best_action(state[None, :])
                state, reward, terminated, truncated, _ = env.step(best_action)
                episode_rewards += reward
                done = terminated or truncated
                if render:
                    env.render()

                if i == EVAL_EPISODES - 1:
                    values.append(model.get_value(state).item())

    env.close()
    mean_value = episode_rewards / EVAL_EPISODES
    print(f'Step: {step}, Average evaluation reward: {mean_value:.2f}')
 
    logged_eval_value_func_curr_seed.append(float(np.mean(values)))
    logged_episodic_returns_eval_curr_seed.append(mean_value)

    plot_value_function(list(range(len(values))), values)

    eval_info[1] += 1

def a2c(k=1, n=1, seed=2):
    n_updates = MAX_STEPS // n
    critic_losses, actor_losses, entropies = [], [], []
    total_steps = 0

    envs = gym.vector.make(ENV_NAME, num_envs=k)
    num_states = envs.single_observation_space.shape[0]
    action_dim = 1 if CONTINUOUS else envs.single_action_space.n
    model = A2C(num_states, action_dim, k, n)
    envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=k*n_updates)

    states, _ = envs_wrapper.reset(seed=seed)

    are_logged_x_vals_eval_collected = len(logged_x_vals_eval) == 0
    are_logged_x_vals_train_collected = len(logged_x_vals_train) == 0

    for _ in tqdm(range(1, n_updates + 1)):
        ep_value_preds = torch.zeros(n, k)
        ep_rewards = torch.zeros(n, k)
        ep_action_log_probs = torch.zeros(n, k)
        masks_end_of_ep = torch.zeros(n, k)
        masks_bootstrap_values = torch.zeros(n, k)
        mask_is_terminated = torch.zeros(n, k)

        for i in range(n):
            masks_end_of_ep[i] = torch.tensor([i + 1] * k)

        # Collect data
        for step in range(n):
            if CONTINUOUS:
                actions, action_log_probs, state_value_preds, entropy = model.select_action_continuous(states)
            else:
                actions, action_log_probs, state_value_preds, entropy = model.select_action(states)
            states, rewards, terminated, truncated, _ = envs_wrapper.step(actions.cpu().numpy())

            # Masking the rewards s.t. the reward is zeroed out with a probability of 0.9
            if STOCHASTIC:   
                mask = np.random.choice([0, 1], size=(k,), p=[0.9, 0.1])
                rewards = rewards * mask

            ep_value_preds[step] = state_value_preds.squeeze()
            ep_rewards[step] = torch.tensor(rewards, requires_grad=True)
            ep_action_log_probs[step] = action_log_probs.squeeze()
            
            for (i, (term, trunc)) in enumerate(zip(terminated, truncated)):
                if term or trunc:
                    masks_end_of_ep[step][i] = step
                if trunc:
                    estimated_value = model.get_value(states[i])
                    masks_bootstrap_values[step][i] = estimated_value
                if term:
                    mask_is_terminated[step][i] = 1

            #gym vectorized environments automatically reset environments when envs are in done state
            #cf p.24 https://gym-docs.readthedocs.io/_/downloads/en/feature-branch/pdf/
            #testing showed using wrapper wrappers.RecordEpisodeStatistics does not effect this behaviour
            #therefore, no need to manually reset environments in terminated or truncated states

        for i in range(k):
            masks_end_of_ep[n - 1][i] = n - 1
        
        estimated_values = model.get_value(states).squeeze()
        masks_bootstrap_values[n - 1]  = estimated_values

        discount_rewards = model.get_discounted_r(ep_rewards, masks_end_of_ep, masks_bootstrap_values, mask_is_terminated)
        critic_loss, actor_loss = model.get_losses(discount_rewards, ep_action_log_probs, ep_value_preds, entropy)

        model.update_parameters(critic_loss, actor_loss)

        critic_losses.append(critic_loss.item())
        actor_losses.append(actor_loss.item())
        entropies.append(entropy.mean().item())

        skipped_log = total_steps % LOG_STEPS + n > LOG_STEPS
        skipped_eval = total_steps % EVAL_STEPS + n > EVAL_STEPS

        total_steps += n

        # Logging and evaluation
        if total_steps % LOG_STEPS == 0 or skipped_log:
            print(f'Step: {total_steps}, Critic Loss: {np.mean(critic_losses)}, Actor Loss: {np.mean(actor_losses)}, Entropy: {np.mean(entropies)}')

            avg_reward = .0
            if len(envs_wrapper.return_queue) < k:
                avg_reward = np.array(envs_wrapper.return_queue).mean()
            else:
                avg_reward = np.array(envs_wrapper.return_queue)[-k:].mean()
                
            print(f'Step: {total_steps}, Average reward: {avg_reward}')

            logged_actor_losses_curr_seed.append(float(actor_losses[-1]))
            logged_critic_losses_curr_seed.append(float(critic_losses[-1]))
            logged_entropies_curr_seed.append(float(entropies[-1]))
            logged_episodic_returns_train_curr_seed.append(float(avg_reward))

            if are_logged_x_vals_train_collected:
                logged_x_vals_train.append(total_steps)

        if total_steps % EVAL_STEPS == 0 or skipped_eval:
            evaluate(model, total_steps)
            if are_logged_x_vals_eval_collected:
                logged_x_vals_eval.append(total_steps)

    envs_wrapper.close()

    logged_actor_losses_all_seeds.append(deepcopy(logged_actor_losses_curr_seed))
    logged_critic_losses_all_seeds.append(deepcopy(logged_critic_losses_curr_seed))
    logged_entropies_all_seeds.append(deepcopy(logged_entropies_curr_seed))
    logged_episodic_returns_train_all_seeds.append(deepcopy(logged_episodic_returns_train_curr_seed))
    logged_episodic_returns_eval_all_seeds.append(deepcopy(logged_episodic_returns_eval_curr_seed))
    logged_eval_value_func_all_seeds.append(deepcopy(logged_eval_value_func_curr_seed))

    logged_actor_losses_curr_seed.clear()
    logged_critic_losses_curr_seed.clear()
    logged_entropies_curr_seed.clear()
    logged_episodic_returns_train_curr_seed.clear()
    logged_episodic_returns_eval_curr_seed.clear()
    logged_eval_value_func_curr_seed.clear()

    eval_info[1] = 0

def create_one_plot(x_vals, data_seeds, title, label, ylabel, path, only_min_max=True, explicit_avg=False):
    plt.figure(figsize=(10, 6))

    min_values = []
    max_values = []
    avg_values = []
    for i in range(len(data_seeds[0])):
        min_val = min([data_seed[i] for data_seed in data_seeds])
        min_values.append(min_val)
        max_val = max([data_seed[i] for data_seed in data_seeds])
        max_values.append(max_val)
        avg_val = sum([data_seed[i] for data_seed in data_seeds]) / len(data_seeds)
        avg_values.append(avg_val)

    if not only_min_max:
        markers = ['o', 's', '^']
        for i, (data_seed, marker) in enumerate(zip(data_seeds, markers)):
            plt.plot(x_vals, data_seed, label=label+f' (seed {SEEDS[i]})', marker=marker)
    
    plt.plot(x_vals, min_values, label=label+' (min)')
    plt.plot(x_vals, max_values, label=label+' (max)')
    if not explicit_avg:
        plt.fill_between(x_vals, min_values, avg_values, color='gray', alpha=0.3)
        plt.fill_between(x_vals, avg_values, max_values, color='gray', alpha=0.3)
    else:
        plt.plot(x_vals, avg_values, label=label+' (average)')
        plt.fill_between(x_vals, min_values, max_values, color='gray', alpha=0.3)

    plt.title(title)
    plt.grid()
    plt.xlabel('Steps')
    plt.ylabel(ylabel)

    plt.legend()

    plt.savefig(path)
    plt.close()

def create_combined_plot(x_vals1, data_seeds1, x_vals2, data_seeds2, title, labels, ylabel, path):
    plt.figure(figsize=(10, 6))

    def plot_data(x_vals, data_seeds, label, alpha=0.3, colors=['lightgray', 'orange']):
        min_values = []
        max_values = []
        avg_values = []
        for i in range(len(data_seeds[0])):
            min_val = min([data_seed[i] for data_seed in data_seeds])
            min_values.append(min_val)
            max_val = max([data_seed[i] for data_seed in data_seeds])
            max_values.append(max_val)
            avg_val = sum([data_seed[i] for data_seed in data_seeds]) / len(data_seeds)
            avg_values.append(avg_val)

        plt.plot(x_vals, min_values, label=label + ' (min & max)', color=colors[0])
        plt.plot(x_vals, max_values, color=colors[0])
        plt.fill_between(x_vals, min_values, max_values, color=colors[0])
        plt.plot(x_vals, avg_values, label=label + ' (average)', color=colors[1])

    plot_data(x_vals1, data_seeds1, labels[0])
    plot_data(x_vals2, data_seeds2, labels[1], alpha=0.5, colors=['gray', 'blue'])

    plt.title(title)
    plt.grid()
    plt.xlabel('Steps')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(path)
    plt.close()

def create_plots(save_to_json=False, load_path=None, combined_plots=True):
    if not os.path.exists('plots'):
        os.makedirs('plots')

    if load_path:
        with open(load_path, 'r') as f:
            data = json.load(f)
            x_vals_train = data['x_vals_train']
            x_vals_eval = data['x_vals_eval']
            actor_losses_all_seeds = data['actor_losses']
            critic_losses_all_seeds = data['critic_losses']
            entropies_all_seeds = data['entropies']
            episodic_returns_train_all_seeds = data['episodic_returns_train']
            episodic_returns_eval_all_seeds = data['episodic_returns_eval']
            eval_value_func_all_seeds = data['eval_value_func']
    else:
        x_vals_train = logged_x_vals_train
        x_vals_eval = logged_x_vals_eval
        actor_losses_all_seeds = logged_actor_losses_all_seeds
        critic_losses_all_seeds = logged_critic_losses_all_seeds
        entropies_all_seeds = logged_entropies_all_seeds
        episodic_returns_train_all_seeds = logged_episodic_returns_train_all_seeds
        episodic_returns_eval_all_seeds = logged_episodic_returns_eval_all_seeds
        eval_value_func_all_seeds = logged_eval_value_func_all_seeds


    if combined_plots:
        create_combined_plot(
            x_vals_train, episodic_returns_train_all_seeds, 
            x_vals_eval, episodic_returns_eval_all_seeds, 
            'Average Undiscounted Trajectory Return', 
            ["Train Return", "Eval Return"], 
            "Return", 
            f'plots/plot_return_combined{eval_info[2]}'
        )

        create_combined_plot(
            x_vals_train, actor_losses_all_seeds, 
            x_vals_train, critic_losses_all_seeds, 
            'Loss Throughout Training', 
            ["Actor Loss", "Critic Loss"], 
            "Loss", 
            f'plots/plot_loss_combined{eval_info[2]}'
        )
    else:
        create_one_plot(x_vals_train, actor_losses_all_seeds, 'Actor loss throughout training', "Actor loss", "Loss", f'plots/plot_actor_loss{eval_info[2]}')
        create_one_plot(x_vals_train, critic_losses_all_seeds, 'Critic loss throughout training', "Critic loss", "Loss", f'plots/plot_critic_loss{eval_info[2]}')
        create_one_plot(x_vals_train, episodic_returns_train_all_seeds, 'Average undiscounted trajectory return throughout training', "Return", "Return", f'plots/plot_return_training{eval_info[2]}')
        create_one_plot(x_vals_eval, episodic_returns_eval_all_seeds, 'Average undiscounted trajectory return throughout evaluation', "Return", "Return", f'plots/plot_return_evaluation{eval_info[2]}')
   
    create_one_plot(x_vals_train, entropies_all_seeds, 'Entropy throughout training', "Entropy", "Entropy", f'plots/plot_entropy{eval_info[2]}')
    create_one_plot(x_vals_eval, eval_value_func_all_seeds, 'Evolution of the value function across evaluations', "Mean value of value function", "Mean value of value function", f'plots/plot_value_func_evaluation{eval_info[2]}')
    
    if save_to_json:
        with open(f'plots/data{eval_info[2][:-4]}.json', 'w') as f:
            data = {
                'x_vals_train': x_vals_train,
                'x_vals_eval': x_vals_eval,
                'actor_losses': actor_losses_all_seeds,
                'critic_losses': critic_losses_all_seeds,
                'entropies': entropies_all_seeds,
                'episodic_returns_train': episodic_returns_train_all_seeds,
                'episodic_returns_eval': episodic_returns_eval_all_seeds,
                'eval_value_func': eval_value_func_all_seeds,
            }
            json.dump(data, f)

NK = [(1, 1), (1, 6), (6, 1), (6, 6)] if not CONTINUOUS else [(1, 1), (6, 6)]
if __name__ == '__main__':
    if not ONLY_PLOT:
        for n, k in NK:
            print(f'Running A2C with {k} workers and {n} steps.')
            suffix = f'_n{n}_k{k}' + ('_stoch' if STOCHASTIC else '') + ('_entropy' if ENTROPY_CORRECTION else '') + ('_continuous' if CONTINUOUS else '') + '.png'
            eval_info[2] = suffix
            for seed in SEEDS:
                print(f'Running A2C with seed {seed}...')
                eval_info[0] = seed
                set_seed(seed)
                a2c(k=k, n=n, seed=seed)
            create_plots()

            logged_actor_losses_all_seeds.clear()
            logged_critic_losses_all_seeds.clear()
            logged_entropies_all_seeds.clear()
            logged_episodic_returns_train_all_seeds.clear()
            logged_episodic_returns_eval_all_seeds.clear()
            logged_eval_value_func_all_seeds.clear()
            logged_x_vals_train.clear()
            logged_x_vals_eval.clear()
    else:
        load_paths = [
            'plots/data_n1_k1.json',
            'plots/data_n1_k1_stoch_entropy.json', 
            'plots/data_n1_k1_stoch_entropy_continuous.json', 

            'plots/data_n1_k6_stoch_entropy.json', 
            'plots/data_n6_k1_stoch_entropy.json', 

            'plots/data_n6_k6_stoch_entropy.json',
            'plots/data_n6_k6_stoch_entropy_continuous.json',
        ]
        for path in load_paths:
            eval_info[2] = path[10:-5] + '_7'
            create_plots(load_path=path)
    
