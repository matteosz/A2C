import numpy as np
import torch
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from model import A2C, ENTROPY_CORRECTION
import matplotlib.pyplot as plt
import os
from copy import deepcopy

SEEDS = [2, 39, 242] 

ENV_NAME = 'CartPole-v1'
MAX_STEPS = int(5e5)
EVAL_STEPS = int(2e4)
LOG_STEPS = int(1e3)
EVAL_EPISODES = 10

STOCHASTIC = True

gym.logger.set_level(40) # silence warnings
#writer = SummaryWriter()

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
                best_action = model.select_best_action(state[None, :])
                state, reward, terminated, truncated, _ = env.step(best_action)
                episode_rewards += reward
                done = terminated or truncated
                env.render()

                if i == EVAL_EPISODES - 1:
                    values.append(model.get_value(state).item())

    env.close()
    mean_value = episode_rewards / EVAL_EPISODES
    print(f'Step: {step}, Average evaluation reward: {mean_value:.2f}')
    #writer.add_scalar('Evaluation/Mean_Value', mean_value, step)

    logged_eval_value_func_curr_seed.append(np.mean(values))
    logged_episodic_returns_eval_curr_seed.append(mean_value)

    plot_value_function(list(range(len(values))), values)

    #for i, values in enumerate(values):
    #    writer.add_scalar('Evaluation/Value_Function', values, i)

    eval_info[1] += 1

def a2c(k=1, n=1, seed=2):
    n_updates = MAX_STEPS // n
    critic_losses, actor_losses, entropies = [], [], []
    total_steps = 0

    envs = gym.vector.make(ENV_NAME, num_envs=k)
    num_actions, num_states = envs.single_action_space.n, envs.single_observation_space.shape[0]
    model = A2C(num_states, num_actions, k, n)
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
            actions, action_log_probs, state_value_preds, entropy = model.select_action(states)
            states, rewards, terminated, truncated, _ = envs_wrapper.step(actions.cpu().numpy())

            # Masking the rewards s.t. the reward is zeroed out with a probability of 0.9
            if STOCHASTIC:   
                mask = np.random.choice([0, 1], size=(k,), p=[0.9, 0.1])
                rewards = rewards * mask

            ep_value_preds[step] = state_value_preds.squeeze()
            ep_rewards[step] = torch.tensor(rewards, requires_grad=True)
            ep_action_log_probs[step] = action_log_probs
            
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
            #writer.add_scalar('Train/Loss/Critic', np.mean(critic_losses), total_steps)
            #writer.add_scalar('Train/Loss/Actor', np.mean(actor_losses), total_steps)
            #writer.add_scalar('Train/Loss/Entropy', np.mean(entropies), total_steps)

            avg_reward = .0
            if len(envs_wrapper.return_queue) < k:
                avg_reward = np.array(envs_wrapper.return_queue).mean()
            else:
                avg_reward = np.array(envs_wrapper.return_queue)[-k:].mean()
                
            print(f'Step: {total_steps}, Average reward: {avg_reward}')
            #writer.add_scalar('Train/Average_Reward', avg_reward, total_steps)

            logged_actor_losses_curr_seed.append(actor_losses[-1])
            logged_critic_losses_curr_seed.append(critic_losses[-1])
            logged_entropies_curr_seed.append(entropies[-1])
            logged_episodic_returns_train_curr_seed.append(avg_reward)

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

def create_one_plot(x_vals, data_seeds, title, label, ylabel, path):
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

    markers = ['o', 's', '^']
    for i, (data_seed, marker) in enumerate(zip(data_seeds, markers)):
        plt.plot(x_vals, data_seed, label=label+f' (seed {SEEDS[i]})', marker=marker)
    plt.plot(x_vals, avg_values, label=label+' (average)', marker='x')

    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel(ylabel)

    plt.legend()

    plt.fill_between(x_vals, avg_values, min_values, color='gray', alpha=0.3)
    plt.fill_between(x_vals, avg_values, max_values, color='gray', alpha=0.3)

    plt.savefig(path)
    plt.close()

def create_plots():
    if not os.path.exists('plots'):
        os.makedirs('plots')

    create_one_plot(logged_x_vals_train, logged_actor_losses_all_seeds, 'Actor loss throughout training', "Actor loss", "Loss", f'plots/plot_actor_loss{eval_info[2]}')
    create_one_plot(logged_x_vals_train, logged_critic_losses_all_seeds, 'Critic loss throughout training', "Critic loss", "Loss", f'plots/plot_critic_loss{eval_info[2]}')
    create_one_plot(logged_x_vals_train, logged_entropies_all_seeds, 'Entropy throughout training', "Entropy", "Entropy", f'plots/plot_entropy{eval_info[2]}')

    create_one_plot(logged_x_vals_train, logged_episodic_returns_train_all_seeds, 'Average undiscounted trajectory return throughout training', "Return", "Return", f'plots/plot_return_training{eval_info[2]}')
    create_one_plot(logged_x_vals_eval, logged_episodic_returns_eval_all_seeds, 'Average undiscounted trajectory return throughout evaluation', "Return", "Return", f'plots/plot_return_evaluation{eval_info[2]}')
    
    create_one_plot(logged_x_vals_eval, logged_eval_value_func_all_seeds, 'Evolution of the value function across evaluations', "Mean value of value function", "Mean value of value function", f'plots/plot_value_func_evaluation{eval_info[2]}')

NK = [(1, 1), (1, 6), (6, 1), (6, 6)]
if __name__ == '__main__':
    for n, k in NK:
        print(f'Running A2C with {k} workers and {n} steps.')
        suffix = f'_n{n}_k{k}' + ('_stoch' if STOCHASTIC else '') + ('_entropy' if ENTROPY_CORRECTION else '') + '.png'
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

    #writer.close()
    