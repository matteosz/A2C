import numpy as np
import torch
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from model import A2C, DEVICE
import matplotlib.pyplot as plt
import os
import copy

ENV_NAME = 'CartPole-v1'
MAX_STEPS = int(5e5)
EVAL_STEPS = int(2e4)
LOG_STEPS = int(1e3)
#MAX_STEPS = 5000
#EVAL_STEPS = 1000
#LOG_STEPS = 500
EVAL_EPISODES = 10

TAKE_TRUNCATION_INTO_ACCOUNT = True

gym.logger.set_level(40) # silence warnings
writer = SummaryWriter()

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

eval_info = [0,0]

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

    plt.title("Value function(seed"+str(eval_info[0])+", "+"eval nr."+str(eval_info[1])+")")
    plt.xlabel('Steps')
    plt.ylabel('Value function')

    path = 'plots/evaluations/plot_value_func_evaluation_seed'+str(eval_info[0])+"_eval_nr"+str(eval_info[1])+'.png'
    plt.savefig(path)
    plt.close()
    return

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
                    values.append(model.get_value(state).detach().item())

    env.close()
    mean_value = episode_rewards / EVAL_EPISODES
    print(f'Step: {step}, Average evaluation reward: {mean_value:.2f}')
    writer.add_scalar('Evaluation/Mean_Value', mean_value, step)

    logged_eval_value_func_curr_seed.append(np.mean(values))
    logged_episodic_returns_eval_curr_seed.append(mean_value)

    plot_value_function(list(range(len(values))), values)

    for i, values in enumerate(values):
        writer.add_scalar('Evaluation/Value_Function', values, i)

    eval_info[1] = eval_info[1] + 1
    return

def a2c(k=1, n=1, seed=42):
    print(f'Running A2C with {k} workers and {n} steps.')

    n_updates = MAX_STEPS // n
    critic_losses, actor_losses, entropies = [], [], []
    total_steps = 0

    envs = gym.vector.make(ENV_NAME, num_envs=k)
    num_actions, num_states = envs.single_action_space.n, envs.single_observation_space.shape[0]
    model = A2C(num_states, num_actions, k, n)
    envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=k*n_updates)

    states, _ = envs_wrapper.reset(seed=seed)

    are_logged_x_vals_eval_collected = False
    if len(logged_x_vals_eval) >0:
        are_logged_x_vals_eval_collected = True

    are_logged_x_vals_train_collected = False
    if len(logged_x_vals_train) >0:
        are_logged_x_vals_train_collected = True

    for _ in tqdm(range(1, n_updates+1)):
        ep_value_preds = torch.zeros(n, k, device=DEVICE)
        ep_rewards = torch.zeros(n, k, device=DEVICE)
        ep_action_log_probs = torch.zeros(n, k, device=DEVICE)
        masks_end_of_ep = torch.zeros(n, k, device=DEVICE)
        masks_bootstrap_values = torch.zeros(n, k, device=DEVICE)

        for i in range(n):
            masks_end_of_ep[i] = torch.tensor([i+1 for j in range(k)], device=DEVICE)

        # Collect data
        for step in range(n):
            actions, action_log_probs, state_value_preds, entropy = model.select_action(states)
            states, rewards, terminated, truncated, _ = envs_wrapper.step(actions.cpu().numpy())

            ep_value_preds[step] = state_value_preds.squeeze()
            ep_rewards[step] = torch.tensor(rewards, device=DEVICE, requires_grad=True)
            ep_action_log_probs[step] = action_log_probs
            
            if TAKE_TRUNCATION_INTO_ACCOUNT:
                for (i, (term, trunc)) in enumerate(zip(terminated, truncated)):
                    if term or trunc:
                        masks_end_of_ep[step][i] = step
                    if trunc:
                        estimated_value = model.get_value(states[i])
                        masks_bootstrap_values[step][i]  = estimated_value

            #gym vectorized environments automatically reset environments when envs are in done state
            #cf p.24 https://gym-docs.readthedocs.io/_/downloads/en/feature-branch/pdf/
            #testing showed using wrapper wrappers.RecordEpisodeStatistics does not effect this behaviour
            #therefore, no need to manually reset environments in terminated or truncated states

        for i in range(k):
            masks_end_of_ep[n-1][i] = n-1
        
        estimated_values = model.get_value(states).squeeze()
        masks_bootstrap_values[n-1]  = estimated_values

        discount_rewards = model.get_discounted_r(ep_rewards, masks_end_of_ep, masks_bootstrap_values)
        critic_loss, actor_loss = model.get_losses(discount_rewards, ep_action_log_probs, ep_value_preds, entropy)
        model.update_parameters(critic_loss, actor_loss)

        critic_losses.append(critic_loss.item())
        actor_losses.append(actor_loss.item())
        entropies.append(entropy.mean().item())

        total_steps += n

        # Logging and evaluation
        if total_steps % LOG_STEPS == 0:
            print(f'Step: {total_steps}, Critic Loss: {np.mean(critic_losses)}, Actor Loss: {np.mean(actor_losses)}, Entropy: {np.mean(entropies)}')
            writer.add_scalar('Train/Loss/Critic', np.mean(critic_losses), total_steps)
            writer.add_scalar('Train/Loss/Actor', np.mean(actor_losses), total_steps)
            writer.add_scalar('Train/Loss/Entropy', np.mean(entropies), total_steps)

            avg_reward = 0
            if len(np.array(envs_wrapper.return_queue)) <k:
                avg_reward = np.array(envs_wrapper.return_queue).mean()
            else:
                avg_reward = np.array(envs_wrapper.return_queue)[-k:].mean()
                
            print(f'Step: {total_steps}, Average reward: {avg_reward}')
            writer.add_scalar('Train/Average_Reward', avg_reward, total_steps)

            logged_actor_losses_curr_seed.append(actor_loss.item())
            logged_critic_losses_curr_seed.append(critic_loss.item())
            logged_entropies_curr_seed.append(entropy.mean().item())
            logged_episodic_returns_train_curr_seed.append(avg_reward)

            if are_logged_x_vals_train_collected == 0:
                logged_x_vals_train.append(total_steps)
            

        if total_steps % EVAL_STEPS == 0:
            evaluate(model, total_steps)
            if are_logged_x_vals_eval_collected == 0:
                logged_x_vals_eval.append(total_steps)

    envs_wrapper.close()

    logged_actor_losses_all_seeds.append(copy.deepcopy(logged_actor_losses_curr_seed))
    logged_critic_losses_all_seeds.append(copy.deepcopy(logged_critic_losses_curr_seed))
    logged_entropies_all_seeds.append(copy.deepcopy(logged_entropies_curr_seed))
    logged_episodic_returns_train_all_seeds.append(copy.deepcopy(logged_episodic_returns_train_curr_seed))
    logged_episodic_returns_eval_all_seeds.append(copy.deepcopy(logged_episodic_returns_eval_curr_seed))
    logged_eval_value_func_all_seeds.append(copy.deepcopy(logged_eval_value_func_curr_seed))

    logged_actor_losses_curr_seed.clear()
    logged_critic_losses_curr_seed.clear()
    logged_entropies_curr_seed.clear()
    logged_episodic_returns_train_curr_seed.clear()
    logged_episodic_returns_eval_curr_seed.clear()
    logged_eval_value_func_curr_seed.clear()

    eval_info[1] = 0
    return 


def create_one_plot(x_vals, data_seed_1, data_seed_2, data_seed_3, title, label, ylabel, path):
    plt.figure(figsize=(10, 6))

    min_values = [min(a, b, c) for a, b, c in zip(data_seed_1, data_seed_2, data_seed_3)]
    max_values = [max(a, b, c) for a, b, c in zip(data_seed_1, data_seed_2, data_seed_3)]
    avg_values = [(a + b + c) / 3 for a, b, c in zip(data_seed_1, data_seed_2, data_seed_3)]

    plt.plot(x_vals, data_seed_1, label=label+' (seed 2)', marker='o')
    plt.plot(x_vals, data_seed_2, label=label+' (seed 42)', marker='s')
    plt.plot(x_vals, data_seed_3, label=label+' (seed 242)', marker='^')
    plt.plot(x_vals, avg_values, label=label+' (average)', marker='x')

    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel(ylabel)

    plt.legend()

    plt.fill_between(x_vals, avg_values, min_values, color='gray', alpha = 0.3)
    plt.fill_between(x_vals, avg_values, max_values, color='gray', alpha = 0.3)

    plt.savefig(path)
    plt.close()
    return


def create_plots():
    if not os.path.exists('plots'):
        os.makedirs('plots')

    create_one_plot(logged_x_vals_train, logged_actor_losses_all_seeds[0], logged_actor_losses_all_seeds[1], logged_actor_losses_all_seeds[2], 'Actor loss throughout training', "Actor loss", "Loss", 'plots/plot_actor_loss.png')
    create_one_plot(logged_x_vals_train, logged_critic_losses_all_seeds[0], logged_critic_losses_all_seeds[1], logged_critic_losses_all_seeds[2], 'Critic loss throughout training', "Critic loss", "Loss", 'plots/plot_critic_loss.png')
    create_one_plot(logged_x_vals_train, logged_entropies_all_seeds[0], logged_entropies_all_seeds[1], logged_entropies_all_seeds[2], 'Entropy throughout training', "Entropy", "Entropy", 'plots/plot_entropy.png')

    create_one_plot(logged_x_vals_train, logged_episodic_returns_train_all_seeds[0], logged_episodic_returns_train_all_seeds[1], logged_episodic_returns_train_all_seeds[2], 'Average undiscounted trajectory return throughout training', "Return", "Return", 'plots/plot_return_training.png')
    create_one_plot(logged_x_vals_eval, logged_episodic_returns_eval_all_seeds[0], logged_episodic_returns_eval_all_seeds[1], logged_episodic_returns_eval_all_seeds[2], 'Average undiscounted trajectory return throughout evaluation', "Return", "Return", 'plots/plot_return_evaluation.png')
    
    create_one_plot(logged_x_vals_eval, logged_eval_value_func_all_seeds[0], logged_eval_value_func_all_seeds[1], logged_eval_value_func_all_seeds[2], 'Evolution of the value function across evaluations', "Mean value of value function", "Mean value of value function", 'plots/plot_value_func_evaluation.png')

    return 


if __name__ == '__main__':
    for seed in [2, 42, 242]:
        eval_info[0] = seed
        set_seed(seed)
        a2c(k=1, n= 1, seed=seed)
        #break # Remove break after testing
    writer.close()
    create_plots()
