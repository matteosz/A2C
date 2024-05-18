import numpy as np
import torch
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from model import A2C, DEVICE

ENV_NAME = 'CartPole-v1'
MAX_STEPS = int(5e5)
EVAL_STEPS = int(2e4)
LOG_STEPS = int(1e3)
EVAL_EPISODES = 10

gym.logger.set_level(40) # silence warnings
writer = SummaryWriter()

def set_seed(agent):
    torch.manual_seed(agent)
    np.random.seed(agent)
    random.seed(agent)

def evaluate(model, step, render=True):
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
    for i, values in enumerate(values):
        writer.add_scalar('Evaluation/Value_Function', values, i)

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
    for _ in tqdm(range(1, n_updates+1)):
        ep_value_preds = torch.zeros(n, k, device=DEVICE)
        ep_rewards = torch.zeros(n, k, device=DEVICE)
        ep_action_log_probs = torch.zeros(n, k, device=DEVICE)
        masks = torch.zeros(n, k, device=DEVICE)

        # Collect data
        for step in range(n):
            actions, action_log_probs, state_value_preds, entropy = model.select_action(states)
            states, rewards, terminated, truncated, _ = envs_wrapper.step(actions.cpu().numpy())

            ep_value_preds[step] = state_value_preds.squeeze()
            ep_rewards[step] = torch.tensor(rewards, device=DEVICE, requires_grad=True)
            ep_action_log_probs[step] = action_log_probs
            # Mask to deal with finished trajectories (both terminated and truncated)
            masks[step] = torch.tensor([not (term or trunc) for term, trunc in zip(terminated, truncated)], device=DEVICE)
            
        discount_rewards = model.get_discounted_r(ep_rewards, masks, states)
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

            avg_reward = np.array(envs_wrapper.return_queue).mean()
            print(f'Step: {total_steps}, Average reward: {avg_reward}')
            writer.add_scalar('Train/Average_Reward', avg_reward, total_steps)

        if total_steps % EVAL_STEPS == 0:
            evaluate(model, total_steps)

    envs_wrapper.close()

if __name__ == '__main__':
    for seed in [2, 42, 242]:
        set_seed(seed)
        a2c(seed=seed)
        break # Remove break after testing
    writer.close()