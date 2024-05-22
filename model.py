import torch
import numpy as np
from torch import nn, optim
from typing import Tuple

# Default hyperparameters
ACTOR_LR = 1e-5
CRITIC_LR = 1e-3
GAMMA = 0.99

'''
Actor network - policy function approximator (π)
'''
class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, activation) -> None:
        super(Actor, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, output_size),
        )

    '''
    Forward pass of the actor network
    Args:
        x (torch.Tensor): Input tensor (S_t)
    Returns:
        torch.Tensor: Output tensor (π(S_t))
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)
    

'''
Critic network - value function approximator (V)
'''
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, activation) -> None:
        super(Critic, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, 1),
        )

    '''
    Forward pass of the critic network
    Args:
        x (torch.Tensor): Input tensor (S_t)
    Returns:
        torch.Tensor: Output tensor (V(S_t))
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)

'''
Advantage Actor-Critic (A2C) model.

This model is used to train the agent using the A2C algorithm. 
It contains both the actor and critic networks, as well as the optimizer for each network.
'''
class A2C(nn.Module):
    def __init__(self, 
        input_size, 
        output_size, 
        k,
        n,
        hidden_size=64, 
        activation=nn.Tanh,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        gamma=GAMMA,
    ) -> None:
        super(A2C, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.k = k
        self.n = n
        self.hidden_size = hidden_size
        self.activation = activation

        self.actor = Actor(input_size, output_size, hidden_size, activation)
        self.critic = Critic(input_size, hidden_size, activation)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma

    def forward(self, x: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.Tensor(x)
        state_values = self.critic(x)
        action_logits = self.actor(x)
        return state_values, action_logits

    '''
    Select an action given the current state
    Args:
        x (np.ndarray): Current state (S_t), size=(k, input_size)
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing for each k environment
            - the selected actions,
            - the log of the probabilities of the actions,
            - the state values,
            - the entropy.
    '''
    def select_action(
        self, x: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_values, action_logits = self.forward(x)
        # Create a categorical distribution to sample actions from
        action_pd = torch.distributions.Categorical(logits=action_logits)
        actions = action_pd.sample()
        # We return the log (as if we were doing softmax) of the probabilities to calculate the loss
        action_log_probs = action_pd.log_prob(actions)

        return actions, action_log_probs, state_values, action_pd.entropy()
    
    '''
    Select the best action given the current state.
    It's used for evaluation purposes.
    Args:
        x (np.ndarray): Current state (S_t), size=(input_size)
    Returns:
        int: The best action index
    '''
    def select_best_action(self, x: np.ndarray) -> int:
        _, action_logits = self.forward(x)
        return torch.argmax(action_logits).item()
    
    '''
    Forward pass of the critic network
    '''
    def get_value(self, x: np.ndarray) -> torch.Tensor:
        x = torch.Tensor(x)
        return self.critic(x)

    def get_discounted_r(
        self, 
        rewards: torch.Tensor,
        masks_end_of_ep: torch.Tensor,
        masks_bootstrap_values: torch.Tensor,
        mask_is_terminated: torch.Tensor
    ) -> torch.Tensor:
        discount_rewards = torch.zeros(self.n, self.k)
        acc = .0
        for t in reversed(range(self.n)):
            t_tens = torch.tensor([t] * self.k)
            acc = self.gamma * acc * (masks_end_of_ep[t] - t_tens)
            acc += self.gamma * masks_bootstrap_values[t] * (1 - mask_is_terminated[t])
            acc += rewards[t]
            discount_rewards[t] = acc
        return discount_rewards

    '''
    Calculate the losses for the actor and critic networks.

    Args:
        rewards (torch.Tensor): Rewards tensor, size=(n, k)
        last_states (torch.Tensor): Last states tensor, size=(k, input_size)
        action_log_probs (torch.Tensor): Log of the probabilities of the actions, size=(n, k)
        value_preds (torch.Tensor): Value predictions, size=(n, k)
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the critic and actor losses
    '''
    def get_losses(
        self,
        discount_rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = discount_rewards.detach() - value_preds

        critic_loss = advantages.pow(2).mean()
        actor_loss = -(advantages.detach() * action_log_probs).mean()

        return critic_loss, actor_loss

    '''
    Update the parameters of the actor and critic networks.
    Args:
        critic_loss (torch.Tensor): Critic loss
        actor_loss (torch.Tensor): Actor loss
    '''
    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
