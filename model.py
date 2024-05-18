import torch
import numpy as np
from torch import nn, optim

ACTOR_LR = 1e-5
CRITIC_LR = 1e-3
GAMMA = 0.99
LAM = 0.95
ENT_COEF = 0.01
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Code inspired from Gymnasium tutorial https://gymnasium.farama.org/tutorials/gymnasium_basics/vector_envs_tutorial/
class A2C(nn.Module):
    def __init__(self, 
        input_size, 
        output_size, 
        k,
        hidden_size=64, 
        activation=nn.Tanh,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        gamma=GAMMA,
        ent_coef=ENT_COEF
    ) -> None:
        super(A2C, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.k = k
        self.hidden_size = hidden_size
        self.activation = activation

        critic_layers = [
            nn.Linear(input_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, 1),
        ]

        actor_layers = [
            nn.Linear(input_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(
                hidden_size, output_size
            ),  # estimate action logits (will be fed into a softmax later)
        ]

        # define actor and critic networks
        self.critic = nn.Sequential(*critic_layers).to(DEVICE)
        self.actor = nn.Sequential(*actor_layers).to(DEVICE)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.ent_coef = ent_coef

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.Tensor(x).to(DEVICE)
        state_values = self.critic(x)
        action_logits = self.actor(x)
        return state_values, action_logits

    def select_action(
        self, x: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_values, action_logits = self.forward(x)
        action_pd = torch.distributions.Categorical(logits=action_logits)
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)

        return actions, action_log_probs, state_values, action_pd.entropy()
    
    def select_best_action(self, x: np.ndarray) -> int:
        _, action_logits = self.forward(x)
        return torch.argmax(action_logits, dim=1).detach().item()
    
    def get_value(self, x: np.ndarray) -> torch.Tensor:
        x = torch.Tensor(x).to(DEVICE)
        return self.critic(x)

    def get_losses(
        self,
        rewards: torch.Tensor,
        last_states: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(rewards)
        advantages = torch.zeros(n, self.k, device=DEVICE)
        discount_rewards = torch.zeros(self.k, device=DEVICE)

        for t in range(n):
            mask = masks[t-1] if t > 1 else torch.ones(self.k, device=DEVICE)
            discount_rewards += mask * self.gamma ** t * rewards[t]
        last_value_preds = self.get_value(last_states).squeeze()
        max_t = torch.argmax(masks, dim=0) + 1
        discount_rewards += masks[-1] * self.gamma ** max_t * last_value_preds

        advantages = discount_rewards - value_preds

        critic_loss = advantages.pow(2).mean()
        actor_loss = -(advantages.detach() * action_log_probs).mean() - self.ent_coef * entropy.mean()

        return critic_loss, actor_loss

    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()