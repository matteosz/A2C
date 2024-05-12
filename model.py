import json
from actor import Actor
from critic import Critic
import torch
from torch import nn, optim

def load_hyperparameters(filename='hyperparams.json'):
    with open(filename, 'r') as f:
        return json.load(f)
    
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class A2C():
    def __init__(self, input_size, output_size, hidden_size=64, activation=nn.Tanh):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.activation = activation

        self.actor = Actor(input_size, output_size, hidden_size, activation)
        self.critic = Critic(input_size, 1, hidden_size, activation)

        self.actor.share_memory()
        self.critic.share_memory()
        self.actor.to(DEVICE)
        self.critic.to(DEVICE)

        hyperparameters = load_hyperparameters()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=hyperparameters['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hyperparameters['critic_lr'])
        self.gamma = hyperparameters['gamma']

    def copy(self):
        new_model = A2C(self.input_size, self.output_size, self.hidden_size, self.activation)
        new_model.actor.load_state_dict(self.actor.state_dict())
        new_model.critic.load_state_dict(self.critic.state_dict())
        return new_model

    def __get_action_probs(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
        return self.actor(state)

    def sample_action(self, state):
        return torch.multinomial(self.__get_action_probs(state), 1).detach().item()
    
    def get_best_action(self, state):
        return torch.argmax(self.__get_action_probs(state)).detach().item()
    
    def get_reward(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
        return self.critic(state)

    def update_step(self, state, action, discount_reward):
        advantage = discount_reward - self.get_reward(state)

        actor_loss = -torch.log(self.__get_action_probs(state)[action]) * advantage
        critic_loss = advantage ** 2

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        actor_loss.backward(retain_graph=True)
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        return actor_loss.detach().item(), critic_loss.detach().item()