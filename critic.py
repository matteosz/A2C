from torch import nn

'''
Feedforward neural network for the critic model.

There are 2 hidden layers with the same number of neurons
and a final output layer with a linear activation function.
'''
class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, activation):
        super(Critic, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.nn(x)