from torch import nn


class SimpleRNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 sequence_length: int,
                 batch_size: int
                 ):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=output_size,
                          num_layers=3,
                          batch_first=True
                          )
        self.fc = nn.Linear(output_size*sequence_length, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.reshape(self.batch_size, 1, -1)
        x = self.fc(x)
        x = x.squeeze(-1)
        return x
