from torch import nn


class SimpleRNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 sequence_length: int,
                 batch_size: int
                 ):
        """
        Simple RNN + FCN implementation
        Args:
            input_size: dimension of each element of the input sequence
            output_size: dimension of outputs of RNN layers
            sequence_length: length of the sequence
            batch_size: batch size that is given to the model
        """
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=output_size,
                          num_layers=1,
                          batch_first=True
                          )
        self.fc = nn.Linear(output_size * sequence_length, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.reshape(self.batch_size, 1, -1)
        x = self.fc(x)
        x = x.squeeze(-1)
        return x


class SimpleLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 sequence_length: int,
                 batch_size: int
                 ):
        """
        Simple LSTM + FCN implementation
        Args:
            input_size: dimension of each element of the input sequence
            output_size: dimension of outputs of RNN layers
            sequence_length: length of the sequence
            batch_size: batch size that is given to the model
        """
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=output_size,
                           num_layers=1,
                           batch_first=True
                           )
        self.fc = nn.Linear(output_size * sequence_length, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.reshape(self.batch_size, 1, -1)
        x = self.fc(x)
        x = x.squeeze(-1)
        return x
