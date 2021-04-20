import InfluxDB_connection as db
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class LSTM1_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super().__init__()

        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        return self.lstm(x, (h_0, c_0))

def prepare_pir_data(time_for_lstm = 1.0):
    pir_frequency = 10
    client = db.ListicClientDB(server_side=True)
    all_data_queries = client.get_all_pir_values()

    X = []
    aggregated_Xs = []
    Y = []

    previous_walking_speed = None
    for all_data in all_data_queries:
        for data in all_data:
            if previous_walking_speed is None and 'walking_speed' in data and data['walking_speed'] is not None:
                previous_walking_speed = data['walking_speed']
            if 'walking_speed' in data and data['walking_speed'] is not None:
                current_walking_speed = data['walking_speed']
            else:
                current_walking_speed = previous_walking_speed

            previous_walking_speed = current_walking_speed
            if current_walking_speed is None or data['pir0'] is None:
                continue
            X.append([data['pir0'], data['pir1'], data['pir2'], data['pir3']])
            print(data, X[-1])
            Y.append(data['walking_speed'])

    latency = int(pir_frequency * time_for_lstm)
    print(X)
    X = X[:-latency]
    for i in range(len(X)):
        for offset in range(latency):
            aggregated_Xs.append(torch.tensor(X[i + latency]))
    # aggregated_Xs = aggregated_Xs
    Y = Y[latency:]
    return X, Y


if __name__ == '__main__':
    mm = MinMaxScaler()
    ss = StandardScaler()

    X, Y = prepare_pir_data(1.0)
    print(X)
    X = mm.fit_transform(X)
    Y = ss.fit_transform(Y)

    X_train, y_train = X, Y
    X_test, y_test = X, Y

    num_epochs = 100
    learning_rate = 0.0001
    input_size = 5 #number of features
    hidden_size = 2 #number of features in hidden state
    num_layers = 1 #number of stacked lstm layers
    seq_length = 10

    lstm = LSTM1_model(input_size, hidden_size, num_layers, seq_length)
    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    print("Training Shape", X_train.shape, y_train.shape)
    print("Testing Shape", X_test.shape, y_test.shape)

    X_train_tensors = Variable(torch.Tensor(X_train))
    X_test_tensors = Variable(torch.Tensor(X_test))

    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))

    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

    print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
    print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)

    for epoch in range(num_epochs):
        outputs = lstm.forward(X_train_tensors_final)  # forward pass
        optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

        # obtain the loss function
        loss = criterion(outputs, y_train_tensors)

        loss.backward()  # calculates the loss of the loss function

        optimizer.step()  # improve from loss, i.e backprop
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
