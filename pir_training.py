import InfluxDB_connection as db
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, PIR_values, labels):
        'Initialization'
        self.labels = labels
        self.PIR_values = PIR_values

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.PIR_values)

    def __getitem__(self, index):
        # Load data and get label
        X = self.PIR_values[index]
        y = self.labels[index]

        return X, y


class LSTM1_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super().__init__()

        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, 1) #fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

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
            if previous_walking_speed is None and data['walking_speed'] is not None:
                previous_walking_speed = data['walking_speed']

            if data['walking_speed'] is not None:
                current_walking_speed = data['walking_speed']
            else:
                current_walking_speed = previous_walking_speed

            # if current_walking_speed is not None:
            previous_walking_speed = current_walking_speed
            if current_walking_speed is None or data['pir0'] is None:
                continue
            X.append([data['pir0'], data['pir1'], data['pir2'], data['pir3']])
            Y.append(current_walking_speed)

    latency = int(pir_frequency * time_for_lstm)
    X = pd.DataFrame(X[:-latency])
    # for i in range(len(X)):
    #     for offset in range(latency):
    #         aggregated_Xs.append(torch.tensor(X[i + latency]))
    # aggregated_Xs = aggregated_Xs
    Y = pd.DataFrame(Y[latency:])
    return X, Y


if __name__ == '__main__':
    mm = MinMaxScaler()
    ss = StandardScaler()

    X, Y = prepare_pir_data(10.0)
    X = mm.fit_transform(X)
    Y = ss.fit_transform(Y)

    train_size, test_size = len(X)//2, len(X) - len(X)//2

    X_train, y_train = X[train_size:], Y[train_size:]
    X_test, y_test = X[:test_size], Y[:test_size]

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Parameters
    params = {'batch_size': 128,
              'shuffle': False,
              'num_workers': 6}
    num_epochs = 10
    learning_rate = 0.001
    input_size = 4 # number of features
    hidden_size = 2 # number of features in hidden state
    num_layers = 1 # number of stacked lstm layers
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

    training_set = Dataset(X_train_tensors_final, y_train_tensors)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = Dataset(X_test_tensors_final, y_test_tensors)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    for epoch in range(num_epochs):
        current_loss = 0
        for local_batch, local_labels in training_generator:
            outputs = lstm.forward(local_batch)  # forward pass
            optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

            # obtain the loss function
            loss = criterion(outputs, local_labels)
            current_loss += loss.item()

            loss.backward()  # calculates the loss of the loss function

            optimizer.step()  # improve from loss, i.e backprop
            # print(np.mean(abs(outputs.detach().numpy()) - abs(local_labels.detach().numpy())))
        if epoch % 1 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, current_loss))

"""
    # Looking at prediction:
    df_X_ss = X_train_tensors_final  # ss.transform(X_test)  # old transformers
    df_y_mm = y_train_tensors  #  mm.transform(y_test)  # old transformers

    # df_X_ss = Variable(torch.Tensor(df_X_ss))  # converting to Tensors
    # df_y_mm = Variable(torch.Tensor(df_y_mm))
    # reshaping the dataset
    # df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))

    train_predict = lstm(df_X_ss)  # forward pass
    data_predict = train_predict.data.numpy()  # numpy conversion
    dataY_plot = df_y_mm.data.numpy()

    # data_predict = ss.inverse_transform(data_predict)  # reverse transformation
    # dataY_plot = ss.inverse_transform(dataY_plot)
    plt.figure(figsize=(10, 6))  # plotting
    plt.axvline(x=train_size, c='r', linestyle='--')  # size of the training set

    plt.plot(dataY_plot, label='Actual Data')  # actual plot
    plt.plot(data_predict, label='Predicted Data')  # predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show()
"""
