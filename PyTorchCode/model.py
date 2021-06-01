import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# from torch.nn.modules.activation import Softmax

class DeepConv1D(nn.Module):
    def __init__(self, n_filters=40, filter_size=11, n_dense=100, n_channels=4, window_size= 400, n_classes = 20, drop_prob=0.2):
        super(DeepConv1D, self).__init__()

        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_dense = n_dense
        self.n_channels=n_channels
        self.window_size=window_size
        self.n_classes=n_classes

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, n_filters, kernel_size=filter_size, padding=filter_size//2 ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_filters),
                    nn.Dropout(p=drop_prob)
        )
        self.conv2 = nn.Sequential(
                    nn.Conv1d(n_filters, n_filters, kernel_size=filter_size, padding=filter_size//2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(n_filters),
                    nn.Dropout(p=drop_prob)
        )
        
        self.maxpool = nn.MaxPool1d(10,10)
        self.dropout = nn.Dropout(p=drop_prob)

        flatten_size = int(n_filters*window_size/100)
        self.fc = nn.Sequential(
            nn.Linear(flatten_size,n_dense),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(n_dense,self.n_classes),
            # nn.Sigmoid()
        )
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        
        batch_size, C, L = out.size()
        
        flatten = out.view(-1, C*L)
        out = self.fc(flatten)


        return out

class DeepConvLSTM(nn.Module):
    def __init__(self, n_conv = 4, n_hidden = 128, n_layers = 2, n_filters = 64, n_classes = 10, filter_size = 5, drop_prob = 0.5, window_size = 30, n_channels = 4):
        
        super(DeepConvLSTM, self).__init__() # Call init function for nn.Module whenever this function is called
        
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.window_size = window_size
        self.n_channels = n_channels

        self.__dict__.update(locals())

        # Convolutional net
        self.convlayer = nn.ModuleList([nn.Conv1d(n_channels, n_filters, (filter_size), padding=filter_size//2)]) #First layer should map from number of channels to number of filters
        self.convlayer.extend([nn.Conv1d(n_filters, n_filters, (filter_size), padding=filter_size//2) for i in range(n_conv-1)]) # Subsequent layers should map n_filters -> n_filters

        self.maxpool = nn.MaxPool1d(10,10)
        self.batchnorm = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(n_filters),
                        nn.Dropout(p=drop_prob),
        )

		# LSTM layers
        if self.n_layers > 0:
            self.lstm = nn.LSTM(n_filters, n_hidden, n_layers, batch_first=True)
            self.predictor = nn.Sequential(
                            nn.Linear(n_hidden,n_classes)
                            )
        else:
            self.predictor = nn.Linear(n_filters,n_classes)

        self.dropout = nn.Dropout(p=drop_prob)
    
    def forward(self, x, hidden, batch_size):
        #Reshape x if necessary to add the 2nd dimension
        x = x.view(-1, self.n_channels, self.window_size)

        layer_cnt = 0
        for conv in self.convlayer:
            x = conv(x)
            x = self.batchnorm(x)
            # print('x:',layer_cnt, x.size())
            layer_cnt+=1
            if layer_cnt ==2:
                x = self.maxpool(x)
                # print('maxpool:',x.size())
        
        x = self.maxpool(x)
        x = x.view(batch_size, -1, self.n_filters)
        
        if self.n_layers > 0:
            x,hidden = self.lstm(x, hidden)

            x = self.dropout(x)

            x = x.view(batch_size, -1, self.n_hidden)[:,-1,:]
		

        out = self.predictor(x)

        return out, hidden

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data # return a Tensor from self.parameters to use as a base for the initial hidden state.
		
        hidden = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda(), # Hidden state
                weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda()) # Cell state
		
        return hidden
