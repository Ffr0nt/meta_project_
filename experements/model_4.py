import torch
import torch.nn as nn
from random import shuffle


class model_4(nn.Module):
    def __init__(self,
                 bert_output_size_1=5,
                 bert_output_size_2=5,
                 lstm_hidden_size_1 =10,
                 lstm_hidden_size_2 = 10,
                 fc_hidden_size=10,
                 out_class_number=30):
        super().__init__()

        self.bert_output_size_1 = bert_output_size_1 # embedding length for the longest text
        self.bert_output_size_2 = bert_output_size_2 # embedding length for the longest text
        self.hidden_size = fc_hidden_size
        self.lstm_hidden_size_1 = lstm_hidden_size_1 #lstm_hidden_size
        self.lstm_hidden_size_2 = lstm_hidden_size_2 #lstm_hidden_size
        self.out_class_number = out_class_number # number of classes for prediction

        self.BERT_BLOCK_1 = ...
        self.BERT_BLOCK_2 = ...
        self.LSTM_BLOCK_1 = nn.LSTM(input_size=self.bert_output_size,
                                    hidden_size=self.lstm_hidden_size_1)
        self.LSTM_BLOCK_2 = nn.LSTM(input_size=self.bert_output_size,
                                    hidden_size=self.lstm_hidden_size_2)

        self.fc1 = nn.Linear(in_features= 2*self.lstm_hidden_size_1+ 2*self.lstm_hidden_size_2
                             , out_features = self.hidden_size)
        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features = self.out_class_number)

        self.RELU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

        self.text_embeddings_1 = []
        self.text_embeddings_2 = []

    def forward(self, X):

        for t in X["comment"]:
            x = t["text of comment"]
            x = self.BERT_BLOCK_1(x)
            self.text_embeddings_1.append(x)

        for t in X["review"]:
            x = t["text"]
            x = self.BERT_BLOCK_2(x)
            self.text_embeddings_2.append(x)

        shuffle(self.text_embeddings_1) # in order to shuffle embeddings from reviews and comments
        memory_units_1 = self.LSTM_BLOCK_1( self.text_embeddings )[1] #hidden state & cell state

        shuffle(self.text_embeddings_2) # in order to shuffle embeddings from reviews and comments
        memory_units_2 = self.LSTM_BLOCK_2( self.text_embeddings )[1] #hidden state & cell state

        x = torch.cat((memory_units_1[0][0], memory_units_1[1][0],
                       memory_units_2[0][0], memory_units_2[1][0])) #concatinate memory of both modalities
        x = self.fc1(x)
        x = self.RELU(x)
        x = self.fc2(x)
        x = self.Sigmoid(x)

        return x

