import torch
import torch.nn as nn
from random import shuffle


class model_3(nn.Module):
    def __init__(self, bert_output_size=5,
                 fc_hidden_size=10,
                 lstm_hidden_size=10,
                 out_class_number=30):
        super().__init__()

        self.bert_output_size = bert_output_size # embedding length for the longest text
        self.hidden_size = fc_hidden_size
        self.lstm_hidden_size = lstm_hidden_size #lstm_hidden_size
        self.out_class_number = out_class_number # number of classes for prediction

        self.BERT_BLOCK = ...
        self.LSTM_BLOCK = nn.LSTM(input_size=self.bert_output_size,
                                  hidden_size=self.lstm_hidden_size)
        self.fc1 = nn.Linear(in_features= 2*self.lstm_hidden_size, out_features = self.hidden_size)
        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features = self.out_class_number)

        self.RELU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

        self.text_embeddings = []

    def forward(self, X):

        for t in X["comment"]:
            x = t["text of comment"]
            x = self.BERT_BLOCK(x)
            self.text_embeddings.append(x)

        for t in X["review"]:
            x = t["text"]
            x = self.BERT_BLOCK(x)
            self.text_embeddings.append(x)

        # shuffle(self.text_embeddings) # in order to shuffle embeddings from reviews and comments

        memory_units = self.LSTM_BLOCK( self.text_embeddings )[1] #hidden state & cell state
        x = torch.cat((memory_units[0][0], memory_units[1][0]))
        x = self.fc1(x)
        x = self.RELU(x)
        x = self.fc2(x)
        x = self.Sigmoid(x)

        return x

