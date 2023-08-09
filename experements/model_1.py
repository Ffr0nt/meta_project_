import torch
import torch.nn as nn

class model_1(nn.Module):
    def __init__(self, bert_output_size = 5,
                 fc_hidden_size = 10,
                 out_class_number =30):
        super().__init__()

        self.bert_output_size = bert_output_size # embedding length for the longest text
        self.hidden_size = fc_hidden_size
        self.out_class_number = out_class_number # number of classes for prediction

        self.BERT_BLOCK = ...
        self.fc1 = nn.Linear(in_features=self.bert_output_size, out_features=self.hidden_size)
        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.out_class_number)

        self.RELU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

        self.prediction_list = []

    def forward(self, X):
        # x - list of texts no matter of modality

        for t in X["comment"]:
            x = t["text of comment"]
            x = self.BERT_BLOCK(x)
            x = self.fc1(x)
            x = self.RELU(x)
            x = self.fc2(x)
            x = self.Sigmoid(x)
            self.prediction_list.append(x)

        for t in X["review"]:
            x = t["text"]
            x = self.BERT_BLOCK(x)
            x = self.fc1(x)
            x = self.RELU(x)
            x = self.fc2(x)
            x = self.Sigmoid(x)
            self.prediction_list.append(x)

        answer = torch.mean(
            torch.stack(
                self.prediction_list,
                dim=0)
            , 0)

        return answer
