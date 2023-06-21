import torch


class TensorQueue():
    def __init__(self):
        self.data = torch.zeros((0,))

    def put(self, data):
        if self.data.shape == (0,):
            self.data = data
        else:
            self.data = torch.cat((self.data, data), dim=0)

    def get(self):
        return self.data
