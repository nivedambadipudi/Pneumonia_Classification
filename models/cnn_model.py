'''
CNN model for Pneumonia classification
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Pneumonia_Classification_Net(nn.Module):
    def __init__(self, activation, conv_layer, dense_layer, layer_size):
        self.activation = activation
        self.conv_layer = conv_layer
        self.dense_layer = dense_layer
        self.layer_size = layer_size
        return

    def build_model(self):
        return
