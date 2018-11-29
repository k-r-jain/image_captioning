import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        return features.unsqueeze_(1) #Adding channel first

class DecoderCNN(nn.Module):
    def __init__(self, growth_rate, num_layers, target_size, feature_size = 2048, max_sequence_length = 40):
        super(DecoderCNN, self).__init__()
        self.num_layers= num_layers
        self.conv = []
        self.linear = nn.Linear(growth_rate * feature_size, max_sequence_length)
        self.bn = nn.BatchNorm1d(growth_rate * feature_size, momentum=0.01)
        for i in range(self.num_layers):
            in_channels = growth_rate * i + 1
            self.conv.append(nn.Conv1d(in_channels, growth_rate, 3, padding = 1))

    def forward(self, features):
        activations = [features]
        for i in range(self.num_layers):
            # print('Iteration', i)
            new_input = torch.cat([x for x in activations[: (i + 1)]], 1)
            if i == (self.num_layers - 1):
                activations.append(F.softmax(self.conv[i](new_input)))
            else:
                activations.append(F.relu(self.conv[i](new_input)))
            # print('Input, Activation:', new_input.size(), activations[i + 1].size())
        activations = activations[-1].view(-1, activations[-1].shape[1] * activations[-1].shape[2])
        activations = self.linear(self.bn(activations))
        # print('Final:', activations.size())
        return activations
