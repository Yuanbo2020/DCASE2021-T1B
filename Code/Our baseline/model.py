import torch
import torch.nn as nn
import torchvision.models as models

class VideoNet(nn.Module):

    def __init__(self,dim=256):
        """
        dim: feature dimension (default: 256)
        """
        super(VideoNet, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        numFit = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(numFit, dim)
        self.classifer = nn.Linear(dim, 10)

    def forward(self,x):

        embedding_feature = self.resnet(x)
        embedding_feature = torch.relu(embedding_feature)
        x = self.classifer(embedding_feature)
        return x,embedding_feature
