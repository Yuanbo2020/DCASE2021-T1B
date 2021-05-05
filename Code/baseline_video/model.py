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

class Vgg_16(nn.Module):

    def __init__(self,dim=256):
        """
        dim: feature dimension (default: 256)
        """
        super(Vgg_16, self).__init__()

        self.vgg = models.vgg16(pretrained=True)
        numFit = self.vgg.classifier.in_features
        self.vgg.classifier = nn.Linear(numFit, dim)
        self.classifer = nn.Linear(dim, 10)

    def forward(self,x):

        embedding_feature = self.vgg(x)
        embedding_feature = torch.relu(embedding_feature)
        x = self.classifer(embedding_feature)
        return x,embedding_feature


class AlexNet(nn.Module):

    def __init__(self,dim=256):
        """
        dim: feature dimension (default: 256)
        """
        super(AlexNet, self).__init__()

        self.alex = models.alexnet(pretrained=True)
        numFit = self.alex.classifier.in_features
        self.alex.classifier = nn.Linear(numFit, dim)
        self.classifer = nn.Linear(dim, 10)

    def forward(self,x):

        embedding_feature = self.alex(x)
        embedding_feature = torch.relu(embedding_feature)
        x = self.classifer(embedding_feature)
        return x,embedding_feature