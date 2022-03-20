import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet, BasicBlock

class ResNet18_conv(ResNet):
    def __init__(self):
        super(ResNet18_conv, self).__init__(BasicBlock, [2, 2, 2, 2])
        
    def forward(self, x):
        # change forward here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def get_image_extractor(arch = 'resnet18', pretrained = True, feature_dim = None, checkpoint = ''):
    '''
    Inputs
        arch: Base architecture
        pretrained: Bool, Imagenet weights
        feature_dim: Int, output feature dimension
        checkpoint: String, not implemented
    Returns
        Pytorch model
    '''

    if arch == 'resnet18':
        model = models.resnet18(pretrained = pretrained)
        if feature_dim is None:
            model.fc = nn.Sequential()
        else:
            model.fc = nn.Linear(512, feature_dim) 

    if arch == 'resnet18_conv':
        model = ResNet18_conv()
        model.load_state_dict(models.resnet18(pretrained=True).state_dict())

    elif arch == 'resnet50':
        model = models.resnet50(pretrained = pretrained)
        if feature_dim is None:
            model.fc = nn.Sequential()
        else:
            model.fc = nn.Linear(2048, feature_dim) 

    elif arch == 'resnet50_cutmix':
        model = models.resnet50(pretrained = pretrained)
        checkpoint = torch.load('/home/ubuntu/workspace/pretrained/resnet50_cutmix.tar')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if feature_dim is None:
            model.fc = nn.Sequential()
        else:
            model.fc = nn.Linear(2048, feature_dim) 

    elif arch == 'resnet152':
        model = models.resnet152(pretrained = pretrained)
        if feature_dim is None:
            model.fc = nn.Sequential()
        else:
            model.fc = nn.Linear(2048, feature_dim) 

    elif arch == 'vgg16':
        model = models.vgg16(pretrained = pretrained)
        modules = list(model.classifier.children())[:-3]
        model.classifier=torch.nn.Sequential(*modules)
        if feature_dim is not None:
            model.classifier[3]=torch.nn.Linear(4096,feature_dim)

    return model

