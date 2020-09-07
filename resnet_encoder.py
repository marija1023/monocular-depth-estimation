import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers=18, pretrained=True, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        if num_layers != 18:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
        
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnet = models.resnet18

        self.encoder = resnet(pretrained)


    def forward(self, input_image):
        self.features = []
        # normalization
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features