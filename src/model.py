from torch import nn
from jcopdl.layers import conv_block, linear_block
from torchvision.models import resnet50


# class CNN(nn.Module):
#     def __init__(self, in_channel, conv1, conv2, conv3, conv4, kernel, pad, out_channel, in_size, n1, n2, dropout, out_size):
#         super().__init__()

#         self.convolutional = nn.Sequential(

#             conv_block(in_channel, conv1, kernel=kernel, pad=pad),
#             conv_block(conv1, conv2, kernel=kernel, pad=pad),
#             conv_block(conv2, conv3, kernel=kernel, pad=pad),
#             conv_block(conv3, conv4, kernel=kernel, pad=pad),
#             conv_block(conv4, out_channel, kernel=kernel, pad=pad),
#             nn.Flatten()

#         )

#         self.fc = nn.Sequential(

#             linear_block(in_size, n1, dropout=dropout),
#             linear_block(n1, n2, dropout=dropout),
#             linear_block(n2, out_size, activation='lsoftmax')
#         )

#     def forward(self, x):
#         x = self.convolutional(x)
#         x = self.fc(x)
#         return x

# MODEL RESNET50 - TRANSFER LEARNING
class CustomResnet50(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        # download weight dari resnet50
        self.resnet = resnet50(pretrained=True)

        self.freeze()

        # ubah dan reset fully connectednya saja (fc)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.resnet(x)

    def freeze(self):
        # frezee weight dan arsitekturnya
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreeze(self):
        # unfreeze weight dan arsitekturnya
        for name, child in self.resnet.named_children():
            if name in ['layer3', 'layer4']:
                print(name + ' unfreeze')
                for param in child.parameters():
                    param.requires_grad = True
