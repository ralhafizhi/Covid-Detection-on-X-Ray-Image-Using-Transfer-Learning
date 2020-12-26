import io
import torch
from torch import nn
from PIL import Image
# from src.model import CNN
from src.model import CustomResnet50
import torchvision.transforms as transforms
import config as cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# #load model CNN biasa
# def get_model():
#     checkpoint_path = torch.load(
#         'model/weights_bestCNN.pth', map_location="cpu")
#     model = CNN(cfg.in_channel, cfg.conv1, cfg.conv2, cfg.conv3, cfg.conv4, cfg.kernel,
#                 cfg.pad, cfg.out_channel, cfg.in_size, cfg.n1, cfg.n2, cfg.dropout, cfg.out_size)
#     model.load_state_dict(checkpoint_path)
#     with torch.no_grad():
#         model.eval()
#         return model

# #data augmentation CNN biasa

# def get_tensor(image_location):
#     my_transforms = transforms.Compose([transforms.Grayscale(),
#                                         transforms.Resize(240),
#                                         transforms.CenterCrop(cfg.crop_size),
#                                         transforms.ToTensor()])
#     image = Image.open(image_location)
#     return my_transforms(image).unsqueeze(0)


# load model resnet50 - Transfer Learning
def get_model():
    checkpoint_path = torch.load('model/weights_best.pth', map_location="cpu")
    model = CustomResnet50(cfg.output_size).to(device)
    model.load_state_dict(checkpoint_path)
    with torch.no_grad():
        model.eval()
        return model

# data augmentation model resnet50 - Transfer Learning
def get_tensor(image_location):
    my_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                        transforms.Resize(240),
                                        transforms.CenterCrop(cfg.crop_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = Image.open(image_location)
    return my_transforms(image).unsqueeze(0)
