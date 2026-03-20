from model.utils import HyperConv2D
from model.simple_cnn import SimpleCNN

try:
    from model.resnet import Resnet50
except ModuleNotFoundError:
    Resnet50 = None
