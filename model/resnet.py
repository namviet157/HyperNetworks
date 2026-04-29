from model.nets.resnet_v2 import build_resnet_v2_50, build_wrn_40_2


class Resnet50(object):
    def __init__(self, num_classes, hyper_mode=True):
        self.num_classes = num_classes
        self.hyper_mode = hyper_mode

    def build_model(self):
        return build_resnet_v2_50(num_classes=self.num_classes, hyper_mode=self.hyper_mode)


class WideResnet40_2(object):
    def __init__(self, num_classes, hyper_mode=True):
        self.num_classes = num_classes
        self.hyper_mode = hyper_mode

    def build_model(self):
        return build_wrn_40_2(num_classes=self.num_classes, hyper_mode=self.hyper_mode)
