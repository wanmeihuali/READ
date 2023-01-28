import torch
import torch.nn.functional as F
import torchvision

from READ.models.conv import PartialConv2d
class ResnetLoss(torch.nn.Module):
    """
    Resnet Loss: it actually performs worse than VGG loss. See https://arxiv.org/pdf/2104.05623.pdf
    """
    def __init__(self, partialconv=False, save_dir='.cache/torch/models'):
        super().__init__()
        self.partialconv = partialconv
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_layers = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.register_buffer('mean_', torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('std_', torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None])

        # Replace the first conv layer with partial conv for masked inputs
        if self.partialconv:
            part_conv = PartialConv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            part_conv.weight = resnet.conv1.weight
            part_conv.bias = resnet.conv1.bias
            self.resnet_layers[0] = part_conv
            self.layers = [3, 4, 5, 6, 7]

            # Freeze the weights of the resnet
        for param in self.resnet_layers.parameters():
            param.requires_grad = False

    def normalize_inputs(self, x):
        return (x - self.mean_) / self.std_

    def forward(self, im_out: torch.Tensor, im_gt: torch.Tensor) -> torch.Tensor:
        loss = 0

        if self.partialconv:
            eps = 1e-9
            mask = im_gt.sum(1, True) > eps
            mask = mask.float()

        features_input = self.normalize_inputs(im_out)
        features_target = self.normalize_inputs(im_gt)
        for i, layer in enumerate(self.resnet_layers):
            if isinstance(layer, PartialConv2d):
                features_input = layer(features_input, mask)
                features_target = layer(features_target, mask)
            else:
                features_input = layer(features_input)
                features_target = layer(features_target)

            if i in self.layers:
                loss = loss + F.l1_loss(features_input, features_target)

        return loss
