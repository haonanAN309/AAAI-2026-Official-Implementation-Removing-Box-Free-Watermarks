import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x+y)


class HidingResDGSL(nn.Module):
    def __init__(self, in_c=3, out_c=3, only_residual=False):
        super(HidingResDGSL, self).__init__()
        self.only_residual = only_residual

        self.conv1 = nn.Conv2d(in_c, 128, 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(128, affine=True)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        self.conv3 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)

        self.res1 = ResidualBlock(128, dilation=2)
        self.res2 = ResidualBlock(128, dilation=2)
        self.res3 = ResidualBlock(128, dilation=2)
        self.res4 = ResidualBlock(128, dilation=2)
        self.res5 = ResidualBlock(128, dilation=4)
        self.res6 = ResidualBlock(128, dilation=4)
        self.res7 = ResidualBlock(128, dilation=4)
        self.res8 = ResidualBlock(128, dilation=4)
        self.res9 = ResidualBlock(128, dilation=1)

        self.deconv3 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.norm4 = nn.InstanceNorm2d(128, affine=True)
        self.deconv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.norm5 = nn.InstanceNorm2d(128, affine=True)
        self.deconv1 = nn.Conv2d(128, out_c, 1)
        self.only_residual = only_residual

        # DGS-L control
        self.inject_dgs_l = False
        self.dgs_layer = 'shallow'  # shallow / middle / deep
        self.epsilon = 0.01

    def forward(self, x, clean_img=None):
        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y = F.relu(self.norm3(self.conv3(y)))

        if self.inject_dgs_l and self.dgs_layer == 'shallow' and clean_img is not None:
            y = generate_layer_perturbation(y, clean_img, decoder=self._decoder_from('middle'), epsilon=self.epsilon)

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        if self.inject_dgs_l and self.dgs_layer == 'middle' and clean_img is not None:
            y = generate_layer_perturbation(y, clean_img, decoder=self._decoder_from('deep'), epsilon=self.epsilon)

        y = self.res6(y)
        y = self.res7(y)
        y = self.res8(y)
        y = self.res9(y)

        y = F.relu(self.norm4(self.deconv3(y)))
        y = F.relu(self.norm5(self.deconv2(y)))

        if self.inject_dgs_l and self.dgs_layer == 'deep' and clean_img is not None:
            y = generate_layer_perturbation(y, clean_img, decoder=self.deconv1, epsilon=self.epsilon)

        if self.only_residual:
            y = self.deconv1(y)
        else:
            y = F.relu(self.deconv1(y))

        return y

    def _decoder_from(self, part):
        if part == 'middle':
            return nn.Sequential(
                self.res1, self.res2, self.res3, self.res4, self.res5,
                self.res6, self.res7, self.res8, self.res9,
                self.norm4, self.deconv3, self.norm5, self.deconv2, self.deconv1
            )
        elif part == 'deep':
            return nn.Sequential(
                self.res6, self.res7, self.res8, self.res9,
                self.norm4, self.deconv3, self.norm5, self.deconv2, self.deconv1
            )
        else:
            return self.deconv1



def generate_layer_perturbation(feature_map, clean_img, decoder, epsilon=0.01):
    """
    Inject orthogonal perturbation into intermediate feature_map,
    where perturbation direction is orthogonal to ∇L (as in DGS-L).

    Args:
        feature_map (Tensor): Intermediate feature tensor from encoder/decoder
        clean_img (Tensor): Ground truth image for computing loss
        decoder (Module): Decoder network from the injection layer to output
        epsilon (float): Perturbation strength

    Returns:
        Tensor: Perturbed feature map with orthogonal noise added
    """
    # Clone the feature and enable gradient tracking
    trans_feat = feature_map.clone().detach().requires_grad_(True)

    # Decode and compute loss
    output = decoder(trans_feat)
    loss = F.mse_loss(output, clean_img)
    loss.backward()

    with torch.no_grad():
        # Get normalized gradient direction ∇L̂
        grad = trans_feat.grad.detach()
        grad_flat = grad.view(grad.size(0), -1)
        unit_grad = grad_flat / (grad_flat.norm(p=2, dim=1, keepdim=True) + 1e-8)

        # Generate random noise and remove its projection on ∇L̂
        noise = torch.randn_like(grad).view(grad.size(0), -1)
        proj = (noise * unit_grad).sum(dim=1, keepdim=True) * unit_grad
        ortho_noise = noise - proj
        ortho_noise = ortho_noise / (ortho_noise.norm(p=2, dim=1, keepdim=True) + 1e-8)

        # Add orthogonal perturbation η to feature
        eta = ortho_noise.view_as(grad)
        perturbed_feat = trans_feat + epsilon * eta
        perturbed_feat = perturbed_feat.detach()

    # Release unused memory
    del loss, output, grad, eta, noise, proj, ortho_noise
    torch.cuda.empty_cache()

    return perturbed_feat
