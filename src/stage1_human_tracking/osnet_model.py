"""
OSNet: Omni-Scale Network for Person Re-Identification
Paper: https://arxiv.org/abs/1905.00953

This implementation extracts 256-dimensional identity features
for robust person tracking across different camera viewpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """Convolution layer with batch normalization and ReLU activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution using depthwise separable convolution."""
    
    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, 
                               padding=1, groups=in_channels, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, stride=1, 
                               padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    """Channel attention gate for feature recalibration."""
    
    def __init__(self, channels, reduction=16):
        super(ChannelGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class OSBlock(nn.Module):
    """Omni-Scale residual block with multiple receptive fields."""
    
    def __init__(self, in_channels, out_channels, reduction=4):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // reduction
        
        self.conv1 = ConvLayer(in_channels, mid_channels, 1)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = LightConv3x3(mid_channels, mid_channels)
        self.conv2c = LightConv3x3(mid_channels, mid_channels)
        self.conv2d = LightConv3x3(mid_channels, mid_channels)
        self.gate = ChannelGate(mid_channels)
        self.conv3 = ConvLayer(mid_channels, out_channels, 1)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = ConvLayer(in_channels, out_channels, 1)
    
    def forward(self, x):
        identity = x
        
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x2a)
        x2c = self.conv2c(x2b)
        x2d = self.conv2d(x2c)
        
        # Aggregate multi-scale features
        x2 = x2a + x2b + x2c + x2d
        x2 = self.gate(x2)
        x3 = self.conv3(x2)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out = x3 + identity
        return F.relu(out)


class OSNet(nn.Module):
    """
    OSNet model for extracting 256-dimensional person identity features.
    
    Architecture:
    - Input: 256x128x3 RGB image
    - Output: 256-dimensional feature vector
    """
    
    def __init__(self, num_classes=1000, feature_dim=256):
        super(OSNet, self).__init__()
        
        # Initial convolution
        self.conv1 = ConvLayer(3, 64, 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Omni-Scale blocks
        self.conv2 = self._make_layer(64, 256, 2)
        self.conv3 = self._make_layer(256, 384, 2)
        self.conv4 = self._make_layer(384, 512, 2)
        
        # Global average pooling
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension reduction
        self.fc = nn.Linear(512, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)
        
        # Classifier (for training, not used in inference)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        self._init_params()
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(OSBlock(in_channels, out_channels))
        for _ in range(1, num_blocks):
            layers.append(OSBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_feats=False):
        """
        Args:
            x: Input tensor of shape (batch, 3, 256, 128)
            return_feats: If True, return features before classifier
        
        Returns:
            features: 256-dimensional feature vector (if return_feats=True)
            logits: Classification logits (if return_feats=False)
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        
        features = self.fc(x)
        features = self.bn(features)
        
        if return_feats:
            return features
        
        logits = self.classifier(features)
        return logits


def load_pretrained_weights(model, weight_path):
    """
    Load pre-trained weights into OSNet model.
    
    Args:
        model: OSNet model instance
        weight_path: Path to .pth weight file
    
    Returns:
        model: Model with loaded weights
    """
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # Load weights (ignore classifier if dimensions don't match)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() 
                      if k in model_dict and model_dict[k].size() == v.size()}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    return model


def build_osnet(weight_path, device='cpu'):
    """
    Build OSNet model and load pre-trained weights.
    
    Args:
        weight_path: Path to pre-trained weights
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        model: Ready-to-use OSNet model in eval mode
    """
    model = OSNet(num_classes=1000, feature_dim=256)
    model = load_pretrained_weights(model, weight_path)
    model = model.to(device)
    model.eval()
    
    return model