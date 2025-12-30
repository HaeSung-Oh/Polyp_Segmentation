import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GLAM(nn.Module):
    def __init__(self, channels):
        super(GLAM, self).__init__()
        
        self.lcm_branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.lcm_branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.lcm_branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.lcm_branch4 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.lcm_fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.gcm_branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.gcm_branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.gcm_branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=12, dilation=12),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.gcm_branch4 = nn.AdaptiveAvgPool2d(1)
        self.gcm_branch5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=16, dilation=16),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.gcm_fusion = nn.Sequential(
            nn.Conv2d(channels * 5, channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.intermediate_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )


        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.size()[2:]
        
        # Local Context Module
        lcm1 = self.lcm_branch1(x)
        lcm2 = self.lcm_branch2(x)
        lcm3 = self.lcm_branch3(x)
        lcm4 = self.lcm_branch4(x)
        local_concat = torch.cat([lcm1, lcm2, lcm3, lcm4], dim=1)
        local_feat = self.lcm_fusion(local_concat)

        # Global Context Module
        gcm1 = self.gcm_branch1(x)
        gcm2 = self.gcm_branch2(x)
        gcm3 = self.gcm_branch3(x)
        gcm4 = self.gcm_branch4(x)
        gcm4 = F.interpolate(gcm4, size=size, mode='bilinear', align_corners=True)
        gcm5 = self.gcm_branch5(x)
        global_concat = torch.cat([gcm1, gcm2, gcm3, gcm4, gcm5], dim=1)
        global_feat = self.gcm_fusion(global_concat)

        # Fusion
        combined_feat = torch.cat([local_feat, global_feat], dim=1)
        fused_feat = self.intermediate_fusion(combined_feat)

        context_enhanced = fused_feat

        return self.final_conv(context_enhanced)


class EfficientNetEncoder(nn.Module):
    def __init__(self, pretrained=True, model_name='efficientnet_v2_s'):
        super(EfficientNetEncoder, self).__init__()
        
        # EfficientNet 
        if model_name == 'efficientnet_b0':
            try:
                from torchvision.models import efficientnet_b0
                backbone = efficientnet_b0(pretrained=pretrained)
            except:
                print(f"Warning: {model_name} not available, using efficientnet_v2_s")
                from torchvision.models import efficientnet_v2_s
                backbone = efficientnet_v2_s(pretrained=pretrained)
                
        elif model_name == 'efficientnet_b3':
            try:
                from torchvision.models import efficientnet_b3
                backbone = efficientnet_b3(pretrained=pretrained)
            except:
                print(f"Warning: {model_name} not available, using efficientnet_v2_s")
                from torchvision.models import efficientnet_v2_s
                backbone = efficientnet_v2_s(pretrained=pretrained)
                
        elif model_name == 'efficientnet_v2_s':
            from torchvision.models import efficientnet_v2_s
            backbone = efficientnet_v2_s(pretrained=pretrained)
            
        elif model_name == 'efficientnet_v2_m':
            from torchvision.models import efficientnet_v2_m
            backbone = efficientnet_v2_m(pretrained=pretrained)
            
        else:
            print(f"Warning: {model_name} not supported, using efficientnet_v2_s")
            from torchvision.models import efficientnet_v2_s
            backbone = efficientnet_v2_s(pretrained=pretrained)
            

        features = list(backbone.features.children())
        
        self.stage0 = features[0]  # Stem
        self.stage1 = features[1]  # Stage 1
        self.stage2 = features[2]  # Stage 2
        self.stage3 = nn.Sequential(*features[3:5])  # Stage 3
        self.stage4 = nn.Sequential(*features[5:])   # Stage 4
        
        self._get_feature_channels()
        
    def _get_feature_channels(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            e0 = self.stage0(dummy_input)
            e1 = self.stage1(e0)
            e2 = self.stage2(e1)
            e3 = self.stage3(e2)
            e4 = self.stage4(e3)
            
            self.feature_channels = [
                e0.shape[1],  # stage0 
                e1.shape[1],  # stage1 
                e2.shape[1],  # stage2 
                e3.shape[1],  # stage3 
                e4.shape[1]   # stage4 
            ]
            
        print(f"EfficientNet feature channels: {self.feature_channels}")
        
    def forward(self, x):

        e0 = self.stage0(x)      # 1/2 size
        e1 = self.stage1(e0)     # 1/4 size  
        e2 = self.stage2(e1)     # 1/8 size
        e3 = self.stage3(e2)     # 1/16 size
        e4 = self.stage4(e3)     # 1/32 size
        
        return e0, e1, e2, e3, e4


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, stride):
        super(DecoderBlock, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=stride, stride=stride)

        self.context_module = GLAM(skip_channels)

        self.post_fusion = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):  # skip: e0
        x = self.upconv(x)
        
        skip_post = self.context_module(skip) + skip
        if skip_post.shape[2:] != x.shape[2:]:
            skip_post = F.interpolate(skip_post, size=x.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip_post], dim=1)
        x = self.post_fusion(x)

        return x


class model(nn.Module):
    def __init__(self, in_channels=3, num_classes=8, model_name='efficientnet_v2_s', pretrained=True):
        super().__init__()

        self.encoder = EfficientNetEncoder(pretrained=pretrained, model_name=model_name)
        feature_channels = self.encoder.feature_channels

        self.global_context = GLAM(feature_channels[-1])

        # skip 
        self.decoder1 = DecoderBlock(
            feature_channels[-1],  # e4
            feature_channels[0],   # e0 only
            128,
            stride=32
        )

        self.final_conv = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x, return_features=False):
        input_size = x.shape[2:]

        e0, e1, e2, e3, e4 = self.encoder(x)
        g = self.global_context(e4) + e4

        # skip e1, e0
        d1 = self.decoder1(g, e0)

        out = self.final_conv(d1)

        return out
