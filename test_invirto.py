import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, neighborhood_size=3):
        super(FeatureExtractor, self).__init__()
        # 예시로 사전 학습된 WideResNet50 사용
        backbone = models.wide_resnet50_2(pretrained=True)

        # CNN의 각 계층 정의 (WideResNet50 기준)
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.pool_size = neighborhood_size

    def adaptive_local_pool(self, x, pool_size):
        # adaptive 평균 풀링을 통해 지역적(local) 특징 집계
        return F.adaptive_avg_pool2d(x, (x.size(2) // pool_size, x.size(3) // pool_size))

    def forward(self, x):
        # Level 1 (초기 특징, 주로 사용하지 않음)
        x = self.layer1(x)  # [B, C1, H1, W1]

        # Level 2
        feat2 = self.layer2(x)  # [B, C2, H2, W2]
        local_feat2 = self.adaptive_local_pool(feat2, self.pool_size)

        # Level 3
        feat3 = self.layer3(feat2)  # [B, C3, H3, W3]
        local_feat3 = self.adaptive_local_pool(feat3, self.pool_size)

        # 두 계층 특징을 같은 크기로 업샘플링 후 채널 방향으로 병합(concatenate)
        upsampled_feat2 = F.interpolate(local_feat2, size=local_feat3.shape[2:], mode='bilinear', align_corners=False)

        merged_feature = torch.cat([upsampled_feat2, local_feat3], dim=1)

        return merged_feature


# 사용 예시
if __name__ == "__main__":
    model = FeatureExtractor(neighborhood_size=3)
    input_tensor = torch.randn(1, 3, 288, 288)  # 예시 입력 이미지 (288x288 크기)
    output = model(input_tensor)

    print("Merged Feature shape:", output.shape)
