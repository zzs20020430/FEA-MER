import torch.nn as nn
import torch

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, input, target):
#         ce_loss = nn.CrossEntropyLoss(reduction=self.reduction)(input, target)

#         if self.alpha is not None:
#             alpha = self.alpha.to(target.device)
#             pt = torch.exp(-ce_loss)
#             focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
#         else:
#             focal_loss = (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss

#         return focal_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)

        if self.alpha is not None:
            alpha = self.alpha.to(target.device)
            # 获取每个样本对应的权重
            at = alpha.gather(0, target.data.view(-1))
            pt = torch.exp(-ce_loss)
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss
        
        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss