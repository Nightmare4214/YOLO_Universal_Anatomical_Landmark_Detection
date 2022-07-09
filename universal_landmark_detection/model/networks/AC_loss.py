#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import torch
from torch import nn


class AC_loss(nn.Module):

    def __init__(self, reduction='mean') -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.reduction = reduction

    def unravel_index(self, index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = torch.div(index, dim, rounding_mode='floor')
        return tuple(reversed(out))

    def angle_matrix(self, predict_xy):
        predict_xy = predict_xy / torch.norm(predict_xy, p=2, dim=-1, keepdim=True)
        # predict_cos = torch.einsum('ni,mi->nm', predict_xy, predict_xy)
        predict_cos = torch.mm(predict_xy, predict_xy.T)
        return torch.acos(predict_cos)

    def forward(self, predict, gt):
        B, C, H, W = predict.shape
        loss = 0
        for b in range(B):
            predict_xy = []
            gt_xy = []
            # landmark
            for c in range(C):
                predict_xy.append(self.unravel_index(torch.argmax(predict[b, c]), (H, W)))
                gt_xy.append(self.unravel_index(torch.argmax(gt[b, c]), (H, W)))

            predict_xy = torch.tensor(predict_xy, dtype=torch.float)
            gt_xy = torch.tensor(gt_xy, dtype=torch.float)
            # distance matrix
            predict_D = torch.cdist(torch.unsqueeze(predict_xy, 0), torch.unsqueeze(predict_xy, 0), p=2)
            gt_D = torch.cdist(torch.unsqueeze(gt_xy, 0), torch.unsqueeze(gt_xy, 0), p=2)
            # angle matrix
            predict_A = self.angle_matrix(predict_xy)
            gt_A = self.angle_matrix(gt_xy)

            w_ac = torch.log2(self.mse(predict_D, gt_D)) + torch.log2(self.mse(predict_A, gt_A))
            l2 = self.mse(predict[b], gt[b])
            loss += w_ac * l2
        if self.reduction == 'mean':
            return loss / B
        return loss


if __name__ == '__main__':
    a = torch.rand((4, 19, 416, 512))
    b = torch.rand((4, 19, 416, 512))
    ac_loss = AC_loss()
    print(ac_loss(a, b))
