import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import Variable
import json


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        # inputs.shape=torch.Size([6, 10]) and targets shape=torch.Size([6])

        N = inputs.size(0)  # 6
        # print(N)
        C = inputs.size(1)  # 10
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        #         print("Focal loss forward inputs shape={} and alpha shape={}".format(inputs.size(), alpha.size()))
        #         print("Focal loss forward ids shape={} and ids.data shape={}".format(ids.size(), ids.data.view(-1).size()))  # [6, 1], [6]

        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class binary_FocalLoss(nn.Module):
    def __init__(self, gamma, class_num, class_count_json, size_average=True):
        super(binary_FocalLoss, self).__init__()
        with open(class_count_json, 'r') as fb:
            self.class_ratio = json.load(fb)
        self.gamma = gamma
        self.class_num = class_num
        # self.beta = 0.999
        self.size_average = size_average
        self._init_class_weight()

    def _init_class_weight(self):
        self.register_buffer('class_weight', torch.zeros(80))
        for i in range(1, 81):
            self.class_weight[i - 1] = 1 - self.class_ratio[str(i)]
            # n = self.class_ratio[str(i)]
            # self.class_weight[i - 1] = (1 - self.beta) / (1 - self.beta ** n)

    def forward(self, inputs, targets):
        '''
        inputs: (N, C) -- result of sigmoid
        targets: (N, C) -- one-hot variable
        '''
        assert self.class_num == targets.size(1)
        assert self.class_num == inputs.size(1)
        assert inputs.size(0) == targets.size(0)

        weight_matrix = self.class_weight.expand(inputs.size(0), self.class_num)
        weight_p1 = torch.exp(weight_matrix[targets == 1])
        weight_p0 = torch.exp(1 - weight_matrix[targets == 0])
        # weight_p1 = weight_matrix[targets == 1]
        # weight_p0 = 1 - weight_matrix[targets == 0]
        p_1 = inputs[targets == 1]
        p_0 = inputs[targets == 0]

        # loss = torch.sum(torch.log(p_1)) + torch.sum(torch.log(1 - p_0))  # origin bce loss
        loss1 = torch.pow(1 - p_1, self.gamma) * torch.log(p_1) * weight_p1
        loss2 = torch.pow(p_0, self.gamma) * torch.log(1 - p_0) * weight_p0
        loss = -torch.sum(loss1) - torch.sum(loss2)
        if self.size_average:
            loss /= inputs.size(0)

        return loss
