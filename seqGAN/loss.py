import torch
import torch.nn as nn
from torch.autograd import Variable


class NLLLoss(nn.Module):
    '''
    损失函数
    '''
    def __init__(self, weight):
        '''

        :param weight: class_num
        '''
        super(NLLLoss, self).__init__()
        self.weight = weight

    def forward(self, prob, target):
        '''

        :param prob: N*C
        :param target: N,
        :return:
        '''

        N = target.size(0)
        C = prob.size(1)
        weight = Variable(self.weight).view((1, -1))
        weight = weight.expand(N, C)  # (N, C)
        if prob.is_cuda:
            weight = weight.cuda()
        prob = weight * prob

        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        return -torch.sum(loss)