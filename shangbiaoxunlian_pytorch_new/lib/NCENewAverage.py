import torch
from torch.autograd import Function
from torch import nn
from .alias_multinomial import AliasMethod
import math


class NCEFunction(Function):
    @staticmethod
    def forward(self, x, y, memory, idx, params):
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()

        momentum = params[3].item()
        batchSize = x.size(0)
        outputSize = memory.size(0)
        inputSize = memory.size(1)

        # sample positives & negatives
        idx.select(1, 0).copy_(y.data)  # idx.select(1,0) = idx[:,0]
        # idx 使第一列为样本的index值，为indentity

        # sample correspoinding weights
        weight = torch.index_select(memory, 0, idx.view(-1))
        weight.resize_(batchSize, 4096 + 1, inputSize)
        # weight 是根据索引采样memory来构成的二维特征矩阵，一维是批次，一维是样本，memory是所有样本的特征

        # inner product
        out = torch.bmm(weight, x.data.resize_(batchSize, inputSize, 1))  # m = inputSize ???
        # bmm:batch matrix multiply . out:余弦距离(cos similarity)
        # out = [batchSize,K+1,1]  [batchSize,K+1]
        out.div_(T).exp_()  # [batchSize,K+1]   # exp(vfi/t)
        out_for_sort = out.narrow(1, 1, 4096)

        sorted, indexes_sort = torch.sort(out_for_sort, dim=1, descending=True, out=None)
        indexes_sort += 1

        indices_cpu = [i for i in range(int(K/2))]+[4096+i-int(K) for i in range(int(K/2))]
        #indices_cpu = [i for i in range(int(K))]
        indices = torch.cuda.LongTensor(indices_cpu)
        #indices = torch.cuda.LongTensor([0,1,2,3,4,5,6,7,8,9,10,4087,4088,4089,4090,4091,4092,4093,4094,4095])

        indexes_sort = indexes_sort.squeeze()

        idx_sorted = torch.cat((torch.cuda.LongTensor(batchSize, 1).zero_(),torch.index_select(indexes_sort, 1, indices)),1)

        sorted_out = out.narrow(1, 0, K+1)
        for i in range(batchSize):
           # out
           #torch.index_select(out[i], 0, idx_sorted[i])
           sorted_out[i] = torch.index_select(out[i], 0, idx_sorted[i])

        sorted_weight = weight.narrow(1, 0, K+1)
        for i in range(batchSize):
           # out
           #torch.index_select(weight[i], 0, idx_sorted[i])
           sorted_weight[i] = torch.index_select(weight[i], 0, idx_sorted[i])

        x.data.resize_(batchSize, inputSize)
        # K为采样的数目
        # inputSize 为特征的维度
        # outputSize 为总数据大小
        #out = sorted_out
        #weight = sorted_weight
        #out = out.narrow(1, 0, K + 1)
        #weight = weight.narrow(1, 0, K + 1)
        indices2_cpu = [i for i in range(int(K+1))]
        indices2 = torch.cuda.LongTensor(indices2_cpu)

        out = torch.index_select(out, 1, indices2)
        weight = torch.index_select(weight, 1, indices2)
        if Z < 0:
            params[2] = out.mean() * outputSize  # Z = n/m*sum(exp(vfi/t))  n=outputSize
            Z = params[2].item()
            print("normalization constant Z is set to {:.1f}".format(Z))

        #K=21
        out.div_(Z).resize_(batchSize, K + 1)  # P(i|v) = exp(vfi)/Z

        self.save_for_backward(x, memory, y, weight, out, params)

        # 此处返回的out 为采样样本的归一化后的概率
        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, weight, out, params = self.saved_tensors
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()
        momentum = params[3].item()
        batchSize = gradOutput.size(0)

        # gradients d Pm / d linear = exp(linear) / Z
        gradOutput.data.mul_(out.data)
        # add temperature
        gradOutput.data.div_(T)

        gradOutput.data.resize_(batchSize, 1, K + 1)

        # gradient of linear
        gradInput = torch.bmm(gradOutput.data, weight)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = weight.select(1, 0).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return gradInput, None, None, None, None

class NCENewAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, Z=None):
        super(NCENewAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)  # 采样方法
        self.multinomial.cuda()
        self.K = K  # 采样的负样本的个数，论文公式的m

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        batchSize = x.size(0)
        idx = self.multinomial.draw(batchSize * (4096+1)).view(batchSize, -1)  # 得到采样样本 +1+1+1!!!!!!
        out = NCEFunction.apply(x, y, self.memory, idx, self.params)
        return out

    def get_memory(self,y):
        return self.memory[y]