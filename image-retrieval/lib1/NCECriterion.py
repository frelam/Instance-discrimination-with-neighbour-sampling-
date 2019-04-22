import torch
from torch import nn

eps = 1e-7

class NCECriterion(nn.Module):

    def __init__(self, nLem):  #nLem训练样本大小
        super(NCECriterion, self).__init__()
        self.nLem = nLem

    def forward(self, x, targets):
        batchSize = x.size(0)
        K = x.size(1)-1
        # x.size() 包括了positive和negative samples的数目。
        # K为 negative sample 的数目。
        Pnt = 1 / float(self.nLem)
        # Pnt 为 Pn(i) = 1/n
        Pns = 1 / float(self.nLem)
        
        # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt) 
        Pmt = x.select(1,0)
        # Pmt 即公式里的 P(i|v)
        Pmt_div = Pmt.add(K * Pnt + eps)  #P(i|v) + m*Pn(i)
        lnPmt = torch.div(Pmt, Pmt_div)
        # lnPmt = h(i|v) = P(i|v) / P(i|v) + m*Pn(i)


        # eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)  # Pns = Pn(i)
        Pon_div = x.narrow(1,1,K).add(K * Pns + eps)
        Pon = Pon_div.clone().fill_(K * Pns)
        lnPon = torch.div(Pon, Pon_div)
        # h(i|v')

        # equation 6 in ref. A
        lnPmt.log_()
        lnPon.log_()
        
        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.view(-1, 1).sum(0)
        
        loss = - (lnPmtsum + lnPonsum) / batchSize
        # loss = -E{log(h(i|v))} - E{log(1 - h(i|v'))}
        return loss

