import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mobileclassifier import mobile_vit_xx_small as create_model


def KL(alpha, c, device):
    beta = torch.ones((1, c)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step, device):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c, device)

    return (A + B)

def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C


class MobileVitClassifier(nn.Module):

    def __init__(self, classes, views, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(MobileVitClassifier, self).__init__()
        self.views = views
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.Classifiers = nn.ModuleList([create_model(self.classes) for _ in range(self.views)])
        self.embedding_layer = nn.Embedding(257, 40)

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.classes/S[v]

            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            S_a = self.classes / u_a
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a, u_a


        for v in range(len(alpha)-1):   # len(alpha)=6 大小为(200,10)
            if v==0:
                alpha_a, uncertain_a= DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a, uncertain_a= DS_Combin_two(alpha_a, alpha[v+1])

        return alpha_a, uncertain_a

    def forward(self, X, y, global_step, device):  # y(200) X是n个packet 里面是(byte_size, )
        con = []
        row_cat = []
        con_emb = []
        img_data = []
        data_x = []
        mul_img =[]


        for each_values in X.values():
            data_x.append(each_values)

        num_01 = len(data_x)//3
        data_x = [data_x[:num_01], data_x[num_01:2*num_01], data_x[-num_01:]]


        for ele in data_x:
            for j in range(1,num_01+1):
                b = ele[j-1]
                s_img = self.embedding_layer(ele[j-1].to(device))
                con.append(s_img)
                if j % 4 == 0:
                    a1 = torch.cat((con[0], con[1]), 2)
                    a2 = torch.cat((con[2], con[3]), 2)
                    l_img = torch.cat((a1, a2), 1)
                    con_emb.append(l_img)
                    con = []
            img_data.append(con_emb)
            con_emb = []
        for i in img_data:
            img = torch.stack(i, dim=1)
            mul_img.append(img)


        evidence = self.infer(mul_img)
        loss = 0
        alpha = dict()
        for v_num in range(self.views):
            alpha[v_num] = evidence[v_num] + 1
            loss += ce_loss(y, alpha[v_num], self.classes, global_step, self.lambda_epochs, device)
        alpha_a, uncertain_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs, device)
        loss = torch.mean(loss)
        return evidence, evidence_a, uncertain_a, loss

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """

        evidence = dict()
        for v_num in range(self.views):
            evidence[v_num] = self.Classifiers[v_num](input[v_num])
        return evidence




