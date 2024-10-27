import numpy as np
import torch
from torch.nn import functional as F

############################# NegLS #############################

def ls_scheduler(epoch, n_epochs):
    if epoch < 50:
        return 0.0
    if epoch < 75:
        factor = 1 / (n_epochs - 50)
        return factor * (epoch + 1 - 50)
    return 1

def mnist_ls_scheduler(epoch, n_epochs):
    if epoch < 5:
        return 0.0
    if epoch < 7:
        factor = 1 / (n_epochs - 5)
        return factor * (epoch + 1 - 5)
    return 1

def cifar10_ls_scheduler(epoch, n_epochs):
    if epoch < 50:
        return 0.0
    if epoch < 75:
        factor = 1 / (n_epochs - 50)
        return factor * (epoch + 1 - 50)
    return 1

class NegLSCrossEntropyLoss(torch.nn.Module):
    def __init__(self, init_ls=0.0, scheduler=ls_scheduler):
        super().__init__()
        self.init_ls = init_ls
        self.label_smoothing = init_ls
        self.scheduler = scheduler

    def step(self, epoch, n_epochs):
        if self.scheduler is None:
            return
        self.label_smoothing = self.init_ls * self.scheduler(epoch, n_epochs)

    def forward(self, logits, labels):
        confidence = 1.0 - self.label_smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.label_smoothing * smooth_loss
        loss_numpy = loss.data.cpu().numpy()
        num_batch = len(loss_numpy)
        return torch.sum(loss) / num_batch

############################# BiDO #############################

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.squeeze(torch.eye(num_classes, device=y.device)[y], dim=1)

def distmat(X):
    """ distance matrix """
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D

def sigma_estimation(X, Y):
    """ sigma from median distance"""
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1E-2:
        med = 1E-2
    return med

def kernelmat(X, sigma, ktype='gaussian', measure='HSIC'):
    """ kernel matrix baker"""
    m = int(X.size()[0])
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])

    if ktype == "gaussian":
        Dxx = distmat(X)
        if sigma:
            variance = 2. * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor) # kernel matrices
        else:
            try:
                sx = sigma_estimation(X, X)
                Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)))
    elif ktype == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)
    elif ktype == 'IMQ':
        Dxx = distmat(X)
        Kx = 1 * torch.rsqrt(Dxx + 1)
    else:
        raise NotImplementedError(f"ktype {ktype} not implemented.")

    if measure == 'HSIC':
        return torch.mm(Kx, H)
    if measure == 'COCO':
        return torch.mm(H, torch.mm(Kx, H))

    raise NotImplementedError(f"Measure {measure} not implemented.")

def hsic_normalized_cca(x, y, sigma, ktype='gaussian'):
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma, measure='HSIC')
    Kyc = kernelmat(y, sigma=sigma, ktype=ktype, measure='HSIC')

    epsilon = 1E-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
    Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy

def coco_normalized_cca(x, y, sigma, ktype='gaussian'):
    m = int(x.size()[0])
    K = kernelmat(x, sigma=sigma, measure='COCO')
    L = kernelmat(y, sigma=sigma, ktype=ktype, measure='COCO')

    res = torch.sqrt(torch.norm(torch.mm(K, L))) / m
    return res

class BiDOLoss(torch.nn.Module):
    def __init__(self, n_classes=1000, measure='HSIC', ktype='linear', lambda_xz=0.05, lambda_yz=0.5, sigma=5):
        super().__init__()
        self.n_classes = n_classes
        self.measure = measure
        if measure == 'HSIC':
            self.measure_func = hsic_normalized_cca
        elif measure == 'COCO':
            self.measure_func = coco_normalized_cca
        else:
            raise NotImplementedError(f"Measure {measure} not implemented.")
        if ktype not in ['gaussian', 'linear', 'IMQ']:
            raise NotImplementedError(f"ktype {ktype} not implemented.")
        self.ktype = ktype
        self.lambda_xz = lambda_xz
        self.lambda_yz = lambda_yz
        self.sigma = sigma

    def forward(self, inputs, hiddens, labels):
        bs = inputs.size(0)
        target = to_categorical(labels, num_classes=self.n_classes).float()
        data = inputs.view(bs, -1)
        loss = 0
        for hidden in hiddens:
            hidden = hidden.view(bs, -1)
            hxz = self.measure_func(hidden, data, sigma=self.sigma)
            hyz = self.measure_func(hidden, target, sigma=self.sigma, ktype=self.ktype)
            loss += self.lambda_xz * hxz - self.lambda_yz * hyz
        return loss
