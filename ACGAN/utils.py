import torch

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    x = labels.view(-1).data
    y = preds.view(-1).data
    x = [1. if x_s > 0.5 else 0. for x_s in x]
    y = [1. if y_s > 0.5 else 0. for y_s in y]
    z = [1 if x_s == y_s else 0 for x_s, y_s in zip(x,y)]
    correct = sum(z)
    acc = float(correct) / float(len(z)) * 100.0
    return acc

# compute the gradient penalty
def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).cuda()
    dydx = torch.autograd.grad(outputs=y,
                                inputs=x,
                                grad_outputs=weight,
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)
