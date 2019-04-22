import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch
from tqdm import tqdm_notebook
import time
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def compute_loss(model, X_batch, y_batch, compute_grad=False):
    X_batch = Variable(torch.tensor(X_batch, requires_grad=compute_grad).cuda())
    y_batch = Variable(torch.tensor(y_batch).cuda())
    logits = model(X_batch)
    if compute_grad:
        return F.cross_entropy(logits, y_batch).mean(), X_batch
    else:
        return F.cross_entropy(logits, y_batch).mean()

def attack_optim(model, input_image, true_class, target_class, max_iter=1000, threshold=10, verbose=False):
    """
    Parameters
    ----------
    input_image: 
      Tensor
    true_class: int
      Image class
    target_class: int
      Class that we want our model to classify as
    tol: float
      Optimization tolerance
    num_iter: int
      Maximum number of iterations of optimization method

    Returns
    ----------
    pertrubation: FloatTensor
      Pertrubation
    """
    par = nn.Parameter(torch.zeros_like(input_image).uniform_())

    optimizer = torch.optim.Adam([par], lr=0.008)

    x = Variable(torch.tensor(input_image.unsqueeze(0), requires_grad=True).cuda())
    y_target = Variable(torch.tensor([target_class]).cuda())
    y_true = Variable(torch.tensor([true_class]).cuda())
    initial_prediction = model(x).argmax(1).item()

    if initial_prediction != y_true:
            warnings.warn('The model incorrecly classifies the image', RuntimeWarning)

    for i in range(max_iter):
        if verbose:
            print(i)
        optimizer.zero_grad()

        loss, _ = compute_loss(model, torch.clamp(x + par, 0, 1), y_target, compute_grad=True)
        
        regularization = par.pow(2).sum() + par.abs().sum()
        if verbose:
            print(regularization)

        loss += regularization

        loss.backward()
        optimizer.step()

        adv_prediction = model(x + par).argmax(1).item()

    if adv_prediction == y_target and regularization < threshold:
        if verbose:
            print('Found example') 
        return par.data

def attack_fastgrad(model, input_image, target_class, verbose=False):
    """
    Parameters
    ----------
    model: torch.Sequential
        Model that we are trying to fool
    input_image: torch.FloatTensor
        Image in the form of FloatTensor that will be fed to a model
    target_class: torch.LongTensor
      Image class
    tol: float
      Optimization tolerance
    num_iter: int
      Maximum number of iterations of optimization method

    Returns
    ----------
    pertrubation: FloatTensor
      Pertrubation
    """
    model.cuda()

    x = Variable(torch.tensor(input_image.unsqueeze(0), requires_grad=True)).cuda()
    y = Variable(torch.tensor([target_class])).cuda()
    
    initial_prediction = model(x).argmax(1).cpu().data.numpy()
    
    if initial_prediction != y:
        warnings.warn('The model incorrecly classifies the image', RuntimeWarning)
    
    loss, _ = compute_loss(model, x, y, compute_grad=True)
    loss.backward()
    
    EPS = 0.009

    gradient = torch.sign(x.cpu().grad.data)

    adversarial_example = x.data + EPS * gradient
    adversarial_prediction = model(adversarial_example.cuda()).argmax(1).cpu().data.numpy()
    
    if verbose:
        print('Prediction before the attack: {}'.format(initial_prediction))
        print('Prediction after the attack: {}'.format(adversarial_prediction))

    return EPS * gradient
