"""
Implements a network visualization in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

# import os
import torch
# import torchvision
# import torchvision.transforms as T
# import random
# import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from a4_helper import *


def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from network_visualization.py!')

def compute_saliency_maps(X, y, model):
  """
  Compute a class saliency map using the model for images X and labels y.

  Input:
  - X: Input images; Tensor of shape (N, 3, H, W)
  - y: Labels for X; LongTensor of shape (N,)
  - model: A pretrained CNN that will be used to compute the saliency map.

  Returns:
  - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
  images.
  """
  # Make input tensor require gradient
  X.requires_grad_()
  
  saliency = None
  ##############################################################################
  # TODO: Implement this function. Perform a forward and backward pass through #
  # the model to compute the gradient of the correct class score with respect  #
  # to each input image. You first want to compute the loss over the correct   #
  # scores (we'll combine losses across a batch by summing), and then compute  #
  # the gradients with a backward pass.                                        #
  # Hint: X.grad.data stores the gradients                                     #
  ##############################################################################
  # Replace "pass" statement with your code
  model.eval()
  N, _, H, W = X.shape
  y_hat = model(X)
  saliency = torch.zeros((N, H, W), dtype=X.dtype, device=X.device)
  for i in range(N):
    # print(f"{y[i]}, {y[i].dtype}")
    score = y_hat[i, y[i]]
    score.backward(retain_graph=True)
    saliency[i, :, :] = X.grad.data[i, ...].max(dim=0)[0]
    X.grad.zero_()
  
  # model.train()
  ##############################################################################
  #               END OF YOUR CODE                                             #
  ##############################################################################
  return saliency

def make_adversarial_attack(X, target_y, model, max_iter=100, verbose=True):
  """
  Generate an adversarial attack that is close to X, but that the model classifies
  as target_y.

  Inputs:
  - X: Input image; Tensor of shape (1, 3, 224, 224)
  - target_y: An integer in the range [0, 1000)
  - model: A pretrained CNN
  - max_iter: Upper bound on number of iteration to perform
  - verbose: If True, it prints the pogress (you can use this flag for debugging)

  Returns:
  - X_adv: An image that is close to X, but that is classifed as target_y
  by the model.
  """
  # Initialize our adversarial attack to the input image, and make it require gradient
  X_adv = X.clone()
  # X_adv = X_adv.requires_grad_()
  X_adv.requires_grad_()
  
  learning_rate = 1
  ##############################################################################
  # TODO: Generate an adversarial attack X_adv that the model will classify    #
  # as the class target_y. You should perform gradient ascent on the score     #
  # of the target class, stopping when the model is fooled.                    #
  # When computing an update step, first normalize the gradient:               #
  #   dX = learning_rate * g / ||g||_2                                         #
  #                                                                            #
  # You should write a training loop.                                          #
  #                                                                            #
  # HINT: For most examples, you should be able to generate an adversarial     #
  # attack in fewer than 100 iterations of gradient ascent.                    #
  # You can print your progress over iterations to check your algorithm.       #
  ##############################################################################
  # Replace "pass" statement with your code
  # print(X_adv.grad.data)
  model.eval()
  # max_iter = 3
  for iter in range(max_iter):
    y_hat = model(X_adv)
    score = y_hat[0, target_y]
    max_score = y_hat.max()
    if score == max_score:
      break

    score.backward()
    X_grad = X_adv.grad.data
    # print(f"{X_grad}, {(X_grad ** 2).sum()}")
    # X_grad = X_grad / (X_grad ** 2).sum().sqrt()
    # print(X_grad)
    # X_adv.grad.zero_()
    with torch.no_grad():
      X_adv = X_adv + learning_rate * (X_grad / (X_grad ** 2).sum().sqrt())
    X_adv.requires_grad_()
    # print(X_adv.grad.data)  ## not set up yet
    # print(X_adv)
    # X_adv.grad.zero_()
    
    
    if iter % 5 == 0:
      print(f"Iteration {iter}: target {score: .3f}, max: {max_score: .3f}")
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return X_adv


def class_visualization_step(img, target_y, model, **kwargs):
    """
    Performs gradient step update to generate an image that maximizes the 
    score of target_y under a pretrained model.
  
    Inputs:
    - img: random image with jittering as a PyTorch tensor  
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image
    
    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    """

    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    # Hint: You have to perform inplace operations on img.data to update   #
    # the generated image using gradient ascent & reset img.grad to zero   #
    # after each step.                                                     #
    ########################################################################
    # Replace "pass" statement with your code
    model.eval()
    img.requires_grad_()
    y_hat = model(img)
    score = y_hat[0, target_y]
    score.backward()
    with torch.no_grad():
      img = img + learning_rate * (img.grad.data - 2 * l2_reg * img)
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img
