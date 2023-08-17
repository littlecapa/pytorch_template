import torch
import logging

def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    
    
    # Convert log-probabilities to probabilities using exponentiation
    probabilities = torch.exp(outputs)
    
    # Extract the probability of the correct label for each example
    correct_label_probabilities = probabilities[range(outputs.size(0)), labels]
    
    # Compute 1 - probability of the correct label
    loss = 1 - correct_label_probabilities
    
    # Calculate the average loss across all examples
    loss = torch.mean(loss)
    
    return loss
    #num_examples = outputs.size()[0]
    #
    # return -torch.sum(outputs[range(num_examples), labels])/num_examples
