import torch
import torch.nn.functional as F

def label_smoothing_loss(logits, target, smoothing=0.1, weight_matrix=None):
    """
    Custom label smoothing loss for non-uniform smoothing.
    
    Parameters:
      logits: Tensor of shape (batch_size, num_classes)
      target: Tensor of shape (batch_size,) with class indices (0-indexed)
      smoothing: Smoothing factor (e.g., 0.1)
      
    Returns:
      Scalar loss value.
    """
    num_classes = logits.size(1)
    weight_matrix = weight_matrix.to(logits.device)
    
    
    # custom_weights = torch.tensor([
    #     [0.0,   0.15, 0.7,  0.15],   # True label 0 (class 1): similar to class 2 (index 2)
    #     [0.15,  0.0,  0.7,  0.15],   # True label 1 (class 2): similar to class 2 (index 2)
    #     [0.7,   0.15, 0.0,  0.15],   # True label 2 (class 3): similar to class 0 (index 0)
    #     [0.15,  0.15, 0.7,  0.0]     # True label 3 (class 4): similar to class 2 (index 2)
    # ], dtype=logits.dtype, device=logits.device)
    
    # Build the smoothed target distribution for each sample.
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        # print(true_dist)
        for i in range(logits.size(0)):
            label = target[i]
            # print(label)
            # Distribute the smoothing mass among the wrong classes according to custom_weights.
            true_dist[i] = weight_matrix[label] * smoothing
            # print(true_dist[i])
            # Assign the main probability mass to the correct class.
            true_dist[i, label] = 1 - smoothing
        # print(true_dist)
    
    # Compute log probabilities from the logits.
    log_probs = F.log_softmax(logits, dim=1)
    # Compute the cross-entropy loss between the soft targets and the predictions.
    loss = torch.mean(torch.sum(-true_dist * log_probs, dim=1))
    return loss

logits = torch.tensor([[0.1, 0.2, 0.3, 200],[0.2,500,0.3,0.5]])
target = torch.tensor([3,1])
smoothing = 0.1
weight_matrix = torch.tensor([
        [0.0,   0.15, 0.7,  0.15],   # True label 0 (class 1): similar to class 2 (index 2)
        [0.15,  0.0,  0.7,  0.15],   # True label 1 (class 2): similar to class 2 (index 2)
        [0.7,   0.15, 0.0,  0.15],   # True label 2 (class 3): similar to class 0 (index 0)
        [0.15,  0.15, 0.7,  0.0]     # True label 3 (class 4): similar to class 2 (index 2)
    ])

loss = label_smoothing_loss(logits, target, smoothing, weight_matrix)
print("Loss:", loss.item())
