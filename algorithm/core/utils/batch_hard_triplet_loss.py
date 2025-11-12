import torch
from torch import nn

class BatchHardTripletLoss(nn.Module):
    """
    Triplet loss with batch hard mining.
    Takes embeddings and labels as input.
    """
    def __init__(self, margin=1.0, p=2, epsilon=1e-8):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.epsilon = epsilon # Small value for numerical stability
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings (torch.Tensor): (batch_size, embedding_dim)
            labels (torch.Tensor): (batch_size,)
        """
        
        # Calculate the pairwise distance matrix
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances_sq = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
        
        # Clamp distances_sq to be non-negative (handling small numerical errors)
        distances_sq = torch.clamp(distances_sq, min=0.0)
        
        if self.p == 2:
            # Add epsilon BEFORE sqrt to prevent division by zero in backward pass
            distances = torch.sqrt(distances_sq + self.epsilon)
        else:
            distances = distances_sq # Fallback if p!=2 is ever used

            
        # --- Batch Hard Mining ---
        mask_positive = labels.unsqueeze(1).eq(labels.unsqueeze(0))
        mask_negative = ~mask_positive
        
        # Hardest Positive
        mask_positive_no_self = mask_positive.clone()
        mask_positive_no_self.fill_diagonal_(False)
        
        hardest_positive_dist = distances.clone()
        hardest_positive_dist[~mask_positive_no_self] = 0.0 # Use 0 for non-positives so max works
        # We use max here because we want the largest distance among positives
        hardest_positive_dist, _ = torch.max(hardest_positive_dist, dim=1)

        # Hardest Negative
        hardest_negative_dist = distances.clone()
        hardest_negative_dist[~mask_negative] = float('inf') # Use inf for non-negatives so min works
        # We use min here because we want the smallest distance among negatives
        hardest_negative_dist, _ = torch.min(hardest_negative_dist, dim=1)

        # --- Triplet Loss ---
        # loss = max(0, margin + hardest_positive - hardest_negative)
        triplet_loss = torch.clamp(self.margin + hardest_positive_dist - hardest_negative_dist, min=0.0)
        
        return triplet_loss.mean()