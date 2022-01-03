import torch
import torch.nn as nn

class MaskedLoss(nn.Module):
    """
    An interface for different losses that masks out padded tokens from gradient calculation.
    """
    def __init__(self, loss_fn):
        super(MaskedLoss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, prediction:torch.Tensor, target:torch.Tensor):
        """
        :param prediction: a tensor (ndim=2) containing logits for each item in the sequence; 
        its dimension is (sequence x logits_size);
        :param target: a tensor (ndim=1) containing class indices ground truths for each item in the sequence; its
        dimension is (sequence_size) 
        """
        pad_mask = torch.all(prediction != -1, dim=-1).int().detach() #getting a binary mask for vectors that are all -1 (padding tokens' vectors)
        target = target * pad_mask #zero-ing elementwise target first, since these have same size
        target = target.long()
        
        pad_mask = pad_mask.view(-1, 1) #reshaping for 
        prediction = pad_mask * prediction
        
        return self.loss_fn(prediction, target)
