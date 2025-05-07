import torch

class SemanticFocalLoss(torch.nn.Module):
    """
    Focal loss implementation from:

    - Lin, Tsung-Yi, et al. "Focal loss for dense object detection."
      Proceedings of the IEEE international conference on computer vision. 2017.
    """
    def __init__(self, gamma=0.0, weight=None):
        """
        Parameters
        ==========
            gamma : float - the regularization parameter, gamma = 0. will return the value as the SemanticCrossEntropyLoss.
            weight: Optional[Tensor] - a manual rescaling weight given to each class. If given, has to be a Tensor of size C
        """
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels):
        """
        Parameters
        ==========
            preds: Tensor - size (b, C, H, W)
            labels: Tensor - size (b, H, W)
        Returns
        =======
            f_loss: Tensor - size () the focal loss
        """
        n_classes = preds.size(1)
        preds = preds.permute(0, 2, 3, 1).reshape(-1, n_classes)
        labels = labels.flatten()
        probs = torch.nn.functional.softmax(preds, dim=-1)
        weighted_probs = (1 - probs) ** self.gamma * probs.log()
        return torch.nn.functional.nll_loss(weighted_probs, labels, weight=self.weight)