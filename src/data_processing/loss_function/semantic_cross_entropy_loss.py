import torch

class SemanticCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight=None):
        """
        Parameters
        ==========
            weight: Optional[Tensor] - a manual rescaling weight given to each class. If given, has to be a Tensor of size C
        """
        super().__init__()
        self.weight = weight

    def forward(self, preds, labels):
        """
        Parameters
        ==========
            preds: Tensor - size (b, C, H, W)
            labels: Tensor - size (b, H, W)
        Returns
        =======
            ce_loss: Tensor - size () the cross entropy loss
        """
        n_classes = preds.size(1)
        preds = preds.permute(0, 2, 3, 1).reshape(-1, n_classes)
        labels = labels.flatten()
        return torch.nn.functional.cross_entropy(preds, labels, weight=self.weight)
