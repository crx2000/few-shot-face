"""
See original implementation (quite far from this one)
at https://github.com/jakesnell/prototypical-networks
"""
import torch
from torch import Tensor, nn

from .few_shot_classifier import FewShotClassifier
from .utils import compute_prototypes


class PrototypicalNetworks(nn.Module):
    """
    Jake Snell, Kevin Swersky, and Richard S. Zemel.
    "Prototypical networks for few-shot learning." (2017)
    https://arxiv.org/abs/1703.05175

    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their euclidean distance to the prototypes.
    """
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone



    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract feature vectors from the support set and store class prototypes.
        """

        support_features = self.backbone.forward(support_images)
        self._raise_error_if_features_are_multi_dimensional(support_features)
        self.prototypes = compute_prototypes(support_features, support_labels)

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> Tensor:
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)
        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores

        ##############
        # 原始版本
        # """
        # Overrides forward method of FewShotClassifier.
        # Predict query labels based on their distance to class prototypes in the feature space.
        # Classification scores are the negative of euclidean distances.
        # """
        # # Extract the features of query images
        # query_features = self.backbone.forward(query_images)
        # self._raise_error_if_features_are_multi_dimensional(query_features)
        #
        # # Compute the euclidean distance from queries to prototypes
        # scores = self.l2_distance_to_prototypes(query_features)
        #
        # return self.softmax_if_specified(scores)
        ############################################
    @staticmethod
    def is_transductive() -> bool:
        return False
