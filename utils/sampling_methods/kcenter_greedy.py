"""This module comprises PatchCore Sampling Methods for the embedding.
- k Center Greedy Method
    Returns points that minimizes the maximum distance of any point to a center.
    . https://arxiv.org/abs/1708.00489
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from sklearn.random_projection import SparseRandomProjection


class kCenterGreedy:
    """Implements k-center-greedy method.
    Args:
        embedding (Tensor): Embedding vector extracted from a CNN
        sampling_size (int): choose coreset size from the embedding size.
    Example:
        >>> embedding.shape
        torch.Size([219520, 1536])
        >>> sampler = KCenterGreedy(embedding=embedding, sampling_size=select_batch_size)
        >>> sampled_idxs = sampler.select_coreset_idxs()
        >>> coreset = embedding[sampled_idxs]
        >>> coreset.shape
        torch.Size([219, 1536])
    """

    def __init__(self, embedding: Tensor, sampling_size: int) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = embedding
        self.coreset_size = sampling_size
        self.model = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma

        self.features: Tensor
        self.min_distances: Tensor = None
        self.n_observations = self.embedding.shape[0]

    def reset_distances(self) -> None:
        """Reset minimum distances."""
        self.min_distances = None

    def update_distances(self, cluster_centers: List[int]) -> None:
        """Update min distances given cluster centers.
        Args:
            cluster_centers (List[int]): indices of cluster centers
        """

        if cluster_centers:
            centers = self.features[cluster_centers]

            distance = F.pairwise_distance(self.features, centers, p=2).reshape(-1, 1)

            if self.min_distances is None:
                self.min_distances = distance
            else:
                self.min_distances = torch.minimum(self.min_distances, distance)

    def get_new_idx(self) -> int:
        """Get index value of a sample.
        Based on minimum distance of the cluster
        Returns:
            int: Sample index
        """

        if isinstance(self.min_distances, Tensor):
            idx = int(torch.argmax(self.min_distances).item())
        else:
            raise ValueError(f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}")

        return idx

    def select_coreset_idxs(self, selected_idxs: Optional[List[int]] = None) -> List[int]:
        """Greedily form a coreset to minimize the maximum distance of a cluster.
        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.
        Returns:
          indices of samples selected to minimize distance to cluster centers
        """

        if selected_idxs is None:
            selected_idxs = []

        if self.embedding.ndim == 2:
            self.model.fit(self.embedding)
            self.features = torch.Tensor(self.model.transform(self.embedding)).to(self.device)
            self.reset_distances()
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1).to(self.device)
            self.update_distances(cluster_centers=selected_idxs)

        selected_coreset_idxs: List[int] = []
        idx = int(torch.randint(high=self.n_observations, size=(1,)).item())
        for _ in range(self.coreset_size):
            self.update_distances(cluster_centers=[idx])
            idx = self.get_new_idx()
            if idx in selected_idxs:
                raise ValueError("New indices should not be in selected indices.")
            self.min_distances[idx] = 0
            selected_coreset_idxs.append(idx)

        return selected_coreset_idxs

    def sample_coreset(self, selected_idxs: Optional[List[int]] = None) -> Tensor:
        """Select coreset from the embedding.
        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.
        Returns:
            Tensor: Output coreset
        Example:
            >>> embedding.shape
            torch.Size([219520, 1536])
            >>> sampler = KCenterGreedy(...)
            >>> coreset = sampler.sample_coreset()
            >>> coreset.shape
            torch.Size([219, 1536])
        """

        idxs = self.select_coreset_idxs(selected_idxs)
        coreset = self.embedding[idxs]

        return coreset