import logging
import numpy as np
from scipy.spatial import cKDTree

from .helpers import build_index_build_kwargs, build_index_search_kwargs

logger = logging.getLogger(__name__)


class ANNEntityIndex:
    def __init__(self, embedding_size):
        self.dimension = embedding_size
        self.data = []
        self.vector_idx_to_id = None
        self.is_built = False

    def insert_vector_dict(self, vector_dict):
        self.vector_idx_to_id = dict(enumerate(vector_dict.keys()))
        self.data = list(vector_dict.values())
        self.is_built = True

    def build(self, index_build_kwargs=None):
        if self.vector_idx_to_id is None:
            raise ValueError("Please call insert_vector_dict first")

        self.is_built = True

    def search_pairs(self, k, sim_threshold, index_search_kwargs=None):
        if not self.is_built:
            raise ValueError("Please call build first")
        if sim_threshold > 1 or sim_threshold < 0:
            raise ValueError(f"sim_threshold={sim_threshold} must be <= 1 and >= 0")

        logger.debug("Searching on approx_knn_index...")

        distance_threshold = 1 - sim_threshold

        index_search_kwargs = build_index_search_kwargs(index_search_kwargs)

        # Using cKDTree for approximate nearest neighbor search
        tree = cKDTree(np.array(self.data))
        distances, indices = tree.query(self.data, k=k+1)  # k+1 to include self

        found_pair_set = set()
        for i in range(len(self.data)):
            for j in range(1, k+1):
                if i != indices[i, j] and distances[i, j] <= distance_threshold:
                    left_id = self.vector_idx_to_id[i]
                    right_id = self.vector_idx_to_id[indices[i, j]]
                    pair = tuple(sorted([left_id, right_id]))
                    found_pair_set.add(pair)

        logger.debug(
            f"Building found_pair_set done. Found len(found_pair_set)={len(found_pair_set)} pairs."
        )

        return found_pair_set


class ANNLinkageIndex:
    def __init__(self, embedding_size):
        self.left_index = ANNEntityIndex(embedding_size)
        self.right_index = ANNEntityIndex(embedding_size)

    def insert_vector_dict(self, left_vector_dict, right_vector_dict):
        self.left_index.insert_vector_dict(vector_dict=left_vector_dict)
        self.right_index.insert_vector_dict(vector_dict=right_vector_dict)

    def build(self, index_build_kwargs=None):
        self.left_index.build(index_build_kwargs=index_build_kwargs)
        self.right_index.build(index_build_kwargs=index_build_kwargs)

    def search_pairs(self, k, sim_threshold, left_vector_dict, right_vector_dict, left_source, index_search_kwargs=None):
        if not self.left_index.is_built or not self.right_index.is_built:
            raise ValueError("Please call build first")
        if sim_threshold > 1 or sim_threshold < 0:
            raise ValueError(f"sim_threshold={sim_threshold} must be <= 1 and >= 0")

        index_search_kwargs = build_index_search_kwargs(index_search_kwargs)
        distance_threshold = 1 - sim_threshold
        all_pair_set = set()

        for dataset_name, index, vector_dict, other_index in [
            (left_source, self.left_index, right_vector_dict, self.right_index),
            (None, self.right_index, left_vector_dict, self.left_index),
        ]:
            logger.debug(f"Searching on approx_knn_index of dataset_name={dataset_name}...")

            # Using cKDTree for approximate nearest neighbor search
            tree = cKDTree(np.array(list(vector_dict.values())))
            distances, indices = tree.query(list(vector_dict.values()), k=k)

            for i in range(len(vector_dict)):
                for j in range(k):
                    if distances[i, j] <= distance_threshold:
                        if dataset_name and dataset_name == left_source:
                            left_id = list(vector_dict.keys())[i]
                            right_id = list(other_index.vector_idx_to_id.values())[indices[i, j]]
                        else:
                            left_id = list(other_index.vector_idx_to_id.values())[i]
                            right_id = list(vector_dict.keys())[indices[i, j]]
                        pair = (left_id, right_id)
                        all_pair_set.add(pair)

            logger.debug(f"Filling all_pair_set with dataset_name={dataset_name} done.")

        logger.debug(
            "All searches done, all_pair_set filled. "
            f"Found len(all_pair_set)={len(all_pair_set)} pairs."
        )

        return all_pair_set
