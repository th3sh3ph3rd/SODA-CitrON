# Copyright 2026 AIT Austrian Institute of Technology GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABCMeta

from ulid import ULID
import numpy as np
from scipy.special import logit, expit
from river import cluster 

class SodaCitronMicroCluster(metaclass=ABCMeta):
    """SODA-CitrON Micro-cluster class"""

    def __init__(self, x=None, conf=None, cov=None, last_update=None, weight=None, id=None, members=None):
        self.center = x
        self.lo_conf = logit(conf)
        self.cov = cov 
        inv_cov = np.linalg.inv(cov)
        self.Y = conf*inv_cov # information matrix
        self.y = conf*(inv_cov @ x) # information vector
        self.last_update = last_update
        self.weight = weight
        self.id = id
        self.members = members

    def merge(self, cluster: 'SodaCitronMicroCluster'):
        # merge information matrix and vector
        self.Y = self.Y + cluster.Y
        self.y = self.y + cluster.y
        # compute new cluster covariance, center and confidence
        self.cov = np.linalg.inv(self.Y)
        self.center = self.cov @ self.y
        self.lo_conf += cluster.lo_conf
        # take ID of stronger cluster
        if self.weight < cluster.weight:
            self.id = cluster.id
        # merge weights
        self.weight += cluster.weight
        self.members += cluster.members
 
    @property
    def conf(self) -> float:
        return expit(self.lo_conf)

class SodaCitron(cluster.DBSTREAM):
    def __init__(
        self,
        clustering_threshold: float = 1.0,
        intersection_factor: float = 0.3,
        minimum_weight: float = 1.0,
    ):
        super().__init__(clustering_threshold=clustering_threshold,
                         fading_factor=0.0,
                         cleanup_interval=1,
                         intersection_factor=intersection_factor,
                         minimum_weight=minimum_weight)
        self._clusters: dict[int, SodaCitronMicroCluster] = {}
        self._micro_clusters: dict[int, SodaCitronMicroCluster] = {}
    
    @staticmethod
    def _distance(point_a, point_b): 
        return np.linalg.norm(point_a - point_b)

    def _update(self, x: np.ndarray, conf: float, cov: np.ndarray, w: float, sample_id: ULID): 
        neighbor_clusters = self._find_fixed_radius_nn(x) # Note: to achieve log-linear complexity, implement efficiently, e.g. with r-tree

        if len(neighbor_clusters) < 1:
            # create new micro cluster
            cluster_id = ULID()
            if len(self._micro_clusters) > 0:
                self._micro_clusters[max(self._micro_clusters.keys()) + 1] = SodaCitronMicroCluster(
                    x=x, conf=conf, cov=cov, last_update=self._time_stamp, weight=w, id=cluster_id, members=[sample_id]
                )
            else:
                self._micro_clusters[0] = SodaCitronMicroCluster(
                    x=x, conf=conf, cov=cov, last_update=self._time_stamp, weight=w, id=cluster_id, members=[sample_id]
                )
        else:
            # update existing micro clusters
            current_centers = {}
            current_Y = {}
            current_y = {}
            current_covs = {}
            current_confs = {}
            for i in neighbor_clusters.keys():
                current_centers[i] = self._micro_clusters[i].center
                current_Y[i] = self._micro_clusters[i].Y
                current_y[i] = self._micro_clusters[i].y
                current_covs[i] = self._micro_clusters[i].cov
                current_confs[i] = self._micro_clusters[i].lo_conf
                self._micro_clusters[i].weight = (
                    self._micro_clusters[i].weight
                    * 2
                    ** (
                        -self.fading_factor
                        * (self._time_stamp - self._micro_clusters[i].last_update)
                    )
                    + w
                )

                # Update the information matrix and vector (i) with overlapping keys (j)
                inv_cov = np.linalg.inv(cov)
                self._micro_clusters[i].Y = self._micro_clusters[i].Y + inv_cov
                self._micro_clusters[i].y = self._micro_clusters[i].y + (inv_cov @ x)
                
                # compute new cluster covariance, center and log-odds confidence
                self._micro_clusters[i].cov = (
                    np.linalg.inv(self._micro_clusters[i].Y)
                )
                self._micro_clusters[i].center = (
                     self._micro_clusters[i].cov @ self._micro_clusters[i].y
                )
                self._micro_clusters[i].lo_conf = (
                    self._micro_clusters[i].lo_conf + logit(conf)
                )

                # add to cluster members
                self._micro_clusters[i].members.append(sample_id)
                
                self._micro_clusters[i].last_update = self._time_stamp

                # update shared density
                for j in neighbor_clusters.keys():
                    if j > i:
                        try:
                            self.s[i][j] = (
                                self.s[i][j]
                                * 2 ** (-self.fading_factor * (self._time_stamp - self.s_t[i][j]))
                                + w
                            )
                            self.s_t[i][j] = self._time_stamp
                        except KeyError:
                            try:
                                self.s[i][j] = w
                                self.s_t[i][j] = self._time_stamp
                            except KeyError:
                                self.s[i] = {j: 1}
                                self.s_t[i] = {j: self._time_stamp}

            # prevent collapsing clusters
            for i in neighbor_clusters.keys():
                for j in neighbor_clusters.keys():
                    if j > i:
                        if (
                            self._distance(
                                self._micro_clusters[i].center,
                                self._micro_clusters[j].center,
                            )
                            < self.clustering_threshold
                        ):
                            # revert states of mc_i and mc_j to previous values
                            self._micro_clusters[i].center = current_centers[i]
                            self._micro_clusters[j].center = current_centers[j]
                            self._micro_clusters[i].Y = current_Y[i]
                            self._micro_clusters[j].Y = current_Y[j]
                            self._micro_clusters[i].y = current_y[i]
                            self._micro_clusters[j].y = current_y[j]
                            self._micro_clusters[i].cov = current_covs[i]
                            self._micro_clusters[j].cov = current_covs[j]
                            self._micro_clusters[i].lo_conf = current_confs[i]
                            self._micro_clusters[j].lo_conf = current_confs[j]

        self._time_stamp += 1
    
    def _generate_clusters_from_labels(self, cluster_labels):
        # Group micro clusters by label in one pass, preserving the input ordering.
        by_label: dict[int, list[SodaCitronMicroCluster]] = {}
        for index, label in cluster_labels.items():
            if label is None:
                continue
            by_label.setdefault(label, []).append(self._micro_clusters[index])

        if not by_label:
            return 0, {}

        clusters: dict[int, SodaCitronMicroCluster] = {}
        for label in sorted(by_label):
            members = by_label[label]
            first = members[0]
            # Build the macro cluster as a fresh SodaCitronMicroCluster so the original
            # micro cluster is not mutated by the subsequent merge calls.
            macro_cluster = SodaCitronMicroCluster(
                x=first.center,
                conf=first.conf,
                cov=first.cov,
                last_update=first.last_update,
                weight=first.weight,
                id=first.id,
                members=first.members,
            )
            for member in members[1:]:
                macro_cluster.merge(member)
            clusters[label] = macro_cluster

        return len(clusters), clusters

    def learn_one(self, x: np.ndarray, conf: float, cov: np.ndarray, w: float=1.0, id: ULID=None):
        self._update(x, conf, cov, w, id)

        if self.fading_factor > 0 and self._time_stamp % self.cleanup_interval == 0:
            self._cleanup()

        self.clustering_is_up_to_date = False

    def predict_one(self, x: np.ndarray, w: float=None):
        raise NotImplementedError

    @property
    def clusters(self) -> dict[int, SodaCitronMicroCluster]:
        self._recluster()
        return self._clusters

    @property
    def micro_clusters(self) -> dict[int, SodaCitronMicroCluster]:
        return self._micro_clusters
 