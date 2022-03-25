# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import json


class IndexCfg:
    def __init__(
        self,
        index_builder_type: str = None,
        faiss_factory: str = None,
        dim: int = 768,
        train_num: int = 0,
        train_ratio: int = 1.0,
        centroids: int = 0,
        metric: str = "dot",
        nprobe: int = 1,
        infer_centroids=False,
        buffer_bsz: int = 50000,
        save_interval_sec: int = -1,
        index_storage_dir: str = None,
        custom_meta_id_idx: int = 0,
        **kwargs,
    ):
        self.index_builder_type = index_builder_type
        self.faiss_factory = faiss_factory
        self.dim = int(dim)
        self.train_num = train_num
        self.train_ratio = train_ratio
        self.centroids = centroids
        self.metric = metric
        self.nprobe = nprobe
        self.infer_centroids = infer_centroids
        self.buffer_bsz = buffer_bsz
        self.save_interval_sec = save_interval_sec
        self.index_storage_dir = index_storage_dir
        self.custom_meta_id_idx = custom_meta_id_idx
        self.extra = kwargs

    def get_metric(self):
        metric = self.metric
        if metric == "dot":
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        elif metric == "l2":
            faiss_metric = faiss.METRIC_L2
        else:
            raise RuntimeError("Only dot and l2 metrics are supported.")
        return faiss_metric

    @classmethod
    def from_json(cls, json_path):
        with open(json_path, "r") as f:
            kwargs = json.load(f)
        return cls(**kwargs)

    def to_json_string(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def __repr__(self) -> str:
        return f"<IndexCFG: {self.__dict__}>"
