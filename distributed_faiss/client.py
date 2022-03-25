#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import random
import time
from multiprocessing.dummy import Pool as ThreadPool
from typing import List, Tuple

import faiss
import numpy as np

from distributed_faiss.index_cfg import IndexCfg
from . import rpc
from .index_state import IndexState

logger = logging.getLogger()

from typing import Optional


class ResultHeap:
    """Accumulate query results from a sliced dataset. The final result will
    be in self.D, self.I. Using faiss.float_maxheap_array_t()"""

    def __init__(self, nq, k):
        "nq: number of query vectors, k: number of results per query"
        self.I = np.zeros((nq, k), dtype="int64")
        self.D = np.zeros((nq, k), dtype="float32")
        self.nq, self.k = nq, k
        heaps = faiss.float_maxheap_array_t()

        heaps.k = k
        heaps.nh = nq
        heaps.val = faiss.swig_ptr(self.D)
        heaps.ids = faiss.swig_ptr(self.I)
        heaps.heapify()
        self.heaps = heaps

    def add_result(self, D, I):
        """D, I do not need to be in a particular order (heap or sorted)"""
        assert D.shape == (self.nq, self.k)
        assert I.shape == (self.nq, self.k)
        self.heaps.addn_with_ids(self.k, faiss.swig_ptr(D), faiss.swig_ptr(I), self.k)

    def finalize(self):
        self.heaps.reorder()


class IndexClient:
    """
    Manages a set of distance sub-indexes. The sub_indexes search a
    subset of the inverted lists. Searches are merged afterwards
    """

    def __init__(self, server_list_path: str, cfg_path: Optional[str] = None):
        """connect to a series of (host, port) pairs"""
        machine_ports = IndexClient.read_server_list(server_list_path)
        self.sub_indexes = IndexClient.setup_connection(machine_ports)
        self.num_indexes = len(self.sub_indexes)

        # index_rank_to_id is a map between logical node id and sub-indexes list id
        # it is useful for debuggind
        # might be useful for a better load balancing upon index restart with failed/lagged behind nodes
        index_ranks = [idx.get_rank() for idx in self.sub_indexes]
        logger.info("index_ranks %s", index_ranks)
        self.index_rank_to_id = {
            index_rank: index_id for index_id, index_rank in enumerate(index_ranks)
        }
        logger.info("index_rank_to_id %s", self.index_rank_to_id)

        # pool of threads. Each thread manages one sub-index.
        self.pool = ThreadPool(self.num_indexes)
        self.verbose = False
        self.cur_server_ids = {}

        random.seed(time.time())
        self.cfg = IndexCfg.from_json(cfg_path) if cfg_path is not None else None

    @staticmethod
    def read_server_list(
        server_list_path,
        initial_timeout=0.1,
        backoff_factor=1.5,
        total_max_timeout=7200,
    ) -> List[Tuple[str, int]]:
        time_waited = 0
        while True:
            with open(server_list_path) as f:
                res = []  # list of [(hostname, port), ...]
                for idx, line in enumerate(f):
                    if idx == 0:
                        num_servers = int(line)
                        continue
                    res.append((str(line.split(",")[0]), int(line.split(",")[1])))

            msg = f"{num_servers} != {len(res)} in server list {server_list_path}."
            if num_servers != len(res):
                print(
                    msg
                    + f" Waiting {round(initial_timeout * 100) / 100} seconds for servers to load..."
                )
                time.sleep(initial_timeout)

            if time_waited + initial_timeout >= total_max_timeout:
                break
            time_waited += initial_timeout
            initial_timeout *= backoff_factor

        assert num_servers == len(res), (
            msg + f" Timed out after waiting {round(time_waited * 100) / 100} seconds"
        )
        return res

    @staticmethod
    def setup_connection(machine_ports) -> List[rpc.Client]:
        sub_indexes = []
        for idx, machine_port in enumerate(machine_ports):
            sub_indexes.append(rpc.Client(idx, machine_port[0], machine_port[1], False))
        return sub_indexes

    def drop_index(self, index_id: str):
        self.pool.map(lambda idx: idx.drop_index(index_id), self.sub_indexes)

    def save_index(self, index_id: str):
        self.pool.map(lambda idx: idx.save_index(index_id), self.sub_indexes)

    def load_index(
        self,
        index_id: str,
        cfg: Optional[IndexCfg] = None,
        force_reload: bool = True,
    ) -> bool:
        def setup_cfg(cfg: Optional[IndexCfg]):
            if cfg is None:
                config_paths = self.pool.map(
                    lambda idx: idx.get_config_path(index_id), self.sub_indexes
                )
                if len(config_paths) > 0 and os.path.isfile(config_paths[0]):
                    cfg = IndexCfg.from_json(config_paths[0])
                else:
                    cfg = IndexCfg()
            return cfg

        if force_reload:
            logger.info("Forced index reload")
            self.pool.map(lambda idx: idx.drop_index(index_id), self.sub_indexes)
        all_loaded = self.pool.map(lambda idx: idx.load_index(index_id, cfg), self.sub_indexes)
        # TODO: remove cfg as client instance attribute, it should be specific index only cfg
        self.cfg = setup_cfg(cfg)
        logger.info(f"Index cfg {self.cfg}")

        if all(all_loaded):
            logger.info(f"The index {index_id} is loaded.")
            return True
        if any(all_loaded):
            logger.warning(f"Some server nodes can't load index: {all_loaded}")
        return False

    def create_index(self, index_id: str, cfg: Optional[IndexCfg] = None):
        if cfg is not None:
            self.cfg = cfg
        if self.cfg is None:
            self.cfg = IndexCfg()
        return self.pool.map(lambda idx: idx.create_index(index_id, self.cfg), self.sub_indexes)

    def add_index_data(
        self,
        index_id: str,
        embeddings: np.array,
        metadata: Optional[List[object]] = None,
        train_async_if_triggered: bool = True,
    ) -> None:
        """
        Randomly select index of server to write data to first time
        writing to it. later use round-robin to select index server
        to balance the load.
        """
        if index_id not in self.cur_server_ids:
            self.cur_server_ids[index_id] = random.randint(0, self.num_indexes - 1)
        cur_server_id = self.cur_server_ids[index_id]
        self.sub_indexes[cur_server_id].add_index_data(
            index_id, embeddings, metadata, train_async_if_triggered
        )
        self.cur_server_ids[index_id] = (self.cur_server_ids[index_id] + 1) % self.num_indexes

    def sync_train(self, index_id: str) -> None:
        self.pool.map(lambda idx: idx.sync_train(index_id), self.sub_indexes)

    def async_train(self, index_id: str):
        self.pool.map(lambda idx: idx.sync_train(index_id), self.sub_indexes)

    def search(
        self, query, topk: int, index_id: str, return_embeddings: bool = False
    ) -> Tuple[np.ndarray, List]:
        """Call idx.search on each rpc client and then aggregates results using heap."""

        q_size = query.shape[0]
        maximize_metric: bool = self.cfg.metric == "dot"
        results = self.pool.imap(
            lambda idx: idx.search(index_id, query, topk, return_embeddings), self.sub_indexes
        )
        return self._aggregate_results(results, topk, q_size, maximize_metric, return_embeddings)

    # TODO: make filter a generic custom function that takes meta and sample's score and return bool value
    def search_with_filter(
        self,
        query: np.array,
        top_k: int,
        index_id: str,
        filter_pos: int = -1,
        filter_value=None,
    ) -> Tuple[np.array, List[List[object]]]:

        # TODO: get from cfg ?
        filter_top_factor = 3  # search for x times more results
        actual_top_k = filter_top_factor * top_k if filter_pos >= 0 else top_k
        (scores, meta) = self.search(query, actual_top_k, index_id)

        if filter_pos < 0:
            return scores, meta

        def _do_filter(scores: np.array, results_meta: List[List[Tuple]]):
            re_query_ids = []
            new_results = []
            new_scores = []

            for i, meta_list in enumerate(results_meta):
                sample_filtered_meta = []
                sample_filtered_scores = []
                for j, meta in enumerate(meta_list):
                    if not meta:
                        logger.warning("No meta for j=%d, score=%s", j, scores[i, j])
                        continue
                    if len(meta) > filter_pos and meta[filter_pos] != filter_value:
                        sample_filtered_meta.append(meta)
                        sample_filtered_scores.append(scores[i, j])
                    if len(sample_filtered_meta) >= top_k:
                        break

                if len(sample_filtered_meta) < top_k:
                    re_query_ids.append(i)

                new_results.append(sample_filtered_meta)
                new_scores.append(
                    np.concatenate([s.reshape(-1, 1) for s in sample_filtered_scores], axis=0)
                )
            # TODO: filtered scores list may be of different lengths so we can't concatenate them here generally
            # new_scores = np.concatenate(new_scores, axis=0)
            return new_scores, new_results, re_query_ids

        new_scores, new_results_meta, re_query_ids = _do_filter(scores, meta)
        logger.info(f"{len(re_query_ids)} samples return less than {top_k} results after filtering")
        # TODO: search again with larger top_k for queries in re_query_ids

        return new_scores, new_results_meta

    @staticmethod
    def _aggregate_results(
        results: List[Tuple],
        topk: int,
        q_size: int,
        maximize_metric: bool,
        return_embeddings: bool,
    ):
        meta = []
        embs = []
        cur_idx = 0

        def to_matrix(l, n):
            return [l[i : i + n] for i in range(0, len(l), n)]

        res_heap = ResultHeap(q_size, topk)
        for DI, MetaI, e in results:
            # Two hacks:
            # 1) for DOT, search for -D because we want to find max dot product, not min
            # 2) create new indexed to later map metadata to the best indexes
            merged_meta = list(itertools.chain(*MetaI))
            meta.extend(merged_meta)
            if return_embeddings:
                merged_embs = list(itertools.chain(*e))
                embs.extend(merged_embs)
            Ii = np.reshape(np.arange(cur_idx, cur_idx + q_size * topk), (q_size, topk))
            if maximize_metric:
                res_heap.add_result(-DI, Ii)
            else:
                res_heap.add_result(DI, Ii)
            cur_idx += q_size * topk
        res_heap.finalize()
        ids = np.reshape(res_heap.I, (-1,)).tolist()
        selected_meta = [meta[i] for i in ids]
        if return_embeddings:
            selected_embs = [embs[i] for i in ids]

        return (
            (
                res_heap.D,
                to_matrix(selected_meta, res_heap.D.shape[1]),
                to_matrix(selected_embs, res_heap.D.shape[1]),
            )
            if return_embeddings
            else (res_heap.D, to_matrix(selected_meta, res_heap.D.shape[1]))
        )

    def get_centroids(self, index_id: str):
        return self.pool.map(lambda idx: idx.get_centroids(index_id), self.sub_indexes)

    def set_nprobe(self, index_id: str, nprobe: int):
        return self.pool.map(lambda idx: idx.set_nprobe(index_id, nprobe), self.sub_indexes)

    def get_state(self, index_id: str) -> IndexState:
        states = self.pool.map(lambda idx: idx.get_state(index_id), self.sub_indexes)
        logging.info("Index nodes states %s", states)
        return IndexState.get_aggregated_states(states)

    def add_buffer_to_index(
        self,
        index_id: str,
    ):
        self.pool.map(lambda idx: idx.add_buffer_to_index(index_id), self.sub_indexes)

    def get_ntotal(self, index_id: str) -> None:
        return sum(self.pool.map(lambda idx: idx.get_ntotal(index_id), self.sub_indexes))

    def get_ids(self, index_id: str) -> set:
        id_set_list = self.pool.map(lambda idx: idx.get_ids(index_id), self.sub_indexes)
        for i in range(len(id_set_list)):
            logging.info("ids i=%s len=%s", i, len(id_set_list[i]))
        return set().union(*id_set_list)

    def set_omp_num_threads(self, num_threads: int) -> None:
        self.pool.map(lambda idx: idx.set_omp_num_threads(num_threads), self.sub_indexes)

    def close(self):
        [index_conn.close() for index_conn in self.sub_indexes]

    def get_num_servers(self):
        return self.num_indexes
