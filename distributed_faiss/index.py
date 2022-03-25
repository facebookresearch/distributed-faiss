# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import _thread
import logging
import math
import os
import pickle
import threading
import time
from typing import Tuple, Optional, List, Union

import faiss
import numpy as np

from distributed_faiss.index_cfg import IndexCfg
from distributed_faiss.index_state import IndexState

logger = logging.getLogger()


def get_quantizer(cfg: IndexCfg):
    metric = cfg.get_metric()
    if metric == faiss.METRIC_INNER_PRODUCT:
        quantizer = faiss.IndexFlatIP(cfg.dim)
    elif metric == faiss.METRIC_L2:
        quantizer = faiss.IndexFlatL2(cfg.dim)
    else:
        raise RuntimeError(f"Metric={metric} is not supported")
    return quantizer


def init_faiss_ivf_simple(cfg: IndexCfg):
    metric = cfg.get_metric()
    index = faiss.IndexIVFFlat(get_quantizer(cfg), cfg.dim, cfg.centroids, metric)
    index.nprobe = cfg.nprobe
    return index


def init_faiss_knnlm(cfg: IndexCfg):
    code_size = cfg.extra.get("code_size", 64)
    bits_per_vector = cfg.extra.get("bits_per_vector", 8)
    index = faiss.IndexIVFPQ(get_quantizer(cfg), cfg.dim, cfg.centroids, code_size, bits_per_vector)
    cfg.nprobe = index.nprobe
    return index


def init_faiss_hnswsq(cfg: IndexCfg):
    assert cfg.get_metric() == faiss.METRIC_L2, "hnsw is supposed to work with L2 sim space"
    index = faiss.IndexHNSWSQ(
        cfg.dim,
        faiss.ScalarQuantizer.QT_8bit,
        cfg.extra.get("store_n", 128),
    )
    index.hnsw.efSearch = cfg.nprobe
    index.hnsw.efConstruction = cfg.extra.get("ef_construction", 100)
    return index


def init_faiss_ivf_scalar_qr(cfg: IndexCfg):
    index = faiss.IndexIVFScalarQuantizer(
        get_quantizer(cfg), cfg.dim, cfg.centroids, faiss.ScalarQuantizer.QT_fp16
    )
    index.nprobe = cfg.nprobe
    return index


def init_faiss_ivf_gpu(cfg: IndexCfg):
    # TODO: implement the logic of selecting specific gpus from cfg
    index = faiss.index_factory(cfg.dim, cfg.faiss_factory)
    metric = cfg.get_metric()
    if metric == faiss.METRIC_INNER_PRODUCT:
        quantizer = faiss.IndexFlatIP(cfg.dim)
    elif metric == faiss.METRIC_L2:
        quantizer = faiss.IndexFlatL2(cfg.dim)
    else:
        raise RuntimeError(f"Metric={metric} is not supported for ivf_gpu factory")

    index_ivf = faiss.extract_index_ivf(index)
    clustering_index = faiss.index_cpu_to_all_gpus(quantizer)
    index_ivf.clustering_index = clustering_index
    index.nprobe = cfg.nprobe
    return index


def init_flat_index(cfg: IndexCfg):
    return get_quantizer(cfg)


faiss_special_index_factories = {
    "flat": lambda cfg: faiss.IndexFlatIP(cfg.dim),
    "ivf_simple": init_faiss_ivf_simple,
    "knnlm": init_faiss_knnlm,
    "hnswsq": init_faiss_hnswsq,
    "ivfsq": init_faiss_ivf_scalar_qr,
    "ivf_gpu": init_faiss_ivf_gpu,
}


def get_index_files(index_storage_dir: str) -> Tuple[str, str, str, str]:
    index_file = os.path.join(index_storage_dir, "index.faiss")
    meta_file = os.path.join(index_storage_dir, "meta.pkl")
    buffer_file = os.path.join(index_storage_dir, "buffer.pkl")
    cfg_file = os.path.join(index_storage_dir, "cfg.json")
    return index_file, meta_file, buffer_file, cfg_file


class Index:
    def __init__(self, cfg: IndexCfg):
        self.cfg = cfg
        self.embeddings_buffer = []
        self.total_data = 0
        self.id_to_metadata = []
        self.buffer_lock = threading.Lock()
        self.index_lock = threading.Lock()
        self.state = IndexState.NOT_TRAINED
        self.faiss_index = None

        self.index_save_time = time.time()
        self.index_saved_size = 0

        if cfg.save_interval_sec > 0:
            self._run_save_watcher()

    def drop_index(self):
        with self.buffer_lock:
            self.embeddings_buffer = []
            self.total_data = 0
            self.id_to_metadata = []

        with self.index_lock:
            self.faiss_index = None
            self.state = IndexState.NOT_TRAINED

    def add_batch(
        self,
        embeddings: np.array,
        metadata: Optional[List[object]],
        train_async_if_triggered: bool = True,
    ):
        embeddings_num = embeddings.shape[0]
        if not metadata:
            metadata = [None] * embeddings_num
        if embeddings_num != len(metadata):
            raise RuntimeError("metadata length should match the batch size of the embeddings")

        # TODO: check why HNSW-SQ doesn't work without it
        embeddings = embeddings.astype(np.float32)

        with self.buffer_lock:
            self.embeddings_buffer.append(embeddings)
            self.id_to_metadata.extend(metadata)
            self.total_data += embeddings_num
            total_data = self.total_data

        logger.info(f"The size of the buffer is {total_data}")

        state = self.get_state()
        if state == IndexState.TRAINED and total_data >= 0:
            self.add_buffer_to_index()  # TODO: revise to avoid double state check here & inside add_buffer_to_index
        elif state == IndexState.NOT_TRAINED and 0 < self.cfg.train_num <= total_data:
            # trigger training
            logger.info(f"The size of the buffer is {total_data}, can start index training.")
            if state in [IndexState.TRAINING, IndexState.TRAINED, IndexState.ADD]:
                logger.info(f"Index state: {state}, skip training start")
                return

            if train_async_if_triggered:
                logger.info(f"Starting index training in a new thread")
                _thread.start_new_thread(self.train, ())
            else:
                logger.info(f"Starting index training sync")
                self.train()

    def get_idx_data_num(self) -> Tuple[int, int]:
        with self.buffer_lock:
            buf_total = self.total_data
        index_total = 0
        with self.index_lock:
            if self.faiss_index:
                index_total = self.faiss_index.ntotal
        return buf_total, index_total

    def train(self) -> None:
        with self.index_lock:
            if self.state in [IndexState.TRAINING, IndexState.TRAINED, IndexState.ADD]:
                return
            self.state = IndexState.TRAINING
        cfg = self.cfg

        with self.buffer_lock:
            embeddings = self.embeddings_buffer
            dim = cfg.dim
            if dim == 0:  # guess
                dim = embeddings[0].shape[1]
                cfg.dim = dim

            # get only part of the buffer sufficient for training
            if cfg.train_num > 0:
                train_num = cfg.train_num
            elif cfg.train_ratio >= 1.0:
                train_num = self.total_data
            else:
                train_num = int(cfg.train_ratio * self.total_data)
            all_data_as_np_array = np.concatenate(embeddings, axis=0)

        train_data = all_data_as_np_array[:train_num]
        np.random.shuffle(train_data)
        total_data_size = all_data_as_np_array.shape[0]
        index = self._init_faiss_index(total_data_size)

        logging.info(f"Created a faiss index of type {type(index)}")
        logger.info(f"Training index with array shaped {train_data.shape}")
        index.train(train_data)
        logger.info(f"Index trained")

        with self.index_lock:
            self.faiss_index = index
            self.state = IndexState.TRAINED
        self.add_buffer_to_index()

    def add_buffer_to_index(self) -> None:
        add_to_index = False
        with self.index_lock:
            if self.state == IndexState.TRAINED:
                add_to_index = True
                self.state = IndexState.ADD
                logging.info("Index is trained, adding data from buffer")
            else:
                logging.info("Index add is already in progress")

        if add_to_index:
            # new vectors addition happens in a separate thread to make this method non blocking and letting next
            # client's batches go to different server nodes without waiting for add_buffer_to_index completion
            _thread.start_new_thread(self._add_buffer_to_idx, ())

    # TODO: overload to get faiss indexes back, not metadata
    def search(
        self, query_batch: np.array, top_k: int = 100, return_embeddings: bool = False
    ) -> Tuple[np.array, List[List[object]], Optional[np.array]]:
        logger.info(f"Searching index, queries {query_batch.shape}")

        with self.index_lock:
            if self.state != IndexState.TRAINED:
                raise RuntimeError(f"Server index is not trained. state: {self.state}")
            # Locking index for search operations
            # from https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls#performance-of-search:
            # "However it is very inefficient to call batches of queries from multiple threads, this will spawn more threads than cores."
            # TODO: avoid locking for single query calls (or small batches?)

            if return_embeddings:
                scores, indexes, embs = self.faiss_index.search_and_reconstruct(query_batch, top_k)
            else:
                scores, indexes = self.faiss_index.search(query_batch, top_k)
                embs = None

        top_k, n = indexes.shape
        with self.buffer_lock:
            results_meta = [
                [
                    self.id_to_metadata[indexes[i, j]] if indexes[i, j] != -1 else None
                    for j in range(n)
                ]
                for i in range(top_k)
            ]
        logger.info(f"Search completed, results_meta len: {len(results_meta)}")
        return scores, results_meta, embs

    def save(self) -> bool:
        state = self.get_state()
        if state == IndexState.TRAINED:
            return self._maybe_save(ignore_time=True)
        elif state == IndexState.ADD:
            # reset index_save_time to trigger save upon the completion of add batch
            logger.info("Index is in ADD state, clearing index_save_time")
            self.index_save_time = 0
        else:
            logger.info("Index is not trained, skip saving")
            return False

    @classmethod
    def from_storage_dir(
        cls, index_storage_dir: str, cfg: IndexCfg = None, ignore_buffer: bool = True
    ) -> Union[None, object]:
        logger.info("Deserializing index from %s", index_storage_dir)
        index_file, meta_file, buffer_file, cfg_file = get_index_files(index_storage_dir)

        if not os.path.exists(index_file):
            logger.info(f"No index found at {index_file}")
            return None
        else:
            logger.info(f"index_file={index_file} meta_file={meta_file} buffer_file={buffer_file}")

        index = faiss.read_index(index_file)
        logger.info("Loaded index of type %s and size %d", type(index), index.ntotal)

        if os.path.exists(meta_file):
            logger.info("Loading meta file from %s", meta_file)
            with open(meta_file, "rb") as reader:
                meta = pickle.load(reader)
            logger.info("metadata deserialized, size: %d", len(meta))
            assert (
                len(meta) >= index.ntotal
            ), "Deserialized meta list should be at least of faiss index size"
        else:
            raise RuntimeError("no meta file found. Can't use index.")

        buffer = []
        if (not ignore_buffer) and buffer_file and os.path.exists(buffer_file):
            logger.info("Loading buffer from %s", buffer_file)
            with open(buffer_file, "rb") as reader:
                buffer = pickle.load(reader)

        if cfg is None:
            if os.path.isfile(cfg_file):
                cfg = IndexCfg.from_json(cfg_file)
            else:
                cfg = IndexCfg()

        logger.info(f"Meta length {len(meta)}")
        logger.info(f"Loaded index size {index.ntotal}")
        buffer_size = sum(v.shape[0] for v in buffer)
        logger.info(f"Buffer size {buffer_size}")

        result = cls(cfg)
        result.faiss_index = index
        result.state = IndexState.TRAINED
        result.upd_cfg(cfg)

        if len(meta) == index.ntotal + buffer_size:
            result.id_to_metadata = meta
            result.embeddings_buffer = buffer
            if buffer_size > 0:
                result.add_buffer_to_index()
        else:
            logger.warning(
                f"Metadata size doesn't match combined index+buffer size ({index.ntotal + buffer_size}): "
                f"ignoring buffer, reducing metadata to index size"
            )
            result.id_to_metadata = meta[: index.ntotal]
        return result

    def get_centroids(self):
        with self.index_lock:
            if self.state != IndexState.TRAINED:
                raise RuntimeError("Server index is not trained")
            return self.faiss_index.quantizer.reconstruct_n(0, self.faiss_index.nlist)

    def set_nprobe(self, nprobe: int):
        self.cfg.nprobe = nprobe
        with self.index_lock:
            if self.faiss_index:
                self.faiss_index.nprobe = nprobe

    def get_state(self):
        with self.index_lock:
            return self.state

    def get_ids(self):
        id_idx = self.cfg.custom_meta_id_idx
        r = {meta[id_idx] for meta in self.id_to_metadata if meta}
        logger.info(f"Metadata Id set size {len(r)}")
        if self.faiss_index:
            with self.buffer_lock:
                with self.index_lock:
                    buffer_size = sum(v.shape[0] for v in self.embeddings_buffer)
                    if len(r) != self.faiss_index.ntotal + buffer_size:
                        logger.warning(
                            f"id set size mismatch, index+buffer={self.faiss_index.ntotal + buffer_size}"
                        )
        return r

    def upd_cfg(self, cfg: IndexCfg):
        self.cfg = cfg
        self._override_nprobe(cfg)

    def _init_faiss_index(self, total_data_size: int):
        cfg = self.cfg
        logger.info(f"Data size for indexing {total_data_size}")
        if cfg.index_builder_type:
            index = faiss_special_index_factories[cfg.index_builder_type](cfg)
        elif cfg.faiss_factory:
            cfg.centroids = int(cfg.centroids)
            if cfg.centroids == 0 or cfg.infer_centroids:
                cfg.centroids = self.infer_n_centroids(total_data_size)
                logger.info(f"Inferring cfg.centroids={cfg.centroids}")

            if "{centroids}" in cfg.faiss_factory:
                index_cfg_str = cfg.faiss_factory.format(centroids=cfg.centroids)
            else:
                index_cfg_str = cfg.faiss_factory
            logger.info(f"Using index factory: {index_cfg_str}")
            index = faiss.index_factory(cfg.dim, index_cfg_str, cfg.get_metric())
        else:
            raise RuntimeError(
                "Either faiss_factory or valid index_builder_type should be specified to initialize index"
            )
        return index

    def _add_buffer_to_idx(self):
        while True:
            bsz = self.cfg.buffer_bsz
            embeddings_to_add = []
            embeddings_to_add_total = 0
            with self.buffer_lock:
                logger.info(f"Current buffer size: {self.total_data}")
                for i, e in enumerate(self.embeddings_buffer):
                    embeddings_to_add.append(e)
                    embeddings_to_add_total += e.shape[0]
                    if embeddings_to_add_total >= bsz:
                        break

            if embeddings_to_add_total == 0:
                logger.info(f"Buffer is empty")
                break
            else:
                self.embeddings_buffer = self.embeddings_buffer[len(embeddings_to_add) :]
                self.total_data -= embeddings_to_add_total
                add_data_as_np = np.concatenate(embeddings_to_add, axis=0)
                logger.info(f"Adding {add_data_as_np.shape[0]} to the trained index.")
                start_time = time.time()
                self.faiss_index.add(add_data_as_np)
                logger.info(
                    f"Add completed in {time.time() - start_time}. ntotal={self.faiss_index.ntotal}"
                )
                saved = self._maybe_save(ignore_time=False)
                logger.info(f"Index saved ={saved}")

        with self.index_lock:
            self.state = IndexState.TRAINED

    def _maybe_save(self, ignore_time: bool = False) -> bool:
        if not ignore_time:
            if self.cfg.save_interval_sec <= 0:  # autosave disabled
                return False
            if time.time() - self.index_save_time < self.cfg.save_interval_sec:
                logger.info(f"Not enough time spent since the latest index save")
                return False

        # TODO:
        #  - Use tmp files for initial save and then convert tmp files to permanent once they are ready
        #  - the case when index is not trained/empty and data are only in the buffer
        #  - part of the buffer may be lost if the process crashes during index ADD

        with self.buffer_lock, self.index_lock:
            if self.faiss_index.ntotal == self.index_saved_size:
                logger.info(f"Index hasn't changed since the latest save")
                return False

            index_storage_dir = self.cfg.index_storage_dir

            logger.info(f"Serializing index to {index_storage_dir}")
            index_file, meta_file, buffer_file, cfg_file = get_index_files(index_storage_dir)
            if os.path.exists(index_file):
                logger.info("Index file already exists. overwriting save")

            faiss.write_index(self.faiss_index, index_file)
            with open(meta_file, mode="wb") as f:
                pickle.dump(self.id_to_metadata, f)
            logger.info(f"Saved index & meta to: {index_file} | {meta_file}")

            with open(buffer_file, mode="wb") as f:
                pickle.dump(self.embeddings_buffer, f)
            logger.info(f"Saved buffer to {buffer_file}")

            with open(cfg_file, mode="w") as f:
                f.write(self.cfg.to_json_string() + "\n")

            self.index_saved_size = self.faiss_index.ntotal
            self.index_save_time = time.time()
            return True

    def _run_save_watcher(self):
        def _save(idx: Index):
            logger.info("Started save watcher thread")
            while True:
                time.sleep(idx.cfg.save_interval_sec)
                logger.info("autosave attempt")
                saved = idx._maybe_save(ignore_time=False)
                logger.info(f"Index saved ={saved}")

        _thread.start_new_thread(_save, (self,))

    def _override_nprobe(self, cfg: IndexCfg):
        logger.info("Overriding nprobe from cfg=%s", cfg)
        if not self.faiss_index:
            logger.warning("Faiss Index is not initialized")
        if hasattr(self.faiss_index, "hnsw"):
            self.faiss_index.hnsw.efSearch = cfg.nprobe
            logger.info("hnsw.efSearch=%s", self.faiss_index.hnsw.efSearch)
        else:
            self.faiss_index.nprobe = cfg.nprobe

    @staticmethod
    def infer_n_centroids(total_data_size):
        if total_data_size < 10e5:
            centroids = int(2 * math.sqrt(total_data_size))
            # TODO: get 4-16 factor from cfg
        elif total_data_size < 10e6:
            centroids = 65536
        elif total_data_size < 10e7:
            centroids = 262144
        else:
            centroids = 1048576
        return centroids
