# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import _thread
import argparse
import logging
import os
import pathlib
import pickle
import selectors
import socket
import sys
import threading
import traceback
import types
from typing import List, Tuple, Optional

import numpy as np

from distributed_faiss.index import Index
from distributed_faiss.index_cfg import IndexCfg
from distributed_faiss.index_state import IndexState
from distributed_faiss.rpc import FileSock, ClientExit, DEFAULT_PORT

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)


class IndexServer:
    def __init__(self, rank: int, index_storage_dir):
        self.indexes = {}
        self.indexes_lock = threading.Lock()
        self.rank = rank
        self.socket = None
        logger.info(f"IndexServer saving to {index_storage_dir} , rank={rank}")
        self.index_storage_dir = index_storage_dir

    def save_index(self, index_id: str):
        """Save index, metadata and buffer to {storage_dir}/{index_id}/{rank}/"""
        with self.indexes_lock:
            if index_id not in self.indexes:
                raise RuntimeError(f"Index with id={index_id} is not initialized")
            index = self.indexes[index_id]
        index.save()

    def load_index(self, index_id: str = "default", cfg: IndexCfg = None) -> bool:
        """Load index, metadata and buffer from {storage_dir}/{index_id}/{rank}/"""

        index_dir = self._get_storage_dir(index_id, cfg)
        if cfg:
            cfg.index_storage_dir = index_dir

        with self.indexes_lock:
            if index_id in self.indexes:
                logging.info("Index already exists: %s", index_id)
                if cfg:
                    self.indexes[index_id].upd_cfg(cfg)
                return True
            else:
                logger.info(f"Loading index from {index_dir}")
                index = Index.from_storage_dir(index_dir, cfg)
                if index:
                    self.indexes[index_id] = index
                    logger.info(f"Index id={index_id} has been loaded")
                    return True
                else:
                    logger.info(f"Can't load index")
                    return False

    def get_ids(self, index_id: str = "default") -> set:
        with self.indexes_lock:
            index = self.indexes[index_id]
        return index.get_ids()

    def get_rank(self) -> int:
        return self.rank

    def index_loaded(self, index_id: str) -> bool:
        """Check if an index with a given index_id exists and if it is trained"""
        with self.indexes_lock:
            return (
                index_id in self.indexes
                and self.indexes[index_id].get_state() == IndexState.TRAINED
            )

    def start_blocking(self, port=DEFAULT_PORT, v6=False, load_index=False):
        if load_index:
            self.load_index()
        HOST = ""  # Symbolic name meaning the local host
        socktype = socket.AF_INET6 if v6 else socket.AF_INET
        s = socket.socket(socktype, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        logger.info("bind %s:%d", HOST, port)
        s.bind((HOST, port))
        s.listen(10)

        while True:
            try:
                conn, addr = s.accept()
            except socket.error as e:
                if e[1] == "Interrupted system call":
                    continue
                raise

            logger.info("Connected by %s", addr)
            tid = _thread.start_new_thread(self.exec_loop_blocking, (conn,))

    def exec_loop_blocking(self, socket):
        """main execution loop. Loops and handles exit states"""

        fs = FileSock(socket)
        logger.info("in exec_loop")
        try:
            while True:
                self.one_function_blocking(socket, fs)
        except ClientExit as e:
            logger.info("ClientExit %s", e)
        except socket.error as e:
            logger.error("socket error %s", e)
        except EOFError:
            logger.error("EOF during communication")
        except BaseException:
            # unexpected
            traceback.print_exc(50, sys.stderr)
            sys.exit(1)
        logger.info("exit server")

    def start(self, port=DEFAULT_PORT, v6=False):
        sel = selectors.DefaultSelector()
        HOST = ""
        socktype = socket.AF_INET6 if v6 else socket.AF_INET
        s = socket.socket(socktype, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        logger.info("bind %s:%d", HOST, port)
        s.bind((HOST, port))
        s.listen(10)
        s.setblocking(False)
        sel.register(s, selectors.EVENT_READ, data=None)
        self.socket = s
        while True:

            try:
                events = sel.select(timeout=None)
                for key, mask in events:
                    if key.data is None:
                        logger.info(" ")
                        self._accept_wrapper(key.fileobj, sel)
                    else:
                        self._service_connection(key, mask, sel)
            except Exception as e:
                if hasattr(e, "message"):
                    print(e.message)
                else:
                    print(e)
                break

        logger.info("Server service loop ended")

    def one_function(self, socket: FileSock) -> int:
        """
        - server sends result: (rid,st,ret)
            st = None, or exception if there was during execution
            ret = return value or None if st!=None
        """
        fname = None
        args = None
        try:
            client_request = pickle.load(socket)
            read_bytes = socket.last_read_len
            # logger.info('read_bytes %s', read_bytes)
            if read_bytes > 0:
                (fname, args) = client_request
                logger.info("fname %s", fname)
        except EOFError:
            return 0

        st = None
        ret = None
        f = None

        try:
            f = getattr(self, fname)
        except AttributeError:
            st = AttributeError("unknown method " + fname)
            logger.error("unknown method %s", fname)
        try:
            # TODO: decide if a separate thread is needed
            if f == self.add_index_data:  # same thread
                ret = f(*args)
            else:
                ret = f(*args)

        except Exception as e:
            st = "".join(traceback.format_tb(sys.exc_info()[2])) + str(e)
            logger.error("exception in method %s", f)
            logger.error("%s", st)

        try:
            pickle.dump((st, ret), socket, protocol=4)
        except EOFError:
            raise ClientExit("function return")

        return read_bytes

    def one_function_blocking(self, socket, fs: FileSock):
        try:
            (fname, args) = pickle.load(fs)
        except EOFError:
            raise ClientExit("read args")
        logger.info("executing method %s", fname)
        st = None
        ret = None
        try:
            f = getattr(self, fname)
        except AttributeError:
            st = "unknown method " + fname
            logger.error("unknown method %s", fname)
        try:
            ret = f(*args)
        except Exception as e:
            st = "".join(traceback.format_tb(sys.exc_info()[2])) + str(e)
            logger.error("exception in method: %s", traceback.print_exc(50))

        logger.info("return")
        try:
            pickle.dump((st, ret), fs, protocol=4)
        except EOFError:
            raise ClientExit("function return")

    def create_index(self, index_id: str, cfg: IndexCfg):
        index_storage_dir = self._get_storage_dir(index_id, cfg)
        cfg.index_storage_dir = index_storage_dir
        pathlib.Path(index_storage_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Set index save dir to: {index_storage_dir}")

        with self.indexes_lock:
            if index_id not in self.indexes:
                self.indexes[index_id] = Index(cfg)
                logger.info(f"Created new index {index_id}")
                logger.info(f"CFG: {str(cfg)}")
                return True
            logger.info(f"Index {index_id} already exists")
            return False

    def add_index_data(
        self,
        index_id: str,
        embeddings: np.array,
        metadata: Optional[List[object]] = None,
        train_async_if_triggered: bool = True,
    ):
        logger.info("adding embeddings idx=%s, embeddings=%s", index_id, embeddings.shape)

        with self.indexes_lock:
            index = self.indexes[index_id]
        index.add_batch(embeddings, metadata, train_async_if_triggered)

    def get_aggregated_ntotal(self, index_id: str) -> int:
        logger.info("getting current buffer data size idx=%s")
        with self.indexes_lock:
            index = self.indexes[index_id]
        return index.get_idx_data_num()[0]

    def stop(self):
        logger.info("Stopping server ...")
        if self.socket:
            self.socket.shutdown(socket.SHUT_RDWR)
            self.socket.close()
            self.socket = None

        for index_id in self.indexes:
            self.indexes[index_id].save()

    def get_ntotal(self, index_id: str) -> int:
        with self.indexes_lock:
            if index_id not in self.indexes:
                return 0
            index = self.indexes[index_id]
        index_data_num = index.get_idx_data_num()
        logger.info("Index id=%s data size idx=%s", index_id, index_data_num)
        return index_data_num[1]

    def drop_index(self, index_id: str):
        with self.indexes_lock:
            logger.info(f"Dropping index {index_id}")
            if index_id in self.indexes:
                del self.indexes[index_id]

    def sync_train(self, index_id: str):
        """
        Warning, this method will block the main clients' service loop so all client requests will be stalled
        :return: 0 as of now,
        """
        # TODO: don't block service loop, allocate a separate thread ?
        logger.info(f"Sync train started")
        self._train_index(index_id)

    def async_train(self, index_id: str):
        class IndexTrainer(threading.Thread):
            def __init__(self, server: IndexServer, *args, **kwargs):
                super(IndexTrainer, self).__init__(*args, **kwargs)
                self.server = server

            def run(self):
                self.server._train_index(index_id)

        t = IndexTrainer(self)
        t.run()

    def search(
        self, index_id: str, query_batch: np.array, top_k: int, return_embeddings: bool
    ) -> Tuple:
        logger.info(f"Query idx={index_id}, query={query_batch.shape}")
        index = self._get_index(index_id)
        assert not isinstance(index, set)
        r = index.search(query_batch, top_k=top_k, return_embeddings=return_embeddings)
        return r

    def get_centroids(self, index_id: str):
        index = self._get_index(index_id)
        return index.get_centroids()

    def set_nprobe(self, index_id: str, nprobe: int):
        index = self._get_index(index_id)
        return index.set_nprobe(nprobe)

    def get_state(self, index_id: str):
        index = self._get_index(index_id)
        return index.get_state()

    def add_buffer_to_index(self, index_id: str):
        index = self._get_index(index_id)
        return index.add_buffer_to_index()

    def get_config_path(self, index_id: str):
        cfg_path = os.path.join(self.index_storage_dir, index_id, str(self.rank), "cfg.json")
        return cfg_path

    def _get_index(self, index_id: str):
        with self.indexes_lock:
            if index_id not in self.indexes:
                raise RuntimeError("Server has no index with id={}".format(index_id))
            return self.indexes[index_id]

    def _accept_wrapper(self, sock, sel):
        conn, addr = sock.accept()  # Should be ready to read
        logger.info("accepted connection from %s", addr)
        conn.setblocking(False)
        data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
        events = selectors.EVENT_READ | selectors.EVENT_WRITE  # TODO: remove write?
        sel.register(conn, events, data=data)

    def _service_connection(self, key, mask, sel):
        sock = key.fileobj
        data = key.data

        fs = FileSock(sock)

        if mask & selectors.EVENT_READ:
            read_bytes = self.one_function(fs)

            if read_bytes == 0:
                logger.info("closing connection to %s", data.addr)
                sel.unregister(sock)
                sock.close()

    def _train_index(self, index_id: str):
        index = self._get_index(index_id)
        logger.info(f"Training index {index_id}")
        index.train()

    def _get_storage_dir(self, index_id: str, cfg: IndexCfg):
        index_storage_dir = cfg.index_storage_dir if cfg else None
        if not index_storage_dir:
            index_storage_dir = os.path.join(self.index_storage_dir, index_id, str(self.rank))
        else:
            index_storage_dir = os.path.join(index_storage_dir, str(self.rank))
        return index_storage_dir


def main():
    parser = argparse.ArgumentParser()

    # reader specific params
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help="TBD")
    parser.add_argument("--ipv4", default=False, action="store_true", help="force ipv4")
    args = parser.parse_args()
    logger.info("starting server ...")
    server = IndexServer()
    server.start(args.port, v6=not args.ipv4)


if __name__ == "__main__":
    main()
