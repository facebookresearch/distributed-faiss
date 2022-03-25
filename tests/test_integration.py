# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import time
import torch
import random
import string
import _thread
import tempfile
from pathlib import Path
from typing import List

from distributed_faiss.index_cfg import IndexCfg
from distributed_faiss.server import IndexServer
from distributed_faiss.client import IndexClient
from distributed_faiss.index_state import IndexState

import numpy as np

REPO_HOME = Path(".").parent


def get_rand_meta(ndocs, nchars) -> List[str]:
    """Return a random string of length n"""
    return [
        "".join(random.choices(string.ascii_uppercase + string.digits, k=nchars))
        for _ in range(ndocs)
    ]


def load_batched_data(
    client,
    index_id="lang_en",
    max_num_docs_per_batch=100,
    embed_dim=512,
    meta_length=5,
    num_batches=10,
):
    for i in range(num_batches):
        num_docs_per_batch = random.randint(1, max_num_docs_per_batch)
        embeddings = torch.rand(num_docs_per_batch, embed_dim).numpy()
        meta = get_rand_meta(num_docs_per_batch, meta_length)
        client.add_index_data(index_id, embeddings, meta, train_async_if_triggered=False)


class TestIntegration(unittest.TestCase):
    # TODO: add mock tests for all the async calls

    @classmethod
    def setUpClass(self):
        self.multi_server_save_dir = tempfile.TemporaryDirectory()
        self.single_server_save_dir = tempfile.TemporaryDirectory()
        self.embed_dim = 512
        self.index_id = 0

        num_servers = 4
        self.multi_servers = []
        self.multi_ports = [1237, 1238, 1239, 1240]
        random.seed(0)
        for server_id, port in enumerate(self.multi_ports):
            server = IndexServer(server_id, index_storage_dir=self.multi_server_save_dir.name)
            _thread.start_new_thread(server.start_blocking, (port,))
            self.multi_servers.append(server)

        self.single_server_port = 1241
        self.single_server = IndexServer(0, index_storage_dir=self.single_server_save_dir.name)
        _thread.start_new_thread(self.single_server.start_blocking, (self.single_server_port,))
        print("Done setting up test")

    @classmethod
    def tearDownClass(self):
        [s.stop() for s in self.multi_servers]
        self.multi_server_save_dir.cleanup()
        self.single_server_save_dir.cleanup()
        print("Done tearing down test")

    def setUp(self):
        # get unique index_id for each test
        self.index_id = self._testMethodName

    def start_server(self, port):
        single_server = IndexServer(0, index_storage_dir="test_tmp")
        _thread.start_new_thread(single_server.start_blocking, (port,))
        self.servers.append(single_server)
        # TODO: figure out how to close

    @staticmethod
    def make_client(server_port):
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(f"{1}\n".encode())
            fp.write(f"localhost,{server_port}\n".encode())
            fp.seek(0)
            fp.read()
            single_client = IndexClient(fp.name)
        return single_client

    @staticmethod
    def make_clients(num_clients, server_ports):
        clients = []
        num_servers = len(server_ports)
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(f"{num_servers}\n".encode())
            for i in range(num_servers):
                fp.write(f"localhost,{server_ports[i]}\n".encode())
            fp.seek(0)
            fp.read()
            for i in range(num_clients):
                client = IndexClient(fp.name)
                clients.append(client)
        return clients

    def test_train_num_honored(self):
        train_num = 10
        not_trained = IndexState.NOT_TRAINED
        cfg = IndexCfg(index_builder_type="flat", dim=self.embed_dim, train_num=train_num)
        client = self.make_client(self.single_server_port)
        client.create_index(self.index_id, cfg)

        def add_data(ndoc):
            embeddings = torch.rand(ndoc, 512).numpy()
            meta = get_rand_meta(ndoc, 5)
            client.add_index_data(self.index_id, embeddings, meta, train_async_if_triggered=False)
            return client.get_state(self.index_id)

        state = add_data(train_num - 1)
        assert state == not_trained, f"{state} != {not_trained} after only 9 docs"
        state = add_data(1)
        assert state != not_trained, f"{state} == {not_trained} after train_num added"

        results = client.search(torch.rand(4, 512).numpy(), 4, self.index_id)
        assert results[0].shape == (4, 4)
        client.save_index(self.index_id)
        results = client.search(torch.rand(4, 512).numpy(), 4, self.index_id)
        assert results[0].shape == (4, 4)
        client.close()
        # Test save/load behavior
        client2 = self.make_client(self.single_server_port)
        client2.load_index(self.index_id, cfg)
        assert client2.get_state(self.index_id) == IndexState.TRAINED
        results = client2.search(torch.rand(4, 512).numpy(), 4, self.index_id)
        client2.close()

    def test_l2_dist(self):
        train_num = 10
        cfg = IndexCfg(
            index_builder_type="flat",
            dim=self.embed_dim,
            metric="l2",
            centroids=4,
            infer_centroids=False,
        )

        client = self.make_client(self.single_server_port)
        client.create_index(self.index_id, cfg)

        def add_data(ndoc):
            embeddings = torch.rand(ndoc, 512).numpy()
            meta = get_rand_meta(ndoc, 5)
            client.add_index_data(self.index_id, embeddings, meta, train_async_if_triggered=False)
            client.sync_train(self.index_id)
            return client.get_state(self.index_id)

        state = add_data(train_num)
        while True:
            state = client.get_state(self.index_id)
            print("Server state ", state)
            if state == IndexState.TRAINED:
                break
            time.sleep(2)
        assert state == IndexState.TRAINED
        _ = client.search(torch.rand(4, 512).numpy(), 4, self.index_id)

        # TODO: check results differ if metric='dot'
        client.close()

    def test_result_aggregation(self):

        mock_results = [
            (
                np.array([[12.1, 13.2, 13.3, 14.3]], dtype=np.float32),
                [[1465, 1460, 443197, 1340]],
                None,
            ),
            (
                np.array([[8.1, 12.6, 13.1, 17.4]], dtype=np.float32),
                [[0, 14, 3, 1]],
                None,
            ),
        ]
        D, i_minimize = IndexClient._aggregate_results(mock_results, 4, 1, False, False)
        _, i_maximize = IndexClient._aggregate_results(mock_results, 4, 1, True, False)
        assert i_maximize != i_minimize
        assert i_minimize[0][0] == 0  # The smallest distance
        assert D[0][0] < D[0][1]

        assert i_maximize[0][0] == 1  # the largest distance

        assert 0 in i_minimize[0]

    def test_search_quality_same_for_multiple_clients(self):
        # start single flat index as souce of truth
        embed_dim = 512
        num_docs_per_query = 16
        num_batches = 4
        topk_per_search = 5
        meta_length = 5
        cfg = IndexCfg(index_builder_type="flat", dim=embed_dim)

        single_client = self.make_client(self.single_server_port)
        single_client.create_index(self.index_id, cfg)

        clients = self.make_clients(4, self.multi_ports)

        self.assertEqual(single_client.get_state(self.index_id), IndexState.NOT_TRAINED)

        for client in clients:
            client.create_index(self.index_id, cfg)
            self.assertEqual(client.get_state(self.index_id), IndexState.NOT_TRAINED)
            num_batches = random.randint(1, 4)
            for _ in range(num_batches):
                num_docs_per_batch = random.randint(1, 12800)
                embeddings = torch.rand(num_docs_per_batch, embed_dim).numpy()
                meta = get_rand_meta(num_docs_per_batch, meta_length)
                client.add_index_data(
                    self.index_id, embeddings, meta, train_async_if_triggered=False
                )
                single_client.add_index_data(
                    self.index_id, embeddings, meta, train_async_if_triggered=False
                )
                self.assertEqual(client.get_state(self.index_id), IndexState.NOT_TRAINED)

            # we added training data but did not start training yet
            self.assertEqual(client.get_state(self.index_id), IndexState.NOT_TRAINED)
        clients[0].sync_train(self.index_id)
        single_client.sync_train(self.index_id)

        while True:
            state = clients[0].get_state(self.index_id)
            print("Server state ", state)
            if state == IndexState.TRAINED:
                break
            time.sleep(2)

        while True:
            state = single_client.get_state(self.index_id)
            print("Server state ", state)
            if state == IndexState.TRAINED:
                break
            time.sleep(2)

        self.assertEqual(
            clients[0].get_ntotal(self.index_id), single_client.get_ntotal(self.index_id)
        )
        query = torch.rand(num_docs_per_query, embed_dim).numpy()
        scores_aggr, meta_aggr = clients[0].search(query, topk_per_search, self.index_id)
        scores_single, meta_single = single_client.search(query, topk_per_search, self.index_id)
        self.assertTrue((scores_aggr == scores_single).all())
        self.assertEqual(len(meta_aggr), len(meta_single))
        self.assertEqual(meta_aggr, meta_single)
        single_client.close()

    def test_index_client_multiple_server(self):
        embed_dim = 512
        num_docs_per_batch = 12800
        num_docs_per_query = 16
        num_batches = 4
        topk_per_search = 5
        meta_length = 5

        clients = self.make_clients(4, self.multi_ports)

        cfg = IndexCfg("flat", dim=embed_dim)
        assert cfg.index_builder_type == "flat"

        for client in clients:
            client.create_index(self.index_id, cfg)
            self.assertEqual(client.get_state(self.index_id), IndexState.NOT_TRAINED)
            for i in range(num_batches):
                embeddings = torch.rand(num_docs_per_batch, embed_dim).numpy()
                meta = get_rand_meta(num_docs_per_batch, meta_length)
                client.add_index_data(
                    self.index_id, embeddings, meta, train_async_if_triggered=False
                )
            # we added training data but did not start training yet
            self.assertEqual(client.get_state(self.index_id), IndexState.NOT_TRAINED)

        clients[0].sync_train(self.index_id)
        for client in clients:
            for i in range(num_batches):
                embeddings = torch.rand(num_docs_per_batch, embed_dim).numpy()
                meta = get_rand_meta(num_docs_per_batch, meta_length)
                client.add_index_data(
                    self.index_id, embeddings, meta, train_async_if_triggered=False
                )
        for client in clients:
            while True:
                state = client.get_state(self.index_id)
                print("Server state ", state)
                if state == IndexState.TRAINED:
                    break
                time.sleep(2)

        for server in self.multi_servers:
            # Make sure that data is ballanced among servers
            self.assertEqual(
                server.get_ntotal(self.index_id),
                2 * num_batches * num_docs_per_batch * len(clients) / len(self.multi_servers),
            )
        for client in clients:
            self.assertEqual(client.get_state(self.index_id), IndexState.TRAINED)
            self.assertEqual(
                client.get_ntotal(self.index_id),
                2 * num_batches * num_docs_per_batch * len(clients),
            )
            self.assertEqual(client.get_ntotal("wrong_id"), 0)
            query = torch.rand(num_docs_per_query, embed_dim).numpy()
            scores, meta = client.search(query, topk_per_search, self.index_id)
            self.assertEqual((num_docs_per_query, topk_per_search), scores.shape)
            self.assertEqual(num_docs_per_query, len(meta))
            self.assertEqual(topk_per_search, len(meta[0]))

        clients[0].save_index(self.index_id)
        clients[0].drop_index(self.index_id)
        for client in clients:
            self.assertEqual(client.get_ntotal(self.index_id), 0)

    def test_config_to_file(self):
        train_num = 13
        nprobe = 16
        cfg = IndexCfg(
            index_builder_type="flat", dim=self.embed_dim, train_num=train_num, nprobe=nprobe
        )

        client = self.make_client(self.single_server_port)
        client.create_index(self.index_id, cfg)

        def add_data(ndoc):
            embeddings = torch.rand(ndoc, 512).numpy()
            meta = get_rand_meta(ndoc, 5)
            client.add_index_data(self.index_id, embeddings, meta, train_async_if_triggered=False)

        add_data(train_num)
        while True:
            state = client.get_state(self.index_id)
            print("Server state ", state)
            if state == IndexState.TRAINED:
                break
            time.sleep(2)

        client.save_index(self.index_id)
        cfg_path = os.path.join(self.single_server_save_dir.name, self.index_id, "0", "cfg.json")
        print("cfg_path", cfg_path)
        assert os.path.isfile(cfg_path)

        client.close()

        # Load config from file
        client2 = self.make_client(self.single_server_port)
        client2.load_index(self.index_id)

        assert client2.cfg.index_builder_type == "flat"
        assert client2.cfg.dim == self.embed_dim
        assert client2.cfg.train_num == train_num
        assert client2.cfg.nprobe == nprobe
        client2.close()

        # Overwrite config at startup
        cfg3 = IndexCfg(
            index_builder_type="flat",
            dim=self.embed_dim,
            train_num=train_num,
            nprobe=nprobe + 1,
        )
        client3 = self.make_client(self.single_server_port)
        client3.load_index(self.index_id, cfg3)
        assert client3.cfg.index_builder_type == "flat"
        assert client3.cfg.dim == self.embed_dim
        assert client3.cfg.train_num == train_num
        assert client3.cfg.nprobe == nprobe + 1
        client3.close()

    def test_get_centroids(self):
        index_id = "ivf_test"
        train_num = 10
        centroids = 2
        cfg = IndexCfg(
            index_builder_type="ivf_simple",
            dim=self.embed_dim,
            train_num=train_num,
            centroids=centroids,
        )
        client = self.make_client(self.single_server_port)
        client.create_index(index_id, cfg)

        def add_data(ndoc):
            embeddings = torch.rand(ndoc, self.embed_dim).numpy()
            meta = get_rand_meta(ndoc, 5)
            client.add_index_data(index_id, embeddings, meta, train_async_if_triggered=False)
            return client.get_state(index_id)

        state = add_data(train_num)

        while True:
            print("Server state ", state)
            if state == IndexState.TRAINED:
                break
            time.sleep(2)
            state = client.get_state(index_id)

        assert client.get_centroids(index_id)[0].shape == (centroids, self.embed_dim)
        client.close()


class TestIndexCfg(unittest.TestCase):
    def test_from_index_cfg_from_json(self):
        IndexCfg.from_json(REPO_HOME.joinpath("tests/test_index_config.json"))


if __name__ == "__main__":
    unittest.main()
