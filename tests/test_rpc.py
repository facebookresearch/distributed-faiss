# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import _thread
from multiprocessing import Pool as ProcessPool

import time
import torch

from distributed_faiss.index_cfg import IndexCfg
from distributed_faiss.index_state import IndexState
from distributed_faiss.rpc import Client
from distributed_faiss.server import IndexServer


def add_train_data(c: Client, index_id):
    print("Call client  ", c.id)
    c.add_index_data(index_id, torch.rand(100, 512).numpy(), None)


def call_train(c: Client):
    c.async_train(0)


def run_client(id):
    print("Run client ", id)
    c = Client(id, "localhost")
    cfg = IndexCfg(index_builder_type="flat", dim=512)
    idx_id = "idx:1"
    c.create_index(idx_id, cfg)
    for i in range(10):
        add_train_data(c, idx_id)

    c.async_train(idx_id)

    c.add_buffer_to_index(idx_id)

    while True:
        state = c.get_state(idx_id)
        print("Server state {}".format(state))
        if state == IndexState.TRAINED:
            break
        time.sleep(2)

    for i in range(10):
        _result = c.search(idx_id, torch.rand(5, 512).numpy(), 5, True)
    c.close()


class TestRPC(unittest.TestCase):
    save_dir = "/tmp/distributed_faiss/"

    def test_single_server_multiple_clients_threaded(self):
        server = IndexServer(0, index_storage_dir=self.save_dir)
        _thread.start_new_thread(server.start_blocking, ())
        time.sleep(2)  # let it start accepting clients
        processes = ProcessPool(processes=10)
        ids = list(range(10))
        processes.map(run_client, ids)
        server.stop()

    @unittest.skip("Fails with ValueError: I/O operation on closed file.")
    def test_single_server_multiple_clients(self):
        server = IndexServer(0, index_storage_dir=self.save_dir)
        _thread.start_new_thread(server.start, ())

        time.sleep(2)  # let it start accepting clients

        clients = []
        for i in range(10):
            c = Client(i, "localhost")
            clients.append(c)

        index_key = "lang_en"

        for i in range(10):
            [
                c.add_train_data(index_key, torch.rand(100, 512).numpy(), None)
                # [("test_meta", c_id*100+i*10+j, None) for j in range(10)]
                for c_id, c in enumerate(clients)
            ]

        [c.async_train(0) for c in clients]

        while True:
            states = [c.get_state() for c in clients]
            print("Server states ", states)
            if all(s == IndexState.TRAINED for s in states):
                break
            time.sleep(2)

        # query
        results = [c.search(index_key, torch.rand(1, 512).numpy(), 5, True) for c in clients]

        print("Result 0", results[0])


if __name__ == "__main__":
    # test_single_server_multiple_clients()
    unittest.main()
