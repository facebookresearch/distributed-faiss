#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import os
from distributed_faiss.client import IndexClient


TESTSERVERLIST_FILENAME = os.path.join(os.path.dirname(__file__), "test_server_list.txt")
TESTSERVERLIST_FILENAME_FAIL = os.path.join(os.path.dirname(__file__), "test_server_list_fail.txt")


class TestClient(unittest.TestCase):
    def test_read_server_list(self):
        servers = IndexClient.read_server_list(TESTSERVERLIST_FILENAME)
        self.assertEqual(len(servers), 4)
        self.assertEqual(servers[0][0], "machine1234")
        self.assertEqual(servers[0][1], 8080)
        self.assertEqual(servers[3][0], "machine5678")
        self.assertEqual(servers[3][1], 8081)

        with self.assertRaises(Exception) as context:
            IndexClient.read_server_list(
                TESTSERVERLIST_FILENAME_FAIL,
                total_max_timeout=0,
            )
        self.assertEqual(
            (
                f"4 != 3 in server list "
                f"{TESTSERVERLIST_FILENAME_FAIL}. Timed "
                "out after waiting 0.0 seconds"
            ),
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
