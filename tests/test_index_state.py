#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from distributed_faiss.index_state import IndexState


class TestIndexState(unittest.TestCase):
    def test_get_aggregated_states(self):
        states = [IndexState.NOT_TRAINED, IndexState.TRAINED, IndexState.TRAINING]
        self.assertEqual(IndexState.get_aggregated_states(states), IndexState.TRAINING)

        states = [IndexState.TRAINED, IndexState.TRAINED, IndexState.TRAINED]
        self.assertEqual(IndexState.get_aggregated_states(states), IndexState.TRAINED)

        states = [IndexState.NOT_TRAINED, IndexState.NOT_TRAINED, IndexState.TRAINED]
        self.assertEqual(IndexState.get_aggregated_states(states), IndexState.NOT_TRAINED)


if __name__ == "__main__":
    unittest.main()
