# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import List


class IndexState(Enum):
    NOT_TRAINED = 1
    TRAINING = 2
    ADD = 3
    TRAINED = 4

    @staticmethod
    def get_aggregated_states(states: List["IndexState"]) -> "IndexState":
        states = set(states)
        assert len(states) > 0
        # check if all the states are consistent
        if len(states) == 1:
            return states.pop()
        # consider cluster to be still in training state
        # if at least one server is still training
        if IndexState.TRAINING in states:
            return IndexState.TRAINING
        # otherwise we are in a state where some servers
        # are trained and some are not trained.
        if IndexState.NOT_TRAINED in states:
            return IndexState.NOT_TRAINED
        # some nodes may be in the ADD-ing state
        if IndexState.ADD in states:
            return IndexState.ADD
        else:
            return IndexState.TRAINED
