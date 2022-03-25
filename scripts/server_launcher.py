#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import submitit
import math
import argparse
import os
import errno
import time
from distributed_faiss.server import IndexServer, DEFAULT_PORT

"""
fcntl.flock is broken for NFS. Using this workaround:
https://stackoverflow.com/questions/37633951/python-locking-text-file-on-nfs
"""


def lockfile(target, link, timeout=300):
    global lock_owner
    poll_time = 10
    while timeout > 0:
        try:
            os.link(target, link)
            lock_owner = True
            break
        except OSError as err:
            if err.errno == errno.EEXIST:
                print("Lock unavailable. Waiting for 10 seconds...")
                time.sleep(poll_time)
                timeout -= poll_time
            else:
                raise err
    else:
        print("Timed out waiting for the lock.")


def releaselock(link):
    try:
        if lock_owner:
            os.unlink(link)
    except OSError:
        print("Error:didn't possess lock.")


def append_to_discovery_config_safe(discovery_config, msg):
    # tmp_link will be destroyed after unlink
    tmp_link = discovery_config + ".link"
    lockfile(discovery_config, tmp_link)
    with open(discovery_config, "a") as config:
        config.write(msg)
    releaselock(tmp_link)


def run_server(discovery_config, base_port, index_storage_dir: str, load_index=False):
    job_env = submitit.JobEnvironment()
    # add local rank to avoid port conflict on the same machine
    port = base_port + job_env.local_rank

    append_to_discovery_config_safe(discovery_config, f"{job_env.hostname},{port}\r\n")

    server = IndexServer(job_env.global_rank, index_storage_dir)
    server.start_blocking(port, v6=False, load_index=load_index)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, help="log data dir")
    parser.add_argument(
        "--discovery-config",
        required=True,
        help="where to store info about running servers",
    )
    parser.add_argument("--num-servers", type=int, required=True, help="how many servers to run")
    parser.add_argument("--save-dir", required=False, default="distributed_faiss_storage")
    parser.add_argument(
        "--num-servers-per-node", type=int, default=10, help="how many servers per node"
    )
    parser.add_argument("--mem-gb", type=int, default=8, help="how much RAM per process")
    parser.add_argument("--cpus-per-node", type=int, default=32, help="how many cpus per node")
    parser.add_argument("--partition", required=True, help="slurm partition")
    parser.add_argument("--timeout-min", type=int, default=60, help="how to run for in min")
    parser.add_argument(
        "--base-port",
        type=int,
        default=DEFAULT_PORT,
        help="base-port + local-rank is going to be the final port",
    )
    parser.add_argument(
        "--load-index",
        default=False,
        action="store_true",
        help="If true server will try to load index and meta from disk",
    )
    parser.add_argument(
        "--comment",
        type=str,
    )

    args = parser.parse_args()

    discovery_config = open(args.discovery_config, "w")
    discovery_config.write(f"{args.num_servers}\r\n")
    discovery_config.close()

    executor = submitit.AutoExecutor(folder=args.log_dir)
    num_nodes = math.ceil(args.num_servers / args.num_servers_per_node)
    cpus_per_server = math.floor(args.cpus_per_node / args.num_servers_per_node)
    executor.update_parameters(
        tasks_per_node=args.num_servers_per_node,
        nodes=num_nodes,
        timeout_min=args.timeout_min,
        slurm_partition=args.partition,
        mem_gb=args.mem_gb,
        cpus_per_task=cpus_per_server,
        comment=args.comment,
    )
    job = executor.submit(
        run_server,
        args.discovery_config,
        args.base_port,
        args.save_dir,
        args.load_index,
    )
    print(job.results())


if __name__ == "__main__":
    main()
