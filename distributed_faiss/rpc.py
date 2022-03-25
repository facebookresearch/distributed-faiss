#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Simplistic RPC implementation.
Exposes all functions of a Server object.
Uses pickle for serialization and the socket interface.

Copied from https://github.com/facebookresearch/faiss/blob/master/benchs/distributed_ondisk/rpc.py
"""

import os
import pickle
import socket

# default
DEFAULT_PORT = 12032


#########################################################################
# simple I/O functions


def inline_send_handle(f, conn):
    st = os.fstat(f.fileno())
    size = st.st_size
    pickle.dump(size, conn)
    conn.write(f.read(size))


def inline_send_string(s, conn):
    size = len(s)
    pickle.dump(size, conn)
    conn.write(s)


class FileSock:
    """
    wraps a socket so that it is usable by pickle/cPickle
    """

    def __init__(self, sock):
        self.sock = sock
        self.nr = 0
        self.last_read_len = 0

    """
    def write(self, buf):
        print("sending %d bytes ", len(buf), flush=True)
        self.sock.sendall(buf)
    """

    def write(self, buf):
        # print("sending %d bytes"%len(buf))
        # self.sock.sendall(buf)
        # print("...done")
        bs = 128 * 512 * 1024
        ns = 0
        while ns < len(buf):
            sent = self.sock.send(buf[ns : ns + bs])
            ns += sent

    def read(self, bs=128 * 512 * 1024):
        self.nr += 1
        b = []
        nb = 0
        while len(b) < bs:
            # print('   loop')
            rb = self.sock.recv(bs - nb)
            if not rb:
                break
            b.append(rb)
            nb += len(rb)

        # logger.info("read nb=%s", nb)

        self.last_read_len = nb
        return b"".join(b)

    def readline(self):
        # print("readline!")
        """may be optimized..."""
        s = bytes()
        while True:
            c = self.read(1)
            s += c
        if len(c) == 0 or chr(c[0]) == "\n":
            return s


class ClientExit(Exception):
    pass


class ServerException(Exception):
    pass


class Client:
    """
    Methods of the server object can be called transparently. Exceptions are
    re-raised.
    """

    def __init__(self, id, HOST, port=DEFAULT_PORT, v6=False):
        self.id = id
        socktype = socket.AF_INET6 if v6 else socket.AF_INET

        sock = socket.socket(socktype, socket.SOCK_STREAM)
        print("connecting", HOST, port, socktype)
        sock.connect((HOST, port))
        self.sock = sock
        self.fs = FileSock(sock)

    def generic_fun(self, fname, args):
        # int "gen fun",fname
        # logger.info('Client=%s, call fname=%s', self.id, fname)
        pickle.dump((fname, args), self.fs, protocol=4)
        return self.get_result()

    def get_result(self):
        (st, ret) = pickle.load(self.fs)
        if st != None:
            raise ServerException(st)
        else:
            return ret

    def close(self):
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()

    def __getattr__(self, name):
        return lambda *x: self.generic_fun(name, x)
