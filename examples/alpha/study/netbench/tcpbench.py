#!/usr/bin/env python3
"""Minimal TCP bandwidth benchmark (iperf3 substitute).

server: python3 tcpbench.py server <port> <nstreams>
client: python3 tcpbench.py client <host> <port> <nstreams> <seconds>
"""
import socket
import sys
import threading
import time

BUF = 1024 * 1024  # 1 MiB


def server(port: int, nstreams: int):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(nstreams)
    print(f"listening on :{port} for {nstreams} streams", flush=True)
    totals = [0] * nstreams
    threads = []

    def handle(conn, i):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            data = conn.recv(BUF)
            if not data:
                break
            totals[i] += len(data)
        conn.close()

    start = None
    for i in range(nstreams):
        conn, _ = srv.accept()
        if start is None:
            start = time.monotonic()
        t = threading.Thread(target=handle, args=(conn, i))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start
    total = sum(totals)
    print(f"received {total / 1e9:.2f} GB in {elapsed:.2f}s "
          f"= {total * 8 / elapsed / 1e9:.2f} Gbit/s aggregate", flush=True)


def client(host: str, port: int, nstreams: int, seconds: float):
    payload = b"\x00" * BUF
    sent = [0] * nstreams
    stop = time.monotonic() + seconds

    def pump(i):
        s = socket.create_connection((host, port))
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while time.monotonic() < stop:
            s.sendall(payload)
            sent[i] += BUF
        s.close()

    threads = [threading.Thread(target=pump, args=(i,)) for i in range(nstreams)]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start
    total = sum(sent)
    print(f"sent {total / 1e9:.2f} GB in {elapsed:.2f}s "
          f"= {total * 8 / elapsed / 1e9:.2f} Gbit/s aggregate "
          f"({nstreams} streams)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "server":
        server(int(sys.argv[2]), int(sys.argv[3]))
    else:
        client(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
