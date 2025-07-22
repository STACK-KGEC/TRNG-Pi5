import socket
import numpy as np
import pickle
import struct
from scipy.stats import norm

HOST = "192.168.43.223"  # Your Pi's IP
PORT = 12345

class LightRandom:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port

    def _request_data(self, length):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.host, self.port))
        client_socket.send(struct.pack('!I', length))  # Send 4-byte request
        data = b""

        while True:
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            data += chunk
            try:
                readings = np.array(pickle.loads(data))
                break
            except:
                continue 

        client_socket.close()

        randomized = self._whiten(readings)

        return randomized
    
    def _whiten(self, readings):
        return np.array([self._temper(x) for x in readings])
    
    def _temper(self, x: int) -> int:
        x ^= (x >> 11)
        x ^= (x << 7) & 0x9D2C5680
        x ^= (x << 15) & 0xEFC60000
        x ^= (x >> 18)
        return x & 0xFFFFFFFF

    def rand(self, *shape):
        count = np.prod(shape) if shape else 1
        values = self._request_data(count)
        return values.reshape(shape)

    def randint(self, low, high=None, size=None):
        if high is None:
            low, high = 0, low
        shape = size if size else ()
        count = np.prod(shape) if shape else 1
        values = self._request_data(count)
        return (low + values * (high - low)).astype(int).reshape(shape)

    def randn(self, *shape):
        u = self.rand(*shape)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return norm.ppf(u)  # Inverse CDF of normal distribution

    def shuffle(self, arr):
        arr = arr.copy()
        n = len(arr)
        indices = self.randint(0, n, size=n)
        for i in reversed(range(1, n)):
            j = indices[i]
            arr[i], arr[j] = arr[j], arr[i]
        return arr

    def choice(self, a, size=None):
        a = np.array(a)
        shape = (size,) if isinstance(size, int) else size
        idx = self.randint(0, len(a), size=shape)
        return a[idx]
