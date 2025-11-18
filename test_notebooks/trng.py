import socket
import numpy as np
import pickle
import struct
from scipy.stats import norm

HOST = "10.38.38.79"  # Your Pi's IP
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
        # Convert floats to 32-bit unsigned ints by scaling
        readings_int = (np.array(readings) * 1e4).astype(np.uint32)
        return np.array([self._temper(x) for x in readings_int], dtype=np.uint32)

    def _temper(self, x: int) -> int:
        x = int(x) & 0xFFFFFFFF  # Ensure 32-bit unsigned int
        x ^= (x >> 11)
        x ^= (x << 7) & 0x9D2C5680
        x ^= (x << 15) & 0xEFC60000
        x ^= (x >> 18)
        return x & 0xFFFFFFFF

    def rand(self, *shape):
        count = np.prod(shape) if shape else 1
        values = self._request_data(count)
        return values.astype(np.float64) / 0xFFFFFFFF  # Normalize to [0, 1]

    def randint(self, low, high=None, size=None):
        if high is None:
            low, high = 0, low
        shape = size if size else ()
        count = np.prod(shape) if shape else 1
        values = self._request_data(count).astype(np.float64) / 0xFFFFFFFF
        return (low + values * (high - low)).astype(int).reshape(shape)
    
    def uniform(self, low=0.0, high=1.0, size=None):
        u = self.rand((size))  # values in [0, 1)
        return low + (high - low) * u

    def randn(self, *shape):
        u = self.rand(*shape)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return norm.ppf(u)  # Inverse CDF of standard normal

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
