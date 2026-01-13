import os
import random
from Crypto.Cipher import ChaCha20
import numpy as np


class TRNG_RNG:
    def __init__(self, trng_file_path: str):
        """
        trng_file_path: path to a text file containing exactly 1M bits ('0'/'1')
        """
        with open(trng_file_path, "r") as f:
            self.bitstream = f.read().strip()

        if len(self.bitstream) < 1000000:
            raise ValueError("TRNG bitstream must contain at least 1,000,000 bits")

        self.seed_256 = self._generate_256bit_seed()
        self.chacha = self._init_chacha(self.seed_256)
        self.buffer = bytearray()
        self.buffer_pos = 0

    # ------------------------------------------------------------
    # STEP 1: Take 4096 random consecutive bits
    # ------------------------------------------------------------
    def _take_4096_bits(self):
        start = random.randint(0, len(self.bitstream) - 4096)
        segment = self.bitstream[start : start + 4096]
        return np.array([int(b) for b in segment], dtype=np.uint8)

    # ------------------------------------------------------------
    # STEP 2: Toeplitz universal hash (4096x256)
    # ------------------------------------------------------------
    def _toeplitz_seed(self, bits4096):
        # Toeplitz first row + first column length = 4096 + 256 - 1
        t = np.random.randint(0, 2, 4096 + 256 - 1, dtype=np.uint8)

        seed_bits = np.zeros(256, dtype=np.uint8)

        # Toeplitz matrix multiplication with GF(2)
        for i in range(256):
            window = t[i:i+4096]
            seed_bits[i] = np.bitwise_xor.reduce(bits4096 & window)

        # Convert bit array → 32-byte seed
        seed_bytes = np.packbits(seed_bits).tobytes()
        return seed_bytes

    def _generate_256bit_seed(self):
        bits4096 = self._take_4096_bits()
        return self._toeplitz_seed(bits4096)

    # ------------------------------------------------------------
    # STEP 3: ChaCha20 expansion using PyCryptodome
    # ------------------------------------------------------------
    def _init_chacha(self, seed32):
        nonce = os.urandom(8)  # ChaCha20 uses 8-byte nonce in PyCryptodome
        return ChaCha20.new(key=seed32, nonce=nonce)

    def _get_bytes(self, n):
        """Return n random bytes, refill buffer using ChaCha20."""
        return self.chacha.encrypt(b"\x00" * n)

    # Convert bytes to integers
    def randbits(self, k):
        nbytes = (k + 7) // 8
        data = self._get_bytes(nbytes)
        val = int.from_bytes(data, "big")
        return val & ((1 << k) - 1)

    # ------------------------------------------------------------
    # NumPy-style RNG functions
    # ------------------------------------------------------------
    def randint(self, low, high=None):
        if high is None:
            low, high = 0, low
        span = high - low
        bits_needed = span.bit_length()
        while True:
            r = self.randbits(bits_needed)
            if r < span:
                return low + r

    def random(self):
        """Return float in [0,1)."""
        return self.randbits(53) / (1 << 53)

    def choice(self, seq):
        idx = self.randint(0, len(seq))
        return seq[idx]

    def uniform(self, a=0.0, b=1.0):
        return a + (b - a) * self.random()

    def shuffle(self, arr):
        """Fisher–Yates shuffle using ChaCha output."""
        for i in reversed(range(1, len(arr))):
            j = self.randint(0, i + 1)
            arr[i], arr[j] = arr[j], arr[i]


# ------------------------------------------------------------
# EXAMPLE usage
# ------------------------------------------------------------
if __name__ == "__main__":
    rng = TRNG_RNG("readings_1.txt")

    print("Random int 0–99:", rng.randint(0, 100))
    print("Random float:", rng.random())
    print("Random 16 bits:", bin(rng.randbits(16)))
    print("Choice from list:", rng.choice([10, 20, 30]))
    x = [1, 2, 3, 4, 5]
    rng.shuffle(x)
    print("Shuffled:", x)