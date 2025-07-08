import smbus2
import time
import socket
import pickle
import struct
from tqdm.auto import tqdm

class BH1750:
    DEVICE = 0x23  # Default I2C address
    POWER_ON = 0x01
    RESET = 0x07
    CONTINUOUS_HIGH_RES_MODE = 0x10

    def __init__(self, bus=1, address=DEVICE):
        self.bus = smbus2.SMBus(bus)
        self.address = address
        self.bus.write_byte(self.address, self.POWER_ON)
        time.sleep(0.05)
        self.bus.write_byte(self.address, self.RESET)
        time.sleep(0.05)

    def read_light(self):
        self.bus.write_byte(self.address, self.CONTINUOUS_HIGH_RES_MODE)
        time.sleep(0.18)  # Wait for high-res measurement
        data = self.bus.read_i2c_block_data(self.address, 0x00, 2)
        result = (data[0] << 8) + data[1]
        return result / 1.2  # Convert to lux

sensor = BH1750()

def generate_raw_data():
    pi_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    pi_socket.bind(("0.0.0.0", 12345))  # Listen on all interfaces
    pi_socket.listen(1)
    print("Server listening...")
    delay = 0.02
    while True:
        
        client, addr = pi_socket.accept()
        print("Connected by", addr)
        
        while True:
        
            data = client.recv(4)
            if not data:
                break
            n = struct.unpack('!I', data)[0]  # Receive 4 bytes as unsigned int (network byte order)
            print(f"Generating {n} Light Data....")
            readings = []
            for i in tqdm(range(n)):
                readings.append(sensor.read_light())
                time.sleep(delay)
            serialized = pickle.dumps(readings)
            print(f"Sending The Data With {delay} Delay....")
            client.sendall(serialized)
        
        
        client.close()
        
generate_raw_data()