import socket
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt 
import pickle
import struct


HOST = "192.168.43.223"  # Your Pi's IP
PORT = 12345

def request_data(len):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    
    client_socket.send(struct.pack('!I', len))  # Send 4 bytes for integer
    data = b""
    print(f"Requesting Readings....\nTotal Size = {len}")
    while True:
        chunk = client_socket.recv(4096)
        if not chunk:
            break
        data += chunk
        readings = np.array(pickle.loads(data))
        break
        
    print(f"Received readings....\nTotal Size = {readings.shape}")
    
    return readings
    
# request_data(len=10)