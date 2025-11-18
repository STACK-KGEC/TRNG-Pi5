import socket
import struct
import tqdm.auto as tqdm

HOST = ''
PORT = 5001

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"[*] Listening on port {PORT}...")
    
    conn, addr = s.accept()
    print(f"[+] Connected by {addr}")

    with conn, open("readings_1.txt", "w") as f:
        buffer = b""
        for _ in tqdm.tqdm(range(62500)):
            data = conn.recv(1024)
            if not data:
                break
            buffer += data

            while len(buffer) >= 2:
                value = struct.unpack('>H', buffer[:2])[0]
                binary_str = format(value, '016b')  # Convert to 16-bit binary
                f.write(binary_str)
                print(f"[RECEIVED] {value} -> {binary_str}")
                buffer = buffer[2:]

    print("[*] Connection closed.")