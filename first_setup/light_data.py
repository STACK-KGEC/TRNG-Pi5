import smbus2
import time
import pandas as pd
from datetime import datetime

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


# === Initialize sensor ===
sensor = BH1750()

# === Data collection parameters ===
data = []
SAVE_INTERVAL = 10  # Save to CSV every 10 records
CSV_FILE = "light_data.csv"

# === Main loop ===
try:
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        lux = sensor.read_light()
        print(f"[{timestamp}] Light Level: {lux:.2f} lux")

        data.append({"timestamp": timestamp, "lux": lux})

        if len(data) >= SAVE_INTERVAL:
            df = pd.DataFrame(data)
            df.to_csv(CSV_FILE, mode='a', index=False, header=not pd.io.common.file_exists(CSV_FILE))
            data.clear()

        time.sleep(0.01)  # Fast sampling (~100 Hz)

except KeyboardInterrupt:
    print("\nExiting gracefully. Saving remaining data...")
    if data:
        df = pd.DataFrame(data)
        df.to_csv(CSV_FILE, mode='a', index=False, header=not pd.io.common.file_exists(CSV_FILE))
    print("All data saved.")
