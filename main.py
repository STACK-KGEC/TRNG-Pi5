
print("HELLO WORLD")

import smbus2
import time

class BH1750:
    DEVICE = 0x23  # I2C address from i2cdetect
    POWER_ON = 0x01
    RESET = 0x07
    CONTINUOUS_HIGH_RES_MODE = 0x10

    def __init__(self, bus=1, address=DEVICE):
        self.bus = smbus2.SMBus(bus)
        self.address = address

        # Power on and reset the sensor
        self.bus.write_byte(self.address, self.POWER_ON)
        time.sleep(0.1)
        self.bus.write_byte(self.address, self.RESET)
        time.sleep(0.1)

    def read_light(self):
        # Start measurement
        self.bus.write_byte(self.address, self.CONTINUOUS_HIGH_RES_MODE)
        time.sleep(0.2)  # Wait for measurement

        # Read 2 bytes of data
        data = self.bus.read_i2c_block_data(self.address, 0x00, 2)
        result = (data[0] << 8) + data[1]
        return result / 1.2  # Convert to lux

sensor = BH1750(bus=1, address=0x23)

while True:
    lux = sensor.read_light()
    print(f"Light Level: {lux:.2f} lux")
    time.sleep(1)
