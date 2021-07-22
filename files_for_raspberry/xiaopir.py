# ADXL345 Python library for Raspberry Pi
#
# author:  Jonathan Williamson
# license: BSD, see LICENSE.txt included in this package
#
# This is a Raspberry Pi Python implementation to help you get started with
# the Adafruit Triple Axis ADXL345 breakout board:
# http://shop.pimoroni.com/products/adafruit-triple-axis-accelerometer

import smbus
#from time import sleep

# Hamming coding to check/correct PIR values
import hamming_coding as hamming
import time

# select the correct i2c bus for this revision of Raspberry Pi
with open('/proc/cpuinfo', 'r') as file:
    revision = ([l[12:-1] for l in file.readlines() if l[:8]=="Revision"]+['0000'])[0]
bus = smbus.SMBus(1 if int(revision, 16) >= 4 else 0)


PIR_I2C_ADDR        = 0x20

nb_err = 0

class XIAOPIR:

    address = None

    def __init__(self, address = PIR_I2C_ADDR):
        self.address = address
        self.enableMeasurement()

    def enableMeasurement(self):
        bus.write_byte(self.address, 10)

    def getVal(self):
        # Get the 2 bytes sent
        bytes = bus.read_i2c_block_data(self.address, 0, 2)
        # Merge those bytes to create a 16bit integer
        merged_bytes = bytes[1]+bytes[0]*256
        
        b1, b2 = hamming.desenterlace(merged_bytes, 8)
        original = (b1 << 8) | b2
        correction = hamming.correct_coding(original, hamming.compute_syndrom(original))
        merged_bytes = hamming.hamming_decode(correction)
        merged_bytes = merged_bytes << 8
        if correction & 0xFFFF == 0xFFFF :
            return None # raise hamming.SyndromException(f'correction was {hamming.display(correction, 16)}')
#        if merged_bytes > 20000:
#            print(merged_bytes, hamming.display(correction, 16), original ^ correction, hamming.display(hamming.compute_syndrom(correction), 16), hamming.display(hamming.compute_syndrom(original), 16))
#        return merged_bytes
#        # Or get the 4 bytes sent
#        bytes = bus.read_i2c_block_data(self.address, 0, 4)
#        # Merge those bytes to create a 32bit integer
#        merged_bytes = bytes[0]+bytes[1]*256 + bytes[2] * 256*256 + bytes[3] * 256*256*256
##        if bytes[0] == 255 or bytes[1] == 255 or bytes[2] == 255 or bytes[3] == 255:
##            raise Exception('err')
#        # Decode this data with Hamming decoding
#        original = merged_bytes
#        if original == 0xFFFF:
#            return -1
#        syndrom = hamming.compute_syndrom(merged_bytes)
#        if merged_bytes != 90:
#            print(hamming.hamming_distance(90, merged_bytes), "\t", hamming.hamming_decode(merged_bytes), "\t", bytes)
#        mask = hamming.error_mask(syndrom)
#        merged_bytes = hamming.correct_coding(merged_bytes, syndrom)
#        new_mask = hamming.error_mask(syndrom)
#        merged_bytes = hamming.hamming_decode(merged_bytes)
#        if merged_bytes != 10:
#            global nb_err
#            nb_err += 1
#            print(f"{nb_err} {hamming.hamming_distance(0, mask)} errors : {merged_bytes}\tNew syndrom = {syndrom}\tSent: {hamming.display(original, 16)}")
#        return merged_bytes
#        print(bytes) 
#        if merged_bytes == 65535 or bytes[0] == 255 or bytes[1] == 255:
#            raise Exception('err')
#        merged_bytes = hamming.hamming_decode(merged_bytes)
        return merged_bytes
    
if __name__ == "__main__":
    pass
