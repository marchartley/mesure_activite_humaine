class SyndromException(Exception):
	pass

matHammingCode = [15, 51, 85, 255]
matHammingVerif = [1, 3, 5, 7, 9, 11, 13, 15]
matHammingDecode = [15, 2, 4, 0, 8, 0, 0, 0]

data_bytes = 4
coded_bytes = data_bytes * 2
data_bits = data_bytes * 4
coded_bits = coded_bytes * 4

def bit(byte, index):
	return (byte >> index) & 1

def display(byte, length):
	str = ""
	for i in range(0, length):
		if bit(byte, length - i - 1):
			str += "1"
		else:
			str += "0"
	return str

def hamming_distance(a, b):
	errors = 0
	for i in range(32):
		if bit(a, i) != bit(b, i):
			errors += 1
	return errors

def hamming_coding(byte) :
	full_res = 0
	for current_byte in range(data_bytes):
		res = 0
		for i in range(4):
			if bit(byte, data_bits - 1 - (i + 4 * current_byte)):
				res ^= matHammingCode[i]
		full_res = (full_res << 8) | res
	return full_res

def compute_syndrom(coding):
	full_res = 0
	for current_byte in range(coded_bytes//2):
		res = 0
		for i in range(8):
			if bit(coding, coded_bits - 1 - (i + 8 * current_byte)):
				res ^= matHammingVerif[i]
		full_res = (full_res << 8) | res
	return full_res

def error_mask(syndrom):
	mask = 0
	for current_byte in range(coded_bytes):
		subsyndrom = 0xF & (syndrom >> (4 * current_byte))
		if subsyndrom == 0:
			pass # Nothing to do
		else:
			if not bit(subsyndrom, 0):
				return 0xFFFF # At least 2 errors, the message cannot be found
			else:
				error_index = (0xF ^ (0xF & subsyndrom)) >> 1
				mask = mask | 1 << (error_index + 4 * current_byte)
	return mask

def correct_coding(coding, syndrom):
	mask = error_mask(syndrom)
	if mask == 0xFFFF:
		raise SyndromException(f"The coded value {coding} cannot be retrieved...")
	else:
		return coding ^ mask

def hamming_decode(byte):
	if byte == 0:
		return 0
	else:
		full_res = 0
		for current_byte in range(data_bytes):
			res = 0
			for i in range(8):
				if bit(byte, coded_bits - 1 - (i + 8 * current_byte)):
					res ^= matHammingDecode[i]
			full_res = (full_res << 4) | res
		return full_res
    
def desenterlace(coded, number_of_bits):
    a = 0
    b = 0
    for i in range(number_of_bits):
        a |= ((coded >> (2 * i + 1)) & 1) << i
        b |= ((coded >> (2 * i)) & 1) << i
    return a, b

if __name__ == '__main__':
	data = 31125
	changed = hamming_coding(data) ^ 1024 ^ 1
	print(display(data, 16))
	print("Normal:  ", display(hamming_coding(data), 32))
	print("Changed: ", display(changed, 32))
	syndrom_changed = compute_syndrom(changed)
	print("Good syn ", display(compute_syndrom(hamming_coding(data)), 32))
	print("Syndrom: ", display(syndrom_changed, 32))
	print("Mask:    ", display(error_mask(syndrom_changed), 32))
	print("Correct: ", display(correct_coding(changed, syndrom_changed), 32))
	print()
	print("Error    ", hamming_decode(correct_coding(changed, syndrom_changed)))
	print("Decoding ", display(hamming_decode(hamming_coding(data)), 16), " = ", int(hamming_decode(hamming_coding(data))))
	print("Original ", display(data, 16))