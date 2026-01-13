import sys

def convert_ascii_to_binary(ascii_filename, binary_filename):
    with open(ascii_filename, 'r') as f_in:
        # Read the entire file content, assuming it's just '0's and '1's
        data = f_in.read().replace('\n', '').replace('\r', '')
    
    with open(binary_filename, 'wb') as f_out:
        for bit_char in data:
            # Convert '0' or '1' char to a single byte (0x00 or 0x01)
            byte_val = 0x01 if bit_char == '1' else 0x00
            f_out.write(bytes([byte_val]))

    print(f"Converted {len(data)} bits from {ascii_filename} to {binary_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_script.py input_ascii_file.txt output_binary_file.bin")
    else:
        convert_ascii_to_binary(sys.argv[1], sys.argv[2])