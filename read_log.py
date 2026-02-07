
import sys

def read_log(filename):
    encodings = ['utf-8', 'utf-16', 'utf-16le', 'cp1252', 'gbk']
    content = ""
    for enc in encodings:
        try:
            with open(filename, 'r', encoding=enc) as f:
                content = f.read()
            print(f"Successfully read with {enc}")
            break
        except Exception:
            continue
            
    if not content:
        print("Failed to read file with any encoding.")
        return

    # Print first 100 lines
    for line_idx, line in enumerate(content.splitlines()):
        if line_idx < 100:
            print(line)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        read_log(sys.argv[1])
