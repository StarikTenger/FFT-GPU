import sys

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_entries = int(lines[0].strip())
        data = [tuple(map(float, line.strip().split())) for line in lines[1:num_entries+1]]
    return data

def compare_files(file1, file2, epsilon=1e-6):
    data1 = read_file(file1)
    data2 = read_file(file2)
    
    if len(data1) != len(data2):
        return False, "Number of entries differ between files"
    
    for i, ((a1, b1), (a2, b2)) in enumerate(zip(data1, data2)):
        if abs(a1 - a2) > epsilon or abs(b1 - b2) > epsilon:
            return False, f"Difference found at line {i+2}: ({a1}, {b1}) != ({a2}, {b2})"
    
    return True, "Result are identical with precision of " + str(epsilon)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare.py <file1> <file2>")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    result, message = compare_files(file1, file2, 1e-3)
    print(message)