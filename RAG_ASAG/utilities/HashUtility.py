import hashlib
import os

digest_alg ="SHA256"
def write_hash_to_file(out_file_path, file_path,hash_value):
    file_name = os.path.basename(file_path)
    prefix =f"{digest_alg}({file_name})= "
    with open(out_file_path, 'w') as f:
        f.write(prefix + hash_value)
        f.close()

def hash_file_calc(file_path):
    mach = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            mach.update(chunk)
    return mach.hexdigest()


def calculate_and_write_hash_to_file(file_path, out_file_path):
    hash_digest = hash_file_calc(file_path)
    write_hash_to_file(out_file_path, file_path,  hash_digest)