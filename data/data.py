from __future__ import absolute_import, print_function, division
import hashlib, json, os


def generate_file_md5(filename, blocksize=2**20):
    m = hashlib.md5()
    with open(filename, "rb") as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            m.update(buf)
    return m.hexdigest()


def check_hashes(d):
    all_good = True
    for k, v in d.items():
        digest = generate_file_md5(k)
        if v == digest:
            print("The file {0} has the correct hash.".format(k))
        else:
            print("ERROR: The file {0} has the WRONG hash!".format(k))
            all_good = False
    return all_good


def make_hash_dict(top):
    """
    Returns a dictionary with file paths and corresponding hashes.
    
    Parameters
    ----------
    data : str
        The path to the directory containing files needing hashes.

    Returns
    -------
    hash_dict : dict
        Dictionary with file paths as keys and hashes as values.
    """
    paths = []
    for root, dirs, files in os.walk(top):
        paths.extend(os.path.join(root, file) for file in files)
    # generate_file_md5() takes as input a file path and outputs its hash
    hash_dict = {path: generate_file_md5(path) for path in paths}
    with open(top + "_hashes.json", "w") as hashes:
        json.dump(hash_dict, hashes, indent = 4)
    return hash_dict


if __name__ == "__main__":
    with open("data/ds005_hashes.json", "r") as hash_dict:
        d = json.load(hash_dict)
    check_hashes(d)
