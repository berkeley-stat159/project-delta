import data, json, os

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
    hash_dict = {path: data.generate_file_md5(path) for path in paths}
    return hash_dict

if __name__ == "__main__":
    hash_dict = make_hash_dict("ds005")
    with open("ds005_hashes.json", "w") as outfile:
        json.dump(hash_dict, outfile, indent = 4)