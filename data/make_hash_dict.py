import data, json, os, sys

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
    paths = [os.path.join(root, files) for root, dirs, files in os.walk(top)]
    # generate_file_md5() takes as input a file path and outputs its hash
    hash_dict = [data.generate_file_md5(path) for path in paths]
    return hash_dict