from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

class run(object):
    """
    This class allows organization of the data by runs. Methods attached perform
    the indicated analyses of the data.
    """

    def __init__(self, sub_id, run_id, binary_resp=True, rm_nonresp=True):
        """
        Each object of this class created contains the fMRI data along with the
        corresponding behavioral data.

        Parameters
        ----------
        sub_id : str
            The unique key used to identify the subject (i.e., 001, ..., 016)
        run_id : str
            The unique key used to identify the run number (i.e, 001, ..., 003)
        binary_resp : bool
            True converts the subject's responses to binary (i.e., 1 for any
            flavor of agreement, 0 for any flavor of declination)
        rm_nonresp : bool
            True removes trials that resulted in subject nonresponse
        """
        # Save the path to the directory containing the subject's data.
        path_data = "../data/ds005/sub%s/" % (sub_id,)

        # Extract subject's BOLD signal data for the specified run.
        path_BOLD = path_data + "BOLD/task001_run%s/bold.nii.gz" % (run_id,)
        self.data = nib.load(path_BOLD).get_data()

        # Extract subject's behavioral data for the specified run.
        path_behav = path_data + "behav/task001_run%s/behavdata.txt" % (run_id,)
        # Read in all but the first line, which is a just a header.
        raw = np.array([row.split() for row in list(open(path_behav))[1:]])
        kept_rows = raw[:, 4] != "0" if rm_nonresp else ...
        rare = raw[kept_rows].astype("float")
        resp_col = 4 if binary_resp else 5
        (onset, gain, loss, resp) = (rare[:, 0], rare[:, 1], rare[:, 2],
            rare[:, resp_col])
        # Volumes are captured every two seconds.
        self.behav = np.array([onset // 2, gain, loss, resp], dtype=int).T
