This directory contains mainly components of the reposity that have to do with
the dataset of interest. Because the data must be downloaded separately from the
project itself, the script `data.py` contains utilities to check hashes of each
downloaded file to verify that complete and correct copies have been downloaded.
In addition, it includes a utility to create a new hash dictionary but this
should not be necessary as a JSON file containing the hash dictionary for the
complete data set used in this project is already included.

When the dataset is downloaded, it is stored with this directory as a subfolder
by the name of `ds005`. This subdirectory first sorts the data by subject.
Within each subject's folder is then four more subdirectories:
- `anatomy`: Contains high-resolution images demonstrating the subject's unique
  brain anatomy.
- `behav`: Contains conditions and responses pertaining to the behavioral
  component of the study.
- `BOLD`: Contains raw fMRI images and quality assurance data on their validity
  and usefulness in blood oxygen-level dependent analysis.
- `model`: Contains filtered fMRI images and condition data from the behavioral
  component of the study that is used to produce neural time course predictions.
