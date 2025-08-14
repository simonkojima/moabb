import os
import re
import string
from pathlib import Path

import mne

from moabb.datasets import download as dl

from .base import BaseDataset


_manifest_link = "https://dataverse.harvard.edu/api/datasets/export"
_api_base_url = "https://dataverse.harvard.edu/api/access/datafile/"

_events_mappings = {
    "A": ["1", "101"],
    "B": ["2", "102"],
    "C": ["3", "103"],
    "D": ["4", "104"],
    "E": ["5", "105"],
    "F": ["6", "106"],
    "G": ["7", "107"],
    "H": ["8", "108"],
    "I": ["9", "109"],
    "J": ["10", "110"],
    "K": ["11", "111"],
    "L": ["12", "112"],
    "M": ["13", "113"],
    "N": ["14", "114"],
    "O": ["15", "115"],
    "P": ["16", "116"],
    "Q": ["17", "117"],
    "R": ["18", "118"],
    "S": ["19", "119"],
    "T": ["20", "120"],
    "U": ["21", "121"],
    "V": ["22", "122"],
    "W": ["23", "123"],
    "X": ["24", "124"],
    "Y": ["25", "125"],
    "Z": ["26", "126"],
    "SPACE": ["27", "127"],  # low
    "PERIOD": ["28", "128"],  # low
    "COMMA": ["29", "129"],  # mid
    "DELETE": ["30", "130"],  # low
}


def _extract_run_number(path):
    match = re.search(r"run-(\d+)", path.name)
    return int(match.group(1)) if match else -1


class Kojima2025(BaseDataset):
    """Class for Kojima2024B_2stream dataset management. P300 dataset.

    **Dataset description**

    This dataset [1]_ originates from a study investigating a four-class auditory BCI
    based on auditory stream segregation (ASME-BCI) [2]_.

    In the experiment, participants focused on one of four auditory streams, leveraging
    auditory stream segregation to selectively attend to stimuli in the target stream.
    Each stream contained a two-stimulus oddball sequence composed of one deviant
    stimulus and one standard stimulus.

    The current class corresponds to the "ASME-4stream" condition described in [2]_.
    For the "ASME-2stream" condition, see :class:`Kojima2024B_2stream`.

    The sequence below illustrates an example trial. For instance, when D3 is the target
    stimulus, the participant attended to Stream3 and selectively listened for D3.
    In this case, D3 is the target, and D1, D2, and D4 are considered non-target stimuli.

    .. code-block:: text

        Stream4  -------- S4 -------- S4 -------- D4 -------- S4 -------- S4 --
        Stream3  ----- S3 -------- S3 -------- S3 -------- D3 -------- S3 -----
        Stream2  -- S2 -------- S2 -------- D2 -------- S2 -------- S2 --------
        Stream1  S1 -------- D1 -------- S1 -------- S1 -------- S1 -----------

    Each participant completed 1 session consisting of 6 runs.
    Each run included 4 trials, each with a different target stimulus.
    In each trial, all deviant stimuli (D1--D4) were presented 15 times.

    Recording Detailes:
        - EEG signals were recorded using a BrainAmp system (Brain Products, Germany)
          at a sampling rate of 1000 Hz.

        - Data were collected in Tokyo, Japan, where the power line frequency is 50 Hz.

        - EEG was recorded from 64 scalp electrodes according to the international 10--20 system:
          Fp1, Fp2, AF7, AF3, AFz, AF4, AF8, F7, F5, F3, F1, Fz, F2, F4, F6, F8,
          FT9, FT7, FC5, FC3, FC1, FCz, FC2, FC4, FC6, FT8, FT10, T7, C5, C3, C1,
          Cz, C2, C4, C6, T8, TP9, TP7, CP5, CP3, CP1, CPz, CP2, CP4, CP6, TP8,
          TP10, P7, P5, P3, P1, Pz, P2, P4, P6, P8, PO7, PO3, POz, PO4, PO8,
          O1, Oz, O2

          EEG signals were referenced to the right mastoid and grounded to the left mastoid.

        - EOG was recorded using 2 electrodes (vEOG and hEOG), placed above/below and
          lateral to one eye.

    Parameters
    ----------

    keep_trial_structure : bool, default=False
        In MOABB, all classification tasks are performed as binary classification problems for P300 datasets.
        If you want to perform 4-class classification for each trial, set ``keep_trial_structure=True``.

        Note that this is only compatible with the :meth:`base.BaseDataset.get_data` method.
        It cannot be used with :class:`moabb.paradigms.base.BaseParadigm` or :class:`moabb.paradigms.P300`.
        To make it compatible with these, you need to provide an appropriate ``process_pipelines`` and ``postprocess_pipeline``
        argument to the :meth:`moabb.paradigms.base.BaseProcessing.get_data`,
        :meth:`moabb.evaluations.base.BaseEvaluation.evaluate` or
        :meth:`moabb.evaluations.base.BaseEvaluation.process` etc...

    References
    ----------

    .. [1] Kojima, S. (2024).
        Replication Data for: Four-class ASME BCI: investigation of the feasibility and comparison of two strategies for multiclassing.
        Harvard Dataverse, V1. DOI: https://doi.org/10.7910/DVN/1UJDV6
    .. [2] Kojima, S. & Kanoh, S. (2024).
        Four-class ASME BCI: investigation of the feasibility and comparison of two strategies for multiclassing.
        Frontiers in Human Neuroscience 18:1461960. DOI: https://doi.org/10.3389/fnhum.2024.1461960
    """

    def __init__(
        self,
        keep_trial_structure=False,
    ):

        self.subject_list = list(range(1, 11))
        self.n_channels = 64
        self.keep_trial_structure = keep_trial_structure

        super().__init__(
            self.subject_list,
            sessions_per_subject=1,
            events=dict(Target=1, NonTarget=0),
            code="Kojima2025",
            interval=[-0.5, 1.2],
            paradigm="p300",
            doi="10.7910/DVN/1UJDV6",
        )

    def _get_files_list(self, subject, manifest):

        subject_id = self.convert_subject_to_subject_id(subject)

        manifest_files = manifest["datasetVersion"]["files"]

        files_to_load = []

        for file in manifest_files:

            if (
                (f"sub-{subject_id}" not in file["label"])
                or ("stream_" not in file["label"])
                or ("_eeg" not in file["label"])
            ):
                continue

            fname = file["label"]
            directory = file["directoryLabel"]
            file_id = file["dataFile"]["id"]

            files_to_load.append(
                {"fname": fname, "directory": directory, "file_id": file_id}
            )

        return files_to_load

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            A dictionary containing the raw data for the subject.
        """

        # Get the file path for the subject's data
        files_path = self.data_path(subject)
        runs = {}
        for file in files_path:

            fname = file.name
            run_id = int(fname.split("_")[3].split("-")[1])
            task = fname.split("_")[2].split("-")[1]

            if task == "online":
                run_id += 6

            # raw = mne.io.read_raw_brainvision(file, eog=["vEOG", "hEOG"])
            raw = mne.io.read_raw_edf(file, eog=["vEOG", "hEOG"])
            raw = raw.load_data()

            raw = raw.set_montage("standard_1020")

            if self.keep_trial_structure:
                try:
                    import tag_mne as tm
                except ImportError:
                    raise ImportError(
                        "Package tag_mne is required when keep_trial_structure=True. Install it with pip install tag-mne"
                    )

                events, event_id = mne.events_from_annotations(raw)

                """
                bv_to_marker_mapping = {}
                for key in list(event_id.keys()):
                    bv_to_marker_mapping[key] = str(int(key.split("/")[1][1:]))

                raw.annotations.rename(bv_to_marker_mapping)
                events, event_id = mne.events_from_annotations(raw)
                """

                samples, markers = tm.markers_from_events(events, event_id)

                markers = tm.add_event_names(markers, _events_mappings)
                markers = tm.add_tag(markers, f"task:{task}")
                markers = tm.add_tag(markers, f"run:{run_id}")
                markers = tm.split_trials(
                    markers, trial=[str(marker) for marker in range(201, 231)]
                )

                markers = tm.add_tag_to_markers(
                    markers,
                    Target=[str(marker) for marker in range(101, 131)],
                    NonTarget=[str(marker) for marker in range(1, 31)],
                )

                events, event_id = tm.events_from_markers(samples, markers)

                event_desc = {v: k for k, v in event_id.items()}

                annotations = mne.annotations_from_events(
                    events=events, sfreq=raw.info["sfreq"], event_desc=event_desc
                )

                raw = raw.set_annotations(annotations)

            else:

                events, event_id = mne.events_from_annotations(raw)

                annotations_mapping = {}
                for key, value in event_id.items():
                    if (int(key) >= 101) and (int(key) <= 130):
                        annotations_mapping[key] = "Target"
                    if (int(key) >= 1) and (int(key) <= 30):
                        annotations_mapping[key] = "NonTarget"

                raw.annotations.rename(annotations_mapping)

            runs.update({f"{run_id}{task}": raw})

        sessions = {"0": runs}

        return sessions

    def convert_subject_to_subject_id(self, subjects):
        """
        Convert subject number(s) to subject ID(s).
        (In this dataset, subject IDs are encoded using alphabet letters.)

        Parameters
        ----------
        subjects : int or list of int
            Subject number(s) to convert.

        Returns
        -------
        subject_id : str or list of str
            Converted subject ID(s).
        """

        if isinstance(subjects, int):
            subject_id = list(string.ascii_uppercase)[subjects - 1]
        elif isinstance(subjects, list):
            subject_id = []
            for subject in subjects:
                subject_id.append(list(string.ascii_uppercase)[subject - 1])
        else:
            raise TypeError("Type of subejcts must be either int or list.")

        return subject_id

    def data_path(self, subject, path=None):
        """
        Return the data paths of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
            If it doesn’t exist, the “~/mne_data” directory is used. If the
            dataset is not found under the given path, the data
            will be automatically downloaded to the specified folder.

        Returns
        -------
        list
            A list containing the Path object for the subject's data file.
        """

        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        # Download and extract the dataset
        dataset_path = self.download_by_subject(subject=subject, path=path)

        files = os.listdir(dataset_path / f"sub-{subject:02d}" / "ses-01" / "eeg")

        paths = []
        for file in files:
            if file.endswith(".edf") and (("online" in file) or ("offline" in file)):
                paths.append(
                    dataset_path / f"sub-{subject:02d}" / "ses-01" / "eeg" / file
                )

        paths = sorted(paths, key=_extract_run_number)

        return paths

    def download_by_subject(self, subject, path=None):
        """
        Download and extract the dataset.

        Parameters
        ----------
        subject : int
            The subject number to download the dataset for.

        path : str | None
            The path to the directory where the dataset should be downloaded.
            If None, the default directory is used.


        Returns
        -------
        path : str
            The dataset path.
        """

        path = Path(dl.get_dataset_path(self.code, path)) / (f"MNE-{self.code}-data")

        """
        # checking it there is manifest file in the dataset folder.
        dl.download_if_missing(path / "kojima2024_manifest.json", _manifest_link)

        with open(path / "kojima2024_manifest.json", "r") as f:
            manifest = json.load(f)

        files = self._get_files_list(subject, manifest)

        for file in tqdm(files):
            download_url = _api_base_url + str(file["file_id"])
            dl.download_if_missing(
                path / file["directory"] / file["fname"],
                download_url,
                warn_missing=False,
            )
        """

        return path
