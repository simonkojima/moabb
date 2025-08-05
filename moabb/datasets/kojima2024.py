"""
URL PATH: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1UJDV6

check the following note for development
https://github.com/NeuroTechX/moabb/blob/master/CONTRIBUTING.md

"""

import json
import os
import string
import warnings
from pathlib import Path

import mne

# from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
from tqdm import tqdm

from moabb.datasets import download as dl

from .base import BaseDataset


_manifest_link = "https://dataverse.harvard.edu/api/datasets/export?exporter=dataverse_json&persistentId=doi%3A10.7910/DVN/1UJDV6"
# _manifet_link = "https://dataverse.harvard.edu/api/datasets/export?exporter=dataverse_json&persistentId=doi%3A10.7910/DVN/1UJDV6"
# _metainfo_link = "https://osf.io/download/67c9e8234f014fc76e0411ba/"

# _osf_tag = "8tdk5"
# _api_base_url = f"https://files.de-1.osf.io/v1/resources/{_osf_tag}/providers/osfstorage/"
_api_base_url = "https://dataverse.harvard.edu/api/access/datafile/"


class _Kojima2024Base(BaseDataset):
    """
    Parent class of Kojima2024_2 and Kojima2024_4
    Should not be instantiated.
    """

    def __init__(self, task):

        self.task = task

        # self.subject_list = subjects
        self.subject_list = list(range(1, 16))
        self.n_channels = 64

        super().__init__(
            self.subject_list,
            sessions_per_subject=1,
            events=dict(Target=1, NonTarget=0),
            code="Kojima2024",
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

        print("_get_single_subject_data was called")

        # Get the file path for the subject's data
        files_path = self.data_path(subject)
        print(files_path)
        runs = {}
        for file in files_path:

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                print(file)

                run_id = file.name.split("_")[2].split("-")[1]
                print(self.task)
                task = f"{self.task}stream"
                print(task)

                raw = mne.io.read_raw_brainvision(file, eog=["vEOG", "hEOG"])
                raw = raw.load_data()

                raw = raw.set_montage("standard_1020")

                raw.annotations.rename(
                    {
                        "Stimulus/S111": "Target",
                        "Stimulus/S112": "Target",
                        "Stimulus/S113": "Target",
                        "Stimulus/S114": "Target",
                        "Stimulus/S101": "NonTarget",
                        "Stimulus/S102": "NonTarget",
                        "Stimulus/S103": "NonTarget",
                        "Stimulus/S104": "NonTarget",
                    }
                )

                # exit()
                # Read the subject's raw data and set the montage
                # raw = read_raw_bids(bids_path=file, verbose=False)
                # raw = raw.load_data()
                # We are losting several annotations because there is no fuck
                # place explaining what it is the events ids :)
                # raw.annotations.rename({"769": "left_hand", "770": "right_hand"})
                runs.update({f"{run_id}{task}": raw})

        sessions = {"0": runs}

        return sessions

    def convert_subject_to_subject_id(self, subjects):

        if isinstance(subjects, int):
            # 1 -> A, 2 -> B, etc...
            subject_id = list(string.ascii_uppercase)[subjects - 1]
        elif isinstance(subjects, list):
            subject_id = []
            for subject in subjects:
                subject_id.append(list(string.ascii_uppercase)[subject - 1])
        else:
            raise TypeError("Type of subejcts must be either int or list.")

        return subject_id

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """
        Return the data BIDS paths of a single subject.

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
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python config
            to the given path.
            If None, the user is prompted.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose()).

        Returns
        -------
        list
            A list containing the BIDSPath object for the subject's data file.
        """

        print("data_path was called")

        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        # Download and extract the dataset
        dataset_path = self.download_by_subject(subject=subject, path=path)

        subject_id = self.convert_subject_to_subject_id(subject)

        files = os.listdir(dataset_path / f"sub-{subject_id}" / "eeg")

        paths = []
        for file in files:
            if file.endswith(".vhdr") and f"task-{self.task}stream" in file:
                paths.append(dataset_path / f"sub-{subject_id}" / "eeg" / file)

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

        print("download_by_subject was called")

        path = Path(dl.get_dataset_path(self.code, path)) / (f"MNE-{self.code}-data")

        # checking it there is manifest file in the dataset folder.
        dl.download_if_missing(path / "kojima2024_manifest.json", _manifest_link)

        with open(path / "kojima2024_manifest.json", "r") as f:
            manifest = json.load(f)

        files = self._get_files_list(subject, manifest)

        print(files)

        for file in tqdm(files):
            download_url = _api_base_url + str(file["file_id"])
            dl.download_if_missing(
                path / file["directory"] / file["fname"],
                download_url,
                warn_missing=False,
            )

        return path


class Kojima2024_2(_Kojima2024Base):
    """Class for Dreyer2023A dataset management. MI dataset.

    **Dataset description**

    "A large EEG database with users' profile information for motor imagery
    Brain-Computer Interface research" [1]_ [2]_

    :Data collectors: Appriou Aurélien; Caselli Damien; Benaroch Camille;
                      Yamamoto Sayu Maria; Roc Aline; Lotte Fabien;
                      Dreyer Pauline; Pillette Léa
    :Data manager: Dreyer Pauline
    :Project leader: Lotte Fabien
    :Project members: Rimbert Sébastien; Monseigne Thibaut

    Dataset Dreyer2023A contains EEG, EOG and EMG signals recorded on 60 healthy subjects
    performing Left-Right Motor Imagery experiments
    (29 women, age 19-59, M = 29, SD = 9.32) [1]_.
    Experiments were conducted by six experimenters. In addition, for each recording
    the following pieces of information are provided:
    subject's demographic, personality and cognitive profiles, the OpenViBE experimental
    instructions and codes, and experimenter's gender.

    The experiment is designed for the investigation of the impact of the participant's
    and experimenter's gender on MI BCI performance [1]_.

    A recording contains open and closed eyes baseline recordings and 6 runs of the MI
    experiments. First 2 runs (acquisition runs) were used to train system and
    the following 4 runs (training runs) to train the participant. Each run contained
    40 trials [1]_.

    Each trial was recorded as follows [1]_:
        - t=0.00s  cross displayed on screen
        - t=2.00s  acoustic signal announced appearance of a red arrow
        - t=3.00s  a red arrow appears (subject starts to perform task)
        - t=4.25s  the red arrow disappears
        - t=4.25s  the feedback on performance is given in form of a blue bar
          with update frequency of 16 Hz
        - t=8.00s  cross turns off (subject stops to perform task)

    EEG signals [1]_:
        - recorded with 27 electrodes, namely:
          Fz, FCz, Cz, CPz, Pz, C1, C3, C5, C2, C4, C6, F4, FC2, FC4, FC6, CP2,
          CP4, CP6, P4, F3, FC1, FC3, FC5, CP1, CP3, CP5, P3 (10-20 system),
          referenced to the left earlobe.

    EOG signals [1]_:
        - recorded with 3 electrodes, namely: EOG1, EOG2, EOG3
          placed below, above and on the side of one eye.

    EMG signals [1]_:
        - recorded with 2 electrodes, namely: EMGg, EMGd
          placed 2.5cm below the skinfold on each wrist.

    Demographic and biosocial information includes:
        - gender, birth year, laterality
        - vision, vision assistance
        - familiarity to cognitive science or neurology, level of education
        - physical activity, meditation
        - attentional, neurological, psychiatrics symptoms

    Personality and the cognitive profile [1]_:
        - evaluated via 5th edition of the 16 Personality Factors (16PF5) test
        - and mental rotation test
        - index of learning style

    Pre and post experiment questionnaires [1]_:
        - evaluation of pre and post mood, mindfulness and motivational states

    The online OpenViBE BCI classification performance [1]_:
        - only performance measure used to give the feedback to the participants

    * Subject 59 contains only 4 runs

    References
    ----------

    .. [1] Pillette, L., Roc, A., N’kaoua, B., & Lotte, F. (2021).
        Experimenters' influence on mental-imagery based brain-computer interface user training.
        International Journal of Human-Computer Studies, 149, 102603.
    .. [2] Benaroch, C., Yamamoto, M. S., Roc, A., Dreyer, P., Jeunet, C., & Lotte, F. (2022).
        When should MI-BCI feature optimization include prior knowledge, and which one?.
        Brain-Computer Interfaces, 9(2), 115-128.
    """

    def __init__(self):
        super().__init__(task="2")


class Kojima2024_4(_Kojima2024Base):
    """Class for Dreyer2023B dataset management. MI dataset.

    **Dataset description**

    "A large EEG database with users' profile information for motor imagery
    Brain-Computer Interface research" [1]_ [2]_

    :Data collectors: Appriou Aurélien; Caselli Damien; Benaroch Camille;
                      Yamamoto Sayu Maria; Roc Aline; Lotte Fabien;
                      Dreyer Pauline; Pillette Léa
    :Data manager: Dreyer Pauline
    :Project leader: Lotte Fabien
    :Project members: Rimbert Sébastien; Monseigne Thibaut

    Dataset Dreyer2023B contains EEG, EOG and EMG signals recorded on 21 healthy subjects
    performing Left-Right Motor Imagery experiments
    (8 women, age 19-37, M = 29, SD = 9.318) [2]_.
    Experiments were conducted by female experimenters. In addition, for each recording
    the following pieces of information are provided:
    subject's demographic, personality and cognitive profiles, the OpenViBE experimental
    instructions and codes, and experimenter's gender.

    The experiment is designed for the investigation of the relation between MI-BCI online
    performance and Most Discriminant Frequency Band (MDFB) [2]_.

    A recording contains open and closed eyes baseline recordings and 6 runs of the MI
    experiments. First 2 runs (acquisition runs) were used to train system and
    the following 4 runs (training runs) to train the participant. Each run contained
    40 trials [1]_.

    Each trial was recorded as follows [1]_:
        - t=0.00s  cross displayed on screen
        - t=2.00s  acoustic signal announced appearance of a red arrow
        - t=3.00s  a red arrow appears (subject starts to perform task)
        - t=4.25s  the red arrow disappears
        - t=4.25s  the feedback on performance is given in form of a blue bar
          with update frequency of 16 Hz
        - t=8.00s  cross turns off (subject stops to perform task)

    EEG signals [1]_:
        - recorded with 27 electrodes, namely:
          Fz, FCz, Cz, CPz, Pz, C1, C3, C5, C2, C4, C6, F4, FC2, FC4, FC6, CP2,
          CP4, CP6, P4, F3, FC1, FC3, FC5, CP1, CP3, CP5, P3 (10-20 system),
          referenced to the left earlobe.

    EOG signals [1]_:
        - recorded with 3 electrodes, namely: EOG1, EOG2, EOG3
          placed below, above and on the side of one eye.

    EMG signals [1]_:
        - recorded with 2 electrodes, namely: EMGg, EMGd
          placed 2.5cm below the skinfold on each wrist.

    Demographic and biosocial information includes:
        - gender, birth year, laterality
        - vision, vision assistance
        - familiarity to cognitive science or neurology, level of education
        - physical activity, meditation
        - attentional, neurological, psychiatrics symptoms

    Personality and the cognitive profile [1]_:
        - evaluated via 5th edition of the 16 Personality Factors (16PF5) test
        - and mental rotation test
        - index of learning style

    Pre and post experiment questionnaires [1]_:
        - evaluation of pre and post mood, mindfulness and motivational states

    The online OpenViBE BCI classification performance [1]_:
        - only performance measure used to give the feedback to the participants


    References
    ----------

    .. [1] Pillette, L., Roc, A., N’kaoua, B., & Lotte, F. (2021).
        Experimenters' influence on mental-imagery based brain-computer interface user training.
        International Journal of Human-Computer Studies, 149, 102603.
    .. [2] Benaroch, C., Yamamoto, M. S., Roc, A., Dreyer, P., Jeunet, C., & Lotte, F. (2022).
        When should MI-BCI feature optimization include prior knowledge, and which one?.
        Brain-Computer Interfaces, 9(2), 115-128.
    """

    def __init__(self):
        super().__init__(task="4")
