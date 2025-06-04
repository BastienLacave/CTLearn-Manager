"""
CTLearnModelManager Class.

This class is designed to manage CTLearn models, providing functionalities for initializing, saving, loading, training, and updating model parameters. It also includes methods for handling training data, testing data, DL2 data, and IRF data.
"""

import ast
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import QTable

from ctlearn_manager.utils.utils import (
    ClusterConfiguration,
    CTLearnManagerStyle,
    Cuts,
    DataSample,
    IRFType,
    ParticleType,
    get_irf_type_from_config,
    remove_row_from_table_utils,
    set_mpl_style,
    CTLMDirectories,
)

# from ctlearn_manager.utils.index_tables import IndexTables

__all__ = [
    "CTLearnModelManager",
    "DataSample",
    "ModelRangeOfValidity",
]


class CTLearnModelManager:
    """
    CTLearnModelManager class for managing CTLearn models.

    This class provides methods for initializing, saving, loading, and training CTLearn models. It also includes methods for updating and retrieving model parameters, training data, testing data, DL2 data, and IRF data.

    Attributes
    ----------
    model_index_file : str
        Path to the model index file.
    model_nickname : str
        Nickname of the model.
    model_parameters_table : astropy.table.Table
        Table containing model parameters.
    validity : ModelRangeOfValidity
        Range of validity for the model.
    stereo : bool
        Indicates if the model uses stereo mode.
    telescope_ids : list
        List of telescope IDs used in the model.
    telescope_names : list
        List of telescope names used in the model.
    cluster_configuration : ClusterConfiguration
        Configuration for the cluster.

    Methods
    -------
    __init__(model_parameters, MODEL_INDEX_FILE, load=False, cluster_configuration=ClusterConfiguration())
        Initializes the ModelManager instance.
    save_to_index(model_parameters)
        Save model parameters and training samples to an HDF5 index file.
    launch_training(n_epochs, transfer_learning_model_cpk=None, frozen_backbone=False, config_file=None)
        Launches the training process for the model.
    get_n_epoch_trained()
        Calculate the total number of epochs trained by summing the lengths of all training logs.
    plot_loss()
        Plots the training and validation loss over epochs.
    update_model_manager_parameters_in_index(parameters)
        Update the model manager parameters in the HDF5 index file.
    update_model_manager_testing_data(testing_gamma_dirs, testing_proton_dirs, testing_gamma_zenith_distances, testing_gamma_azimuths, testing_proton_zenith_distances, testing_proton_azimuths, testing_gamma_patterns, testing_proton_patterns)
        Update the model manager's testing data for gamma and proton events.
    update_model_manager_DL2_MC_files(testing_DL2_gamma_files, testing_DL2_proton_files, testing_DL2_gamma_zenith_distances, testing_DL2_gamma_azimuths, testing_DL2_proton_zenith_distances, testing_DL2_proton_azimuths)
        Update the DL2 MC files for gamma and proton testing data in the model manager.
    update_model_manager_DL2_data_files(DL2_files, DL2_zenith_distances, DL2_azimuths)
        Update the DL2 data files for the model manager.
    update_merged_DL2_MC_files(testing_DL2_zenith_distance, testing_DL2_azimuth, testing_DL2_gamma_merged_file=None, testing_DL2_proton_merged_file=None)
        Update the merged DL2 MC files for gamma and proton data.
    update_model_manager_IRF_data(config, cuts_file, irf_file, bencmark_file, zenith, azimuth)
        Update the IRF (Instrument Response Function) data for the model manager.
    get_IRF_data(zenith, azimuth)
        Retrieve the Instrument Response Function (IRF) data for a given zenith and azimuth.
    get_closest_IRF_data(zenith, azimuth)
        Retrieve the closest Instrument Response Function (IRF) data based on the given zenith and azimuth angles.
    get_DL2_MC_files(zenith, azimuth)
        Retrieve DL2 Monte Carlo (MC) files for given zenith and azimuth angles.
    plot_zenith_azimuth_ranges(ax=None)
        Plot the zenith and azimuth ranges on a polar plot.
    plot_training_nodes()
        Plot the training nodes for gamma and proton events on a polar plot.
    """

    def __init__(
        self,
        model_parameters,
        # MODEL_INDEX_FILE,
        project_directories: CTLMDirectories,
        load=False,
        cluster_configuration=ClusterConfiguration(),
    ):
        """
        Initialize the ModelManager class.

        Parameters
        ----------
        model_parameters : dict
            Dictionary containing the parameters for the model. Must include at least
            the key "model_nickname" if a specific nickname is desired.
        MODEL_INDEX_FILE : str
            Path to the HDF5 file containing the model index.
        load : bool, optional
            If True, the model is loaded from the index file. If False, the model
            parameters are saved to the index file. Default is False.
        cluster_configuration : ClusterConfiguration, optional
            Configuration object for the cluster. Default is an instance of
            `ClusterConfiguration`.

        Raises
        ------
        ValueError
            If the model is of type "reco" and the required training patterns for
            gamma diffuse or proton are missing.
        ValueError
            If stereo mode is enabled but fewer than 2 telescopes are provided.
        ValueError
            If the model is of type "reco" with "cameradirection" and stereo mode
            is enabled, as this combination is not supported.
        """
        from astropy.io.misc.hdf5 import read_table_hdf5

        # self.project_directories.model_index_file = MODEL_INDEX_FILE
        self.model_nickname = model_parameters.get("model_nickname", "new_model")
        self.project_directories = project_directories
        if not load:
            self.save_to_index(model_parameters)
            print(f"🧠 Model name: {self.model_nickname}")
        
        self.model_parameters_table = read_table_hdf5(
            f"{self.project_directories.model_index_file}", path=IndexTables(self).PARAMETERS.table_path
        )
        self.validity = ModelRangeOfValidity(self)
        self.telescope_ids = ast.literal_eval(
            self.model_parameters_table["telescope_ids"][0]
        )
        self.telescope_names = ast.literal_eval(
            self.model_parameters_table["telescope_names"][0]
        )
        try:
            self.min_telescopes = self.model_parameters_table["min_telescopes"][0]
        except:
            self.min_telescopes = len(self.telescope_ids)
        self.stereo = self.model_parameters_table["stereo"][
            0
        ]  # True if self.min_telescopes >= 2 else False
        training_table_gamma = read_table_hdf5(
            f"{self.project_directories.model_index_file}",
            path=IndexTables(self, ParticleType.GAMMA_DIFFUSE).TRAINING.table_path,
        )

        if self.model_parameters_table["reco"][0] == "type":
            training_table_proton = read_table_hdf5(
                f"{self.project_directories.model_index_file}",
                path=IndexTables(self, ParticleType.PROTON).TRAINING.table_path,
            )
            if (len(training_table_proton["training_proton_patterns"]) == 0) or (
                len(training_table_gamma["training_gamma_diffuse_patterns"]) == 0
            ):
                raise ValueError(
                    "For reco type, training_proton_patterns and training_gamma_diffuse_patterns are required"
                )

        if self.stereo and len(self.telescope_ids) < 2:
            raise ValueError("For stereo mode, at least 2 telescopes are required")

        if self.model_parameters_table["reco"][0] == "cameradirection" and self.stereo:
            raise ValueError(
                "For reco cameradirection, stereo mode is not supported, use skydirection instead."
            )
        # Check that all gamma related lists are the same length
        # gamma_lengths = [len(training_table_gamma['training_gamma_diffuse_patterns']), len(training_table_gamma['training_gamma_diffuse_zenith_distances']), len(training_table_gamma['training_gamma_diffuse_azimuths'])]
        # if len(set(gamma_lengths)) != 1:
        #     raise ValueError("All gamma related lists must be the same length")

        # Check that all proton related lists are the same length
        # proton_lengths = [len(training_table_proton['training_proton_patterns']), len(training_table_proton['training_proton_zenith_distances']), len(training_table_proton['training_proton_azimuths'])]
        # if len(set(proton_lengths)) != 1:
        #     raise ValueError("All proton related lists must be the same length")

        self.cluster_configuration = cluster_configuration
        set_mpl_style()

        # current_model_dir = self.model_parameters_table['model_dir'][0]
        # if f"/{self.model_nickname}" not in current_model_dir:
        #     print("⚠️ Updating model directories for compatibility with the new version of CTLearnManager")
        #     import os
        #     import glob
        #     os.system(f"mkdir {current_model_dir}/{self.model_nickname}")
        #     model_dirs = glob.glob(f"{current_model_dir}/{self.model_nickname}_v*")
        #     for model_dir in model_dirs:
        #         print(f"➡️ Moving {model_dir} to {model_dir}/{self.model_nickname}")
        #         os.system(f"mv {model_dir} {model_dir}/{self.model_nickname}/")

    def save_to_index(self, model_parameters):
        """
        Save model parameters and training samples to an HDF5 index file.

        Parameters
        ----------
        model_parameters : dict
            A dictionary containing model parameters and training sample details.
            Expected keys include:
            - "model_dir" (str): Absolute path to the model directory.
            - "reco" (str, optional): Reconstruction type, one of
              ['type', 'energy', 'cameradirection', 'skydirection']. Defaults to "default_reco".
            - "channels" (list of str, optional): List of channel names. Defaults to
              ["cleaned_image", "cleaned_relative_peak_time"].
            - "telescope_names" (list of str, optional): Names of telescopes. Defaults to an empty list.
            - "telescope_ids" (list of int, optional): IDs of telescopes. Defaults to an empty list.
            - "notes" (str, optional): Notes about the model. Defaults to an empty string.
            - "max_training_epochs" (int, optional): Maximum number of training epochs. Defaults to 10.
            - "min_telescopes" (int, optional): Minimum number of telescopes. Defaults to 1.
            - "stereo" (bool, optional): Whether the model uses stereo mode. Defaults to True if
              `min_telescopes` >= 2, otherwise False.
            - "training_samples" (list, optional): List of training sample objects. Each object must
              have the following attributes:
              - `particle_type` (enum): Type of particle.
              - `directory` (str): Directory of the training sample.
              - `pattern` (str): Pattern of the training sample.
              - `zenith_distance` (float): Zenith distance in degrees.
              - `azimuth` (float): Azimuth in degrees.
              - `energy_range` (tuple of float): Minimum and maximum energy in TeV.
              - `nsb_range` (tuple of float): Minimum and maximum NSB in Hz.

        Raises
        ------
        ValueError
            If the "model_dir" is not an absolute path.
        AssertionError
            If any of the following conditions are not met:
            - Telescope names and IDs have the same length.
            - `telescope_ids` is a 1-dimensional array.
            - `telescope_names` is a 1-dimensional array.
            - `channels` is a 1-dimensional array.
            - `reco` is one of ['type', 'energy', 'cameradirection', 'skydirection'].
            - `max_training_epochs` is an integer.
            - `min_telescopes` is an integer.

        Notes
        -----
        This method creates or updates an HDF5 file to store model parameters and training sample
        details. If the model nickname already exists in the index, it will not overwrite the existing
        entry. Training sample details are stored under paths specific to the particle type.
        """
        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5

        paramaters_index_table = IndexTables(self).PARAMETERS

        try:
            model_table = QTable.read(
                self.project_directories.model_index_file,
                format="hdf5",
                path=paramaters_index_table.table_path,
            )
            print(f"❌ Model nickname {self.model_nickname} already in table")
        except:
            model_table = paramaters_index_table.default_table

            notes = model_parameters.get("notes", "")
            if not Path(model_parameters.get("model_dir", "")).is_absolute():
                raise ValueError("The 'model_dir' must be an absolute path.")
            model_dir = f"{model_parameters.get('model_dir', '')}/{self.model_nickname}"
            reco = model_parameters.get("reco", "default_reco")
            telescope_names = model_parameters.get("telescope_names", [])
            telescope_ids = model_parameters.get("telescope_ids", [])
            channels = model_parameters.get(
                "channels", ["cleaned_image", "cleaned_relative_peak_time"]
            )
            max_training_epochs = model_parameters.get("max_training_epochs", 10)
            min_telescopes = model_parameters.get("min_telescopes", 1)
            stereo = model_parameters.get(
                "stereo", True if min_telescopes >= 2 else False
            )

            assert len(telescope_names) == len(telescope_ids), (
                "Telescope names and IDs must have the same length"
            )
            assert np.ndim(telescope_ids) == 1, (
                "telescope_ids must be a 1-dimensional array"
            )
            assert np.ndim(telescope_names) == 1, (
                "telescope_names must be a 1-dimensional array"
            )
            assert np.ndim(channels) == 1, "channels must be a 1-dimensional array"
            assert reco in ["type", "energy", "cameradirection", "skydirection"], (
                "reco must be one of ['type', 'energy', 'cameradirection', 'skydirection']"
            )
            assert type(max_training_epochs) is int, (
                "max_training_epochs must be an integer"
            )
            assert type(min_telescopes) is int, "min_telescopes must be an integer"

            model_table.add_row(
                [
                    self.model_nickname,
                    model_dir,
                    reco,
                    str(channels),
                    str(telescope_names),
                    str(telescope_ids),
                    notes,
                    max_training_epochs,
                    min_telescopes,
                    stereo,
                ]
            )
            write_table_hdf5(
                model_table,
                self.project_directories.model_index_file,
                path=paramaters_index_table.table_path,
                append=True,
                overwrite=True,
            )

            training_samples = model_parameters.get("training_samples", [])

            for training_sample in training_samples:
                particle_type = training_sample.particle_type

                training_index_table = IndexTables(self, particle_type).TRAINING

                try:
                    training_table = read_table_hdf5(
                        self.project_directories.model_index_file,
                        path=training_index_table.table_path,
                    )
                except:
                    training_table = training_index_table.default_table

                training_table.add_row(
                    [
                        training_sample.directory,
                        training_sample.pattern,
                        training_sample.zenith_distance,
                        training_sample.azimuth,
                        min(training_sample.energy_range),
                        max(training_sample.energy_range),
                        min(training_sample.nsb_range),
                        max(training_sample.nsb_range),
                    ]
                )
                write_table_hdf5(
                    training_table,
                    self.project_directories.model_index_file,
                    path=training_index_table.table_path,
                    append=True,
                    overwrite=True,
                    serialize_meta=True,
                )
            print(f"✅ Model nickname {self.model_nickname} added to table")

    def launch_training(
        self,
        n_epochs,
        save_best_validation_only=None,
        transfer_learning_model_cpk=None,
        trainable_backbone=True,
        force_dl1_lookup=False,
        config_file=None,
        batch_size=64,
    ):
        """
        Launch the training process for the model.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to train the model. If set to 0, training will not proceed.
        save_best_validation_only : bool, optional
            Whether to save only the best validation model during training. Overrides the default behavior.
        transfer_learning_model_cpk : str, optional
            Path to a checkpoint file for transfer learning. If provided, the model will be initialized from this checkpoint.
        trainable_backbone : bool, default=True
            Whether the backbone of the model should be trainable.
        force_dl1_lookup : bool, default=False
            Whether to force a lookup for DL1 data.
        config_file : str, optional
            Path to a configuration file. If not provided, a new configuration file will be generated.
        batch_size : int, default=64
            Batch size to use during training.

        Returns
        -------
        None
            This method does not return any value. It either launches the training process or exits early if conditions are not met.

        Notes
        -----
        - If the model has already been trained for the maximum number of epochs, training will not proceed.
        - Automatically handles model versioning and directory creation for saving models.
        - Generates a configuration file if none is provided.
        - Supports both local and cluster-based training execution.
        """
        import glob
        import json
        import os

        import numpy as np
        from astropy.io.misc.hdf5 import read_table_hdf5

        self.cluster_configuration.info()
        if n_epochs == 0:
            print("🛑 Number of epochs set to 0. Will not train the model.")
            return
        n_epoch_trained = self.get_n_epoch_trained()
        max_training_epochs = self.model_parameters_table["max_training_epochs"][0]
        base_model_dir = self.model_parameters_table["model_dir"][0]
        if n_epochs > max_training_epochs - n_epoch_trained:
            print(
                f"⚠️ Number of epochs increased from {max_training_epochs} to {n_epochs}"
            )
            self.update_model_manager_parameters_in_index(
                {"max_training_epochs": n_epochs}
            )
            max_training_epochs = n_epochs
            n_epochs = max_training_epochs - n_epoch_trained
        if n_epoch_trained >= max_training_epochs:
            print(
                f"🛑 Model already trained for {n_epoch_trained} epochs. Will not train further."
            )
            self.plot_loss()
            return

        trained_string = "―" * (n_epoch_trained - 1)
        trained_spaces = " " * (n_epoch_trained - 1)
        remaining_string = "·" * (max_training_epochs - n_epoch_trained)
        to_train_string = "―" * (n_epochs - len(str(n_epoch_trained)))
        print(
            f"{trained_string}o{remaining_string} | {n_epoch_trained}/{max_training_epochs} epochs"
        )
        print(
            f"{trained_spaces}{n_epoch_trained}{to_train_string}> 🚀 Training for {n_epochs} epochs"
        )

        models_dir = np.sort(glob.glob(f"{base_model_dir}/{self.model_nickname}_v*"))
        load_model = False
        if len(models_dir) > 0:
            last_model_dir = Path(models_dir[-1])
            size = sum(
                f.stat().st_size for f in last_model_dir.glob("**/*") if f.is_file()
            )
            model_version = int(models_dir[-1].split("_v")[-1])
            if size > 1e6:
                model_version += 1
                model_dir = f"{base_model_dir}/{self.model_nickname}_v{model_version}/"
                print(
                    f"➡️ Model already exists: will continue training and create {model_dir}"
                )
                _save_best_validation_only = True
                model_to_load = f"{base_model_dir}/{self.model_nickname}_v{model_version - 1}/ctlearn_model.cpk"
                load_model = True
                os.system(f"mkdir -p {model_dir}")
            else:
                model_dir = f"{base_model_dir}/{self.model_nickname}_v{model_version}/"
                if model_version > 0:
                    model_to_load = f"{base_model_dir}/{self.model_nickname}_v{model_version - 1}/ctlearn_model.cpk"
                    load_model = True
                    print(
                        f"➡️ Model already exists: will continue training and create {model_dir}"
                    )
                    _save_best_validation_only = True
                else:
                    print(f"🆕 Model does not exist: will create {model_dir}")
                    _save_best_validation_only = False
        else:
            model_version = 0
            model_dir = f"{base_model_dir}/{self.model_nickname}_v{model_version}/"
            print(f"🆕 Model does not exist: will create {model_dir}")
            os.system(f"mkdir -p {model_dir}")
            _save_best_validation_only = False

        if save_best_validation_only is not None:
            _save_best_validation_only = save_best_validation_only

        if load_model:
            load_model_string = f"--TrainCTLearnModel.model_type=LoadedModel --LoadedModel.load_model_from={model_to_load} "
        else:
            load_model_string = (
                ""
                if transfer_learning_model_cpk is None
                else f"--TrainCTLearnModel.model_type=LoadedModel --LoadedModel.load_model_from={transfer_learning_model_cpk} "
            )
        training_gamma_table = read_table_hdf5(
            self.project_directories.model_index_file, path=IndexTables(self, ParticleType.GAMMA_DIFFUSE).TRAINING.table_path
        )
        signal_patterns = ""
        for pattern in training_gamma_table["training_gamma_diffuse_patterns"]:
            signal_patterns += f'--pattern-signal "{pattern}" '
        background_patterns = ""
        if self.model_parameters_table["reco"][0] == "type":
            training_proton_table = read_table_hdf5(
                self.project_directories.model_index_file, path=IndexTables(self, ParticleType.PROTON).TRAINING.table_path
            )
            for pattern in training_proton_table["training_proton_patterns"]:
                background_patterns += f'--pattern-background "{pattern}" '
        background_string = (
            f"--background {training_proton_table['training_proton_dir'][0]} "
            if self.model_parameters_table["reco"][0] == "type"
            else ""
        )
        channels = ast.literal_eval(self.model_parameters_table["channels"][0])
        stereo_mode = "stereo" if self.stereo else "mono"
        stack_telescope_images = True if self.stereo else False
        allowed_tels = ast.literal_eval(self.model_parameters_table["telescope_ids"][0])

        if config_file is None:
            config = {}
            config["TrainCTLearnModel"] = {}
            config["TrainCTLearnModel"]["save_best_validation_only"] = (
                _save_best_validation_only
            )
            config["TrainCTLearnModel"]["n_epochs"] = int(n_epochs)
            config["TrainCTLearnModel"]["stack_telescope_images"] = (
                stack_telescope_images
            )
            config["TrainCTLearnModel"]["reco_tasks"] = [
                self.model_parameters_table["reco"][0]
            ]
            config["TrainCTLearnModel"]["output_dir"] = model_dir

            config["TrainCTLearnModel"]["DLImageReader"] = {}
            config["TrainCTLearnModel"]["DLImageReader"]["allowed_tels"] = allowed_tels
            config["TrainCTLearnModel"]["DLImageReader"]["min_telescopes"] = int(
                self.min_telescopes
            )
            config["TrainCTLearnModel"]["DLImageReader"]["force_dl1_lookup"] = (
                force_dl1_lookup
            )
            config["TrainCTLearnModel"]["DLImageReader"]["mode"] = stereo_mode
            config["TrainCTLearnModel"]["DLImageReader"]["channels"] = channels

            config["LoadedModel"] = {}
            config["LoadedModel"]["trainable_backbone"] = trainable_backbone

            config_file = f"{base_model_dir}/train_config{self.model_nickname}_v{model_version}.json"
            with open(config_file, "w") as file:
                json.dump(config, file)
            print(f"Configuration saved to {config_file}")

        cmd = f"ctlearn-train-model {load_model_string} \
--TrainCTLearnModel.batch_size={batch_size} \
--signal {training_gamma_table['training_gamma_diffuse_dir'][0]} {signal_patterns}\
{background_string} {background_patterns}\
--output {model_dir} \
--config {config_file} \
--overwrite \
--verbose"

        if self.cluster_configuration.use_cluster:
            sbatch_file = self.cluster_configuration.write_sbatch_script(
                f"train_{self.model_nickname}_v{model_version}", cmd, base_model_dir
            )
            os.system(f"sbatch {sbatch_file}")
        else:
            print(cmd)
            os.system(cmd)

    def get_n_epoch_trained(self):
        """
        Calculate the total number of epochs trained across all training logs.

        This method searches for training log files in the model directory
        corresponding to the current model nickname, reads them, and sums up
        the number of epochs recorded in each log.

        Returns
        -------
        int
            The total number of epochs trained.
        """
        import glob

        import pandas as pd

        training_logs = np.sort(
            glob.glob(
                f"{self.model_parameters_table['model_dir'][0]}/{self.model_nickname}*/training_log.csv"
            )
        )
        n_epochs = 0
        for training_log in training_logs:
            df = pd.read_csv(training_log)
            n_epochs += len(df)
        return n_epochs

    def plot_loss(self):
        """
        Plot the training and validation loss over epochs.

        This method reads training logs from CSV files, extracts the loss values
        for training and validation, and plots them against the epochs. If no
        training logs are found, it prints an error message and exits.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - The method assumes that training logs are stored in CSV files within
          directories matching the pattern `{model_dir}/{model_nickname}*/training_log.csv`.
        - The CSV files must contain columns named "loss" and "val_loss".
        - If only one epoch is available, the losses are displayed as scatter points.
        """
        import glob

        import matplotlib.pyplot as plt
        import pandas as pd

        training_logs = np.sort(
            glob.glob(
                f"{self.model_parameters_table['model_dir'][0]}/{self.model_nickname}*/training_log.csv"
            )
        )
        losses_train = []
        losses_val = []
        for training_log in training_logs:
            df = pd.read_csv(training_log)
            losses_train = np.concatenate((losses_train, df["loss"].to_numpy()))
            losses_val = np.concatenate((losses_val, df["val_loss"].to_numpy()))
        epochs = np.arange(1, len(losses_train) + 1)
        if len(epochs) == 0:
            print(
                f"❌ No training logs found for {self.model_nickname}, stert the training to see the loss."
            )
            return
        if len(epochs) > 1:
            plt.plot(epochs, losses_train, label="Training", lw=2)
            plt.plot(epochs, losses_val, label="Validation", ls="--")
        else:
            plt.scatter(epochs, losses_train, label="Training", lw=2)
            plt.scatter(epochs, losses_val, label="Validation", ls="--")
        plt.title(f"{self.model_parameters_table['reco'][0]} training".title())
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(np.arange(1, len(losses_train) + 1, 2))
        plt.legend()
        plt.show()

    def update_model_manager_parameters_in_index(self, parameters: dict):
        """
        Update the model manager parameters in the HDF5 index file.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the parameter names and their new values to update.

        Notes
        -----
        - This method reads the model parameters table from the HDF5 file, updates the specified
          parameters, and writes the updated table back to the file.
        - If a parameter's data type is Unicode string, it is converted to a fixed-length string
          format ('S256') to handle long strings.
        - The method also updates the corresponding attributes in the instance's `__dict__`.

        Raises
        ------
        KeyError
            If a specified parameter key does not exist in the model table.
        """
        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5

        

        model_table = read_table_hdf5(
            self.project_directories.model_index_file, path=IndexTables(self).PARAMETERS.table_path
        )
        # model_index = np.where(model_table['model_nickname'] == self.model_nickname)[0][0]
        print(f"💾 Model {self.model_nickname} index update:")
        for key, value in parameters.items():
            if (
                model_table[key].dtype.kind == "U"
            ):  # Check if the dtype is Unicode string
                model_table[key] = model_table[key].astype(
                    "S256"
                )  # Convert to 'S256' for long strings
            model_table[key][0] = value
            self.__dict__[key] = value
            print(f"\t➡️ {key} updated to {value}")
        write_table_hdf5(
            model_table,
            self.project_directories.model_index_file,
            path=IndexTables(self).PARAMETERS.table_path,
            append=True,
            overwrite=True,
            serialize_meta=True,
        )

    def update_model_manager_testing_data(self, testing_data_sample: DataSample):
        """
        Update the testing data for the model manager with a new data sample.

        Parameters
        ----------
        testing_data_sample : DataSample
            The data sample containing testing information to be added or updated.
            It includes the directory, zenith distance, azimuth, pattern, and particle type.

        Raises
        ------
        Exception
            If there is an issue reading or writing the HDF5 file.

        Notes
        -----
        - If the testing data for the given zenith distance and azimuth already exists,
          it updates the directory and pattern for that entry.
        - If no matching entry exists, it adds a new row to the testing data table.
        - The updated table is saved back to the HDF5 file.
        """
        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5

        testing_dir = testing_data_sample.directory
        testing_zenith_distance = testing_data_sample.zenith_distance
        testing_azimuth = testing_data_sample.azimuth
        testing_pattern = testing_data_sample.pattern
        particle_type = testing_data_sample.particle_type

        testing_index_table = IndexTables(self, particle_type).TESTING
        try:
            testing_table = read_table_hdf5(
                self.project_directories.model_index_file,
                path=testing_index_table.table_path,
            )
        except:
            testing_table = testing_index_table.default_table
        # print(f"💾 Model {self.model_nickname} testing data update:")
        if len(testing_table) == 0:
            testing_table = testing_index_table.default_table

        match = np.where(
            (
                testing_table[f"testing_{particle_type.value}_zenith_distances"]
                == testing_zenith_distance
            )
            & (
                testing_table[f"testing_{particle_type.value}_azimuths"]
                == testing_azimuth
            )
        )[0]
        if len(match) > 0:
            testing_table[f"testing_{particle_type.value}_dirs"][match[0]] = testing_dir
            testing_table[f"testing_{particle_type.value}_patterns"][match[0]] = (
                testing_pattern
            )
        else:
            testing_table.add_row(
                [testing_dir, testing_zenith_distance, testing_azimuth, testing_pattern]
            )
        write_table_hdf5(
            testing_table,
            self.project_directories.model_index_file,
            path=testing_index_table.table_path,
            append=True,
            overwrite=True,
            serialize_meta=True,
        )
        # print(
        #     f"Testing {particle_type.value} at ({testing_zenith_distance}, {testing_azimuth}) : {testing_dir}/{testing_pattern} updated"
        # )


    def plot_zenith_azimuth_ranges(self, ax=None, plot_testing_nodes=True):
        """
        Plot the zenith and azimuth ranges on a polar plot.

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes, optional
            The matplotlib axis to plot on. If None, a new polar plot is created.
        plot_testing_nodes : bool, optional
            Whether to plot testing nodes (default is True).

        Notes
        -----
        - The function visualizes the zenith and azimuth ranges for training and testing data.
        - Training data is represented with filled markers, while testing data is represented with outlined markers.
        - The plot includes zenith and azimuth ranges as circles or arcs, depending on the data.
        - The zenith range is displayed in degrees, and the azimuth range is displayed in radians.
        - The function handles cases where the azimuth range is not defined or contains NaN values.

        Raises
        ------
        Exception
            If there is an issue reading the HDF5 tables for training or testing data.

        See Also
        --------
        astropy.io.misc.hdf5.read_table_hdf5 : Used to read HDF5 tables.
        matplotlib.pyplot : Used for plotting.
        """
        import astropy.units as u
        import matplotlib.pyplot as plt
        from astropy.io.misc.hdf5 import read_table_hdf5

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

        zenith_range = self.validity.zenith_range
        azimuth_range = self.validity.azimuth_range

        zenith_min, zenith_max = zenith_range.to(u.deg)

        if azimuth_range is None:
            azimuth_min, azimuth_max = 0, 2 * np.pi
        else:
            azimuth_min, azimuth_max = azimuth_range.to(u.rad)

        if zenith_min == zenith_max:
            if np.isnan(azimuth_min) and np.isnan(azimuth_max):
                # Plot a circle for this zenith
                theta = np.linspace(0, 2 * np.pi, 100) * u.rad
                r = np.full_like(theta, zenith_min).to(u.deg)
                ax.plot(theta, r, lw=3, zorder=0)
            elif azimuth_min == azimuth_max:
                # Plot a point for that position
                ax.scatter(
                    azimuth_min,
                    zenith_min,
                    s=100,
                    zorder=0,
                    label="Training",
                    color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
                )
            else:
                # Plot a portion of a circle between the azimuth range at the correct zenith
                
                theta = np.linspace(azimuth_min, azimuth_max, 100)
                r = np.full_like(theta, zenith_min).to(u.deg)
                ax.plot(theta, r, lw=3, zorder=0)
                training_gamma_table = read_table_hdf5(
                    self.project_directories.model_index_file,
                    path=IndexTables(self, ParticleType.GAMMA_DIFFUSE).TRAINING.table_path,
                )
                zeniths = training_gamma_table[
                    "training_gamma_diffuse_zenith_distances"
                ]
                azimuths = training_gamma_table["training_gamma_diffuse_azimuths"].to(
                    u.rad
                )
                for zenith, azimuth in zip(zeniths, azimuths):
                    ax.scatter(
                        azimuth,
                        zenith,
                        s=50,
                        color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
                        label="Training",
                    )
        else:
            if np.isnan(azimuth_min) and np.isnan(azimuth_max):
                # Plot the area between the two circles
                theta = np.linspace(0, 2 * np.pi, 100) * u.rad
                r1 = np.full_like(theta, zenith_min).to(u.deg)
                r2 = np.full_like(theta, zenith_max).to(u.deg)
                ax.fill_between(
                    theta.value,
                    r1.value,
                    r2.value,
                    alpha=0.3,
                    zorder=0,
                    color=CTLearnManagerStyle.ctlearn_highlight.value,
                )
                ax.plot(
                    theta,
                    r1,
                    lw=3,
                    color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
                    zorder=0,
                )
                ax.plot(
                    theta,
                    r2,
                    lw=3,
                    color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
                    zorder=0,
                )
            else:
                theta = np.linspace(azimuth_min, azimuth_max, 100)
                r1 = np.full_like(theta, zenith_min).to(u.deg).value
                r2 = np.full_like(theta, zenith_max).to(u.deg).value
                theta = theta.value
                ax.fill_between(
                    theta,
                    r1,
                    r2,
                    alpha=0.3,
                    zorder=0,
                    color=CTLearnManagerStyle.ctlearn_highlight.value,
                )
                ax.plot(
                    theta,
                    r1,
                    lw=3,
                    color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
                    zorder=0,
                )
                ax.plot(
                    theta,
                    r2,
                    lw=3,
                    color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
                    zorder=0,
                )
                ax.plot(
                    (theta[0], theta[0]),
                    (r1[0], r2[0]),
                    lw=3,
                    color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
                    zorder=0,
                )
                ax.plot(
                    (theta[-1], theta[-1]),
                    (r1[-1], r2[-1]),
                    lw=3,
                    color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
                    zorder=0,
                )
                ax.set_ylim(0, 60)
                training_gamma_table = read_table_hdf5(
                    self.project_directories.model_index_file,
                    path=IndexTables(self, ParticleType.GAMMA_DIFFUSE).TRAINING.table_path,
                )
                zeniths = training_gamma_table[
                    "training_gamma_diffuse_zenith_distances"
                ]
                azimuths = training_gamma_table["training_gamma_diffuse_azimuths"].to(
                    u.rad
                )
                for zenith, azimuth in zip(zeniths, azimuths):
                    ax.scatter(
                        azimuth,
                        zenith,
                        s=50,
                        color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
                        label="Training",
                    )

        try:
            testing_dl1_table = read_table_hdf5(
                self.project_directories.model_index_file, path=IndexTables(self, ParticleType.GAMMA_POINT).TESTING.table_path
            )
            zeniths = testing_dl1_table["testing_gamma_point_zenith_distances"]
            azimuths = testing_dl1_table["testing_gamma_point_azimuths"].to(u.rad)
            for zenith, azimuth in zip(zeniths, azimuths):
                if (zenith == np.nan) or (azimuth == np.nan) or not plot_testing_nodes:
                    continue
                else:
                    ax.scatter(
                        azimuth,
                        zenith,
                        s=50,
                        facecolors="none",
                        edgecolors=CTLearnManagerStyle.ctlearn_accent_1.value,
                        label="Testing DL1",
                        zorder=3,
                    )
        except:
            a = 1

        try:
            mc_dl2_table = read_table_hdf5(
                self.project_directories.model_index_file, path=IndexTables(self, ParticleType.GAMMA_POINT).DL2_MC.table_path
            )
            zeniths = mc_dl2_table["testing_DL2_gamma_point_zenith_distances"]
            azimuths = mc_dl2_table["testing_DL2_gamma_point_azimuths"].to(u.rad)
            for zenith, azimuth in zip(zeniths, azimuths):
                if (zenith == np.nan) or (azimuth == np.nan) or not plot_testing_nodes:
                    continue
                else:
                    ax.scatter(
                        azimuth,
                        zenith,
                        s=50,
                        color=CTLearnManagerStyle.ctlearn_accent_2.value,
                        label="Testing DL2",
                        zorder=2,
                    )
        except:
            a = 1
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(-30)
        ax.set_ylim(0, 60)
        ax.set_yticks(np.arange(10, 61, 10))
        ax.set_yticklabels(["", "", "30°", "", "", "60°"], fontsize=10)
        ax.set_xlabel("Azimuth [deg]", fontsize=10)

        ax.set_title("Zenith and Azimuth Ranges", pad=30)
        plt.tight_layout()
        if ax is None:
            plt.show()

    def plot_training_nodes(self):
        """
        Plot the training nodes for gamma and proton events in a polar coordinate system.

        This method visualizes the training nodes for gamma and proton events using
        their zenith and azimuth angles. The plot is displayed in a polar coordinate
        system, with specific styling for gamma and proton events.

        Parameters
        ----------
        None

        Notes
        -----
        - Gamma training nodes are read from the HDF5 file at the path
          `<model_nickname>/training/gamma_diffuse`.
        - Proton training nodes are read from the HDF5 file at the path
          `<model_nickname>/training/proton` if the model parameter `reco` is set to "type".
        - If zenith or azimuth values are undefined (NaN), those nodes are skipped.
        - The plot includes custom styling for gamma and proton nodes, with distinct
          colors and markers.
        Warnings
        --------
        - If no valid zenith or azimuth values are found for gamma or proton nodes,
          a message is printed to indicate that the corresponding training nodes
          cannot be shown.

        See Also
        --------
        astropy.io.misc.hdf5.read_table_hdf5 : Used to read the training data tables.
        matplotlib.pyplot.subplots : Used to create the polar plot.
        """
        import astropy.units as u
        import matplotlib.pyplot as plt
        from astropy.io.misc.hdf5 import read_table_hdf5

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        training_gamma_table = read_table_hdf5(
            self.project_directories.model_index_file, path=IndexTables(self, ParticleType.GAMMA_DIFFUSE).TRAINING.table_path
        )
        zeniths = training_gamma_table["training_gamma_diffuse_zenith_distances"]
        azimuths = training_gamma_table["training_gamma_diffuse_azimuths"].to(u.rad)
        i = 0
        for zenith, azimuth in zip(zeniths, azimuths):
            if (zenith == np.nan) or (azimuth == np.nan):
                continue
            else:
                ax.scatter(
                    azimuth,
                    zenith,
                    s=50,
                    color=CTLearnManagerStyle.ctlearn_1.value,
                    label="Gammas",
                    zorder=10,
                )
                i += 1
        if i == 0:
            print(
                "Training nodes for gammas cannot be shown because the zenith or azimuth are not defined."
            )

        if self.model_parameters_table["reco"][0] == "type":
            training_proton_table = read_table_hdf5(
                self.project_directories.model_index_file, path=IndexTables(self, ParticleType.PROTON).TRAINING.table_path
            )
            zeniths = training_proton_table["training_proton_zenith_distances"]
            azimuths = training_proton_table["training_proton_azimuths"].to(u.rad)
            i = 0
            for zenith, azimuth in zip(zeniths, azimuths):
                if (zenith == np.nan) or (azimuth == np.nan):
                    continue
                else:
                    ax.scatter(
                        azimuth,
                        zenith,
                        label="Protons",
                        edgecolor=CTLearnManagerStyle.ctlearn_accent_1.value,
                        facecolors="w",
                        zorder=1,
                        s=100,
                        lw=2,
                    )
                    i += 1
            if i == 0:
                print(
                    "Training nodes for protons cannot be shown because the zenith or azimuth are not defined."
                )

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(-30)
        ax.set_ylim(0, 60)
        ax.set_yticks(np.arange(10, 61, 10))
        ax.set_yticklabels(["", "", "30°", "", "", "60°"], fontsize=10)
        ax.set_xlabel("Azimuth [deg]", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.set_title("Training nodes", pad=30)
        plt.tight_layout()
        plt.show()

    def remove_row_from_table(self, table_path: str, row_index: int):
        remove_row_from_table_utils(self.project_directories.model_index_file, table_path, row_index)


class ModelRangeOfValidity:
    """
    Class to represent the range of validity for a CTLearn model.

    This class extracts and stores the ranges of zenith, azimuth, energy, and NSB
    values from the training gamma data of a CTLearn model. It also provides a
    method to check if given parameters fall within these ranges.

    Parameters
    ----------
    model_manager : CTLearnModelManager
        An instance of CTLearnModelManager containing the model index file and
        model nickname.

    Attributes
    ----------
    zenith_range : astropy.units.Quantity
        The range of zenith distances in the training gamma data.
    azimuth_range : astropy.units.Quantity
        The range of azimuths in the training gamma data.
    energy_range : astropy.units.Quantity
        The range of energies in the training gamma data.
    nsb_range : astropy.units.Quantity
        The range of NSB values in the training gamma data.

    Methods
    -------
    matches(**kwargs)
        Check if the given parameters fall within the model's range of validity.
    """

    def __init__(self, model_manager: CTLearnModelManager):
        """
        Initialize the instance with model parameters from the provided CTLearnModelManager.

        Parameters
        ----------
        model_manager : CTLearnModelManager
            An instance of CTLearnModelManager containing the model index file
            and model nickname to retrieve training data.

        Attributes
        ----------
        zenith_range : astropy.units.Quantity
            The range of zenith distances in the training gamma diffuse data.
        azimuth_range : astropy.units.Quantity
            The range of azimuth angles in the training gamma diffuse data.
        energy_range : astropy.units.Quantity
            The range of energy values in the training gamma diffuse data.
        nsb_range : astropy.units.Quantity
            The range of night sky background (NSB) values in the training gamma diffuse data.
        """
        from astropy.io.misc.hdf5 import read_table_hdf5

        training_gamma_table = read_table_hdf5(
            model_manager.project_directories.model_index_file,
            path=IndexTables(model_manager, ParticleType.GAMMA_DIFFUSE).TRAINING.table_path,
        )
        # training_proton_table = read_table_hdf5(model_manager.model_index_file, path=f'{model_manager.model_nickname}/training/proton')

        training_gamma_zeniths = training_gamma_table[
            "training_gamma_diffuse_zenith_distances"
        ]
        self.zenith_range = [
            min(training_gamma_zeniths).value,
            max(training_gamma_zeniths).value,
        ] * training_gamma_zeniths.unit
        training_gamma_azimuths = training_gamma_table[
            "training_gamma_diffuse_azimuths"
        ]
        self.azimuth_range = [
            min(training_gamma_azimuths.value),
            max(training_gamma_azimuths).value,
        ] * training_gamma_azimuths.unit
        training_gamma_energies_mins = training_gamma_table[
            "training_gamma_diffuse_energy_min"
        ]
        training_gamma_energies_maxs = training_gamma_table[
            "training_gamma_diffuse_energy_max"
        ]
        taining_gamma_energies = np.concatenate(
            (training_gamma_energies_mins, training_gamma_energies_maxs)
        )
        self.energy_range = [
            min(taining_gamma_energies).value,
            max(taining_gamma_energies).value,
        ] * taining_gamma_energies.unit
        training_gamma_nsbs_mins = training_gamma_table[
            "training_gamma_diffuse_nsb_min"
        ]
        training_gamma_nsbs_maxs = training_gamma_table[
            "training_gamma_diffuse_nsb_max"
        ]
        taining_gamma_nsbs = np.concatenate(
            (training_gamma_nsbs_mins, training_gamma_nsbs_maxs)
        )
        self.nsb_range = [
            min(taining_gamma_nsbs).value,
            max(taining_gamma_nsbs).value,
        ] * taining_gamma_nsbs.unit

    def matches(self, **kwargs):
        """
        Check if the given parameters match the defined ranges.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments representing the parameters to check. Supported keys are:
            - "zenith": float, the zenith angle to check.
            - "azimuth": float, the azimuth angle to check.
            - "energy": float, the energy value to check.
            - "nsb": float, the night sky background (NSB) value to check.

        Returns
        -------
        bool
            True if all provided parameters fall within their respective ranges, False otherwise.

        Notes
        -----
        - If a range (e.g., `azimuth_range`, `energy_range`, or `nsb_range`) is `None`, the corresponding parameter is not checked.
        - The `zenith_range` is always checked if the "zenith" key is provided.
        """
        for key, value in kwargs.items():
            if key == "zenith":
                if not (self.zenith_range[0] <= value <= self.zenith_range[1]):
                    return False
            elif key == "azimuth":
                if self.azimuth_range is not None and not (
                    self.azimuth_range[0] <= value <= self.azimuth_range[1]
                ):
                    return False
            elif key == "energy":
                if self.energy_range is not None and not (
                    self.energy_range[0] <= value <= self.energy_range[1]
                ):
                    return False
            elif key == "nsb":
                if self.nsb_range is not None and not (
                    self.nsb_range[0] <= value <= self.nsb_range[1]
                ):
                    return False
        return True


class IndexTables:

    def __init__(self, model_manager: CTLearnModelManager, particle_type: ParticleType=None):
        self.model_manager = model_manager
        self.particle_type = particle_type

        if self.particle_type is not None:

            self.DL2_MC = self.IndexTable(
                QTable(
                        names=[
                            f"testing_DL2_{self.particle_type.value}_files",
                            f"testing_DL2_{self.particle_type.value}_zenith_distances",
                            f"testing_DL2_{self.particle_type.value}_azimuths",
                            "merged",
                        ],
                        dtype=["S256", float, float, bool],
                        units=[None, "deg", "deg", None],
                    ),
                f"{self.model_manager.model_nickname}/DL2/MC/{particle_type.value}"
                )


            self.TRAINING = self.IndexTable(
                QTable(
                        names=[
                            f"training_{particle_type.value}_dir",
                            f"training_{particle_type.value}_patterns",
                            f"training_{particle_type.value}_zenith_distances",
                            f"training_{particle_type.value}_azimuths",
                            f"training_{particle_type.value}_energy_min",
                            f"training_{particle_type.value}_energy_max",
                            f"training_{particle_type.value}_nsb_min",
                            f"training_{particle_type.value}_nsb_max",
                        ],
                        dtype=[
                            "S256",
                            "S256",
                            float,
                            float,
                            float,
                            float,
                            float,
                            float,
                        ],
                        units=[None, None, "deg", "deg", "TeV", "TeV", "Hz", "Hz"],
                    ),
                f"{self.model_manager.model_nickname}/training/{particle_type.value}")
            
            self.TESTING = self.IndexTable(
                QTable(
                    names=[
                        f"testing_{particle_type.value}_dirs",
                        f"testing_{particle_type.value}_zenith_distances",
                        f"testing_{particle_type.value}_azimuths",
                        f"testing_{particle_type.value}_patterns",
                    ],
                    dtype=["S256", float, float, "S256"],
                    units=[None, "deg", "deg", None],
            ),
                f"{self.model_manager.model_nickname}/testing/{particle_type.value}",
            )


        self.PARAMETERS = self.IndexTable(
            QTable(
                names=[
                    "model_nickname",
                    "model_dir",
                    "reco",
                    "channels",
                    "telescope_names",
                    "telescope_ids",
                    "notes",
                    "max_training_epochs",
                    "min_telescopes",
                    "stereo",
                ],
                dtype=[
                    "S256",
                    "S256",
                    "S256",
                    "S256",
                    "S256",
                    "S256",
                    "S256",
                    int,
                    int,
                    bool,
                ],
            ),
            f"{self.model_manager.model_nickname}/parameters"
        )


        self.IRF = self.IndexTable(
            QTable(
                names=[
                    "config",
                    "cuts_file",
                    "irf_file",
                    "benckmark_file",
                    "zenith",
                    "azimuth",
                ],
                dtype=["S256", "S256", "S256", "S256", float, float],
                units=[None, None, None, None, "deg", "deg"],
            ),
            f"{self.model_manager.model_nickname}/IRF"
        )


        self.DL2_DATA = self.IndexTable(
            QTable(
                names=["DL2_files", "DL2_zenith_distances", "DL2_azimuths"],
                dtype=["S256", float, float],
            ),
            f"{self.model_manager.model_nickname}/DL2/Data"
        )


    class IndexTable:
        def __init__(self, default_table: QTable, table_path: str):
            self.default_table = default_table
            self.table_path = table_path
