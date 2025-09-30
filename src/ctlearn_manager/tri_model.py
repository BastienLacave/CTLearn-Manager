"""CTLearnTriModelManager class for handling three CTLearn models: direction, energy, and type."""

import ast
from pathlib import Path

import astropy.units as u
import numpy as np
from tqdm import tqdm

from .io.io import load_DL2_data_MC, load_true_shower_parameters
from .model_manager import CTLearnModelManager, DataSample
from .utils.utils import (
    ClusterConfiguration,
    Cuts,
    CutType,
    DefaultCuts,
    ParticleType,
    angular_distance,
    plot_pointing_on_ax,
    set_mpl_style,
    ExportCurves,
    CurveType,
    CTLMDirectories,
    IRFType,
    CutType,
    get_irf_type_from_config,
    get_color,
    convert_irf_format,
)

__all__ = [
    "CTLearnTriModelManager",
]


class CTLearnTriModelManager:
    """
    A manager class for handling three CTLearn models: direction, energy, and type.

    Attributes
    ----------
        direction_model (CTLearnModelManager): The direction model manager.
        energy_model (CTLearnModelManager): The energy model manager.
        type_model (CTLearnModelManager): The type model manager.

    Methods
    -------
        __init__(direction_model, energy_model, type_model):
            Initializes the CTLearnTriModelManager with the given models.
        launch_testing():
            Placeholder method for launching testing.
        produce_irfs():
            Placeholder method for producing IRFs.
        plot_irfs():
            Uses gammapy to plot the IRFs. (Not yet implemented)
        plot_loss():
            Plots the training and validation loss for each model using matplotlib.
    """

    def __init__(
        self,
        direction_model: CTLearnModelManager,
        energy_model: CTLearnModelManager,
        type_model: CTLearnModelManager,
        project_directories:CTLMDirectories,
        cluster_configuration=ClusterConfiguration(),
    ):
        """
        Initialize the CTLearnTriModelManager.

        Parameters
        ----------
        direction_model : CTLearnModelManager
            The model manager for direction reconstruction. Must be a direction model.
        energy_model : CTLearnModelManager
            The model manager for energy reconstruction. Must be an energy model.
        type_model : CTLearnModelManager
            The model manager for type reconstruction. Must be a type model.
        cluster_configuration : ClusterConfiguration, optional
            The cluster configuration for the model manager. Defaults to a new instance of ClusterConfiguration.

        Raises
        ------
        ValueError
            If `direction_model` is not a direction model.
        ValueError
            If `energy_model` is not an energy model.
        ValueError
            If `type_model` is not a type model.
        ValueError
            If the channels of all models are not the same.
        ValueError
            If the stereo values of all models are not the same.
        ValueError
            If the telescope IDs of all models are not the same.
        ValueError
            If the minimum number of telescopes of all models are not the same.

        Attributes
        ----------
        direction_model : CTLearnModelManager
            The direction model manager.
        energy_model : CTLearnModelManager
            The energy model manager.
        type_model : CTLearnModelManager
            The type model manager.
        channels : list
            The channels used by the models.
        stereo : bool
            Indicates whether the models use stereo reconstruction.
        min_telescopes : int
            The minimum number of telescopes required by the models.
        telescope_ids : list
            The IDs of the telescopes used by the models.
        telescope_names : list
            The names of the telescopes used by the models.
        cluster_configuration : ClusterConfiguration
            The cluster configuration for the model manager.
        reconstruction_method : str
            The reconstruction method used, set to "CTLearn".
        reco_field_suffix : str
            The suffix for the reconstruction field, determined by the stereo value.
        """
        self.project_directories:CTLMDirectories  = project_directories
        if direction_model.model_parameters_table["reco"][0] in [
            "direction",
            "cameradirection",
            "skydirection",
        ]:
            self.direction_model = direction_model
            self.direction_model.cluster_configuration = cluster_configuration
        else:
            raise ValueError("direction_model must be a direction model")
        if energy_model.model_parameters_table["reco"][0] == "energy":
            self.energy_model = energy_model
            self.energy_model.cluster_configuration = cluster_configuration
        else:
            raise ValueError("energy_model must be an energy model")
        if type_model.model_parameters_table["reco"][0] == "type":
            self.type_model = type_model
            self.type_model.cluster_configuration = cluster_configuration
        else:
            raise ValueError("type_model must be a type model")
        import ast

        direction_channels = ast.literal_eval(
            self.direction_model.model_parameters_table["channels"][0]
        )
        energy_channels = ast.literal_eval(
            self.energy_model.model_parameters_table["channels"][0]
        )
        type_channels = ast.literal_eval(
            self.type_model.model_parameters_table["channels"][0]
        )
        if not (direction_channels == energy_channels == type_channels):
            raise ValueError("All models must have the same channels")
        else:
            self.channels = direction_channels

        if not (
            self.direction_model.stereo
            == self.energy_model.stereo
            == self.type_model.stereo
        ):
            raise ValueError(
                f"All models must have the same stereo value, direction model: {self.direction_model.stereo}, energy model: {self.energy_model.stereo}, type model: {self.type_model.stereo}"
            )
        else:
            self.stereo = self.direction_model.stereo
        if not (
            self.direction_model.telescope_ids
            == self.energy_model.telescope_ids
            == self.type_model.telescope_ids
        ):
            raise ValueError("All models must have the same telescope_ids")

        if not (
            self.direction_model.min_telescopes
            == self.energy_model.min_telescopes
            == self.type_model.min_telescopes
        ):
            raise ValueError("All models must have the same min_telescopes")
        else:
            self.min_telescopes = self.direction_model.min_telescopes
        self.telescope_ids = self.direction_model.telescope_ids
        self.telescope_names = self.direction_model.telescope_names
        self.cluster_configuration = cluster_configuration
        self.reconstruction_method = "CTLearn"
        self.reco_field_suffix = (
            self.reconstruction_method
            if self.stereo
            else f"{self.reconstruction_method}_tel"
        )
        self.set_keys()
        print(
            f"🧠🧠🧠 CTLearnTriModelManager ▮ {self.direction_model.model_nickname} ▮ {self.energy_model.model_nickname} ▮ {self.type_model.model_nickname} ▮"
        )
        self.get_available_MC_directions()
        # set_mpl_style()

    def set_keys(self):
        """
        Set the keys for various data fields used in the model.

        This method initializes attributes for accessing specific data fields
        such as reconstructed and true values for energy, altitude, azimuth,
        intensity, and time. The keys are determined based on the configuration
        of the model (e.g., whether stereo mode is enabled).

        Attributes
        ----------
        gammaness_key : str
            Key for the predicted gammaness value.
        reco_energy_key : str
            Key for the reconstructed energy value.
        intensity_key : str
            Key for the Hillas intensity value.
        reco_alt_key : str
            Key for the reconstructed altitude value.
        reco_az_key : str
            Key for the reconstructed azimuth value.
        true_alt_key : str
            Key for the true altitude value.
        true_az_key : str
            Key for the true azimuth value.
        true_energy_key : str
            Key for the true energy value.
        pointing_alt_key : str
            Key for the pointing altitude value, determined by stereo mode.
        pointing_az_key : str
            Key for the pointing azimuth value, determined by stereo mode.
        time_key : str
            Key for the time value.
        """
        self.gammaness_key = (
            f"{self.reco_field_suffix}_prediction"  # if self.CTLearn else "gammaness"
        )
        self.reco_energy_key = (
            f"{self.reco_field_suffix}_energy"  # if self.CTLearn else "reco_energy"
        )
        self.intensity_key = "hillas_intensity"  # if self.CTLearn else "intensity"
        self.reco_alt_key = (
            f"{self.reco_field_suffix}_alt"  # if self.CTLearn else "reco_alt"
        )
        self.reco_az_key = (
            f"{self.reco_field_suffix}_az"  # if self.CTLearn else "reco_az"
        )
        self.true_alt_key = "true_alt"  # if self.CTLearn else "alt"
        self.true_az_key = "true_az"  # if self.CTLearn else "az"
        self.true_energy_key = "true_energy"  # if self.CTLearn else "energy"
        # self.true_type_key = "true_type" #if self.CTLearn else "type"
        self.pointing_alt_key = (
            "array_altitude" if self.stereo else "altitude"
        )  # if self.CTLearn else "alt_tel"
        self.pointing_az_key = (
            "array_azimuth" if self.stereo else "azimuth"
        )  # if self.CTLearn else "az_tel"
        self.time_key = "time"  # if self.CTLearn else "dragon_time"

    def set_testing_data(self, testing_samples: list[DataSample]):
        """
        Set the testing data for the models.

        Parameters
        ----------
        testing_samples : list[DataSample]
            A list of `DataSample` objects to be used as testing data.

        Notes
        -----
        This method updates the testing data for each of the models
        (`direction_model`, `energy_model`, and `type_model`) using the provided
        testing samples. After updating, it retrieves the available testing
        directions.
        """
        for model in tqdm([self.direction_model, self.energy_model, self.type_model], desc="Setting testing data"):
            for data_sample in testing_samples:
                model.update_model_manager_testing_data(data_sample)
        self.get_available_testing_directions()

    # def set_DL2_MC_file(
    #     self, testing_MC_DL2_file: str, testing_MC_DL2_data_sample: DataSample
    # ):
    #     """
    #     Set the DL2 Monte Carlo (MC) file for testing and update associated models.

    #     Parameters
    #     ----------
    #     testing_MC_DL2_file : str
    #         Path to the DL2 MC file to be used for testing.
    #     testing_MC_DL2_data_sample : DataSample
    #         Data sample object containing the testing data.

    #     Notes
    #     -----
    #     This method updates the DL2 MC file and data sample for all associated models,
    #     including the direction, energy, and type models.
    #     """
    #     for model in [self.direction_model, self.energy_model, self.type_model]:
    #         model.update_model_manager_DL2_MC_file(
    #             testing_MC_DL2_file=testing_MC_DL2_file,
    #             testing_MC_DL2_data_sample=testing_MC_DL2_data_sample,
    #         )

    def delete_table_from_index(self, path: str):
        """
        Delete a table from the HDF5 file at the specified path.

        Parameters
        ----------
        path : str
            The path of the table to delete within the HDF5 file.

        Raises
        ------
        KeyError
            If the specified path does not exist in the HDF5 file.

        Notes
        -----
        This method modifies the HDF5 file in place by removing the specified table.
        Ensure that the path exists in the file before calling this method to avoid errors.
        """
        import h5py

        with h5py.File(self.direction_model.model_index_file, "r+") as f:
            del f[path]
            print(f"Table {path} erased from {self.direction_model.model_index_file}")

    def get_available_testing_directions(self):
        """
        Retrieve and display available testing directions for each particle type.

        This method reads testing data from an HDF5 file and extracts zenith and
        azimuth angles for each particle type. It identifies unique combinations
        of zenith and azimuth angles and prints them along with the particle types
        available for each combination.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - The method uses the `astropy.io.misc.hdf5.read_table_hdf5` function to
          read data from the HDF5 file.
        - If an exception occurs while reading data for a particle type, it assumes
          no data is available for that type.
        - The zenith and azimuth angles are sorted by zenith angle before being
          displayed.
        - The output is printed to the console.
        """
        from astropy.io.misc.hdf5 import read_table_hdf5

        zeniths = []
        azimuths = []

        for particle_type in ParticleType:
            try:
                DL2_table = read_table_hdf5(
                    self.project_directories.model_index_file,
                    path=f"{self.direction_model.model_nickname}/testing/{particle_type.value}",
                )
                _zeniths = DL2_table[f"testing_{particle_type.value}_zenith_distances"]
                _azimuths = DL2_table[f"testing_{particle_type.value}_azimuths"]
            except:
                _zeniths = []
                _azimuths = []
            zeniths.append(_zeniths)
            azimuths.append(_azimuths)

        flat_zeniths = [item for sublist in zeniths for item in sublist]
        flat_azimuths = [item for sublist in azimuths for item in sublist]

        coords = set(zip(flat_zeniths, flat_azimuths))
        coords = sorted(coords, key=lambda x: x[0])
        print(coords)
        if len(coords) > 0:
            print("Available testing directions:")
        for zenith, azimuth in coords:
            available_particles = []
            for i, particle_type in enumerate(ParticleType):
                particle_available = (zenith, azimuth) in set(
                    zip(zeniths[i], azimuths[i])
                )
                if particle_available:
                    available_particles.append(particle_type.value)
            if len(available_particles) > 0:
                print(
                    f"(ZD, Az): ({zenith.value} * u.deg, {azimuth.value} * u.deg)\t{' | '.join(available_particles)}"
                )
            else:
                print(f"(ZD, Az): ({zenith.value} * u.deg, {azimuth.value} * u.deg)")

    def get_available_MC_directions(self, verbose=True):
        """
        Retrieve and display available Monte Carlo (MC) directions.

        This method reads MC direction data from an HDF5 file for each particle type
        defined in the `ParticleType` enumeration. It extracts zenith and azimuth
        angles, combines them into unique coordinate pairs, and optionally prints
        the available directions along with the corresponding particle types.

        Parameters
        ----------
        verbose : bool, optional
            If True, print the available MC directions and their associated particle
            types. Default is True.

        Returns
        -------
        list of tuple
            A sorted list of unique (zenith, azimuth) coordinate pairs available
            in the MC data.

        Notes
        -----
        - The method handles missing or unavailable data gracefully by skipping
          particle types that do not have corresponding MC data.
        - The printed output, if `verbose` is True, includes zenith and azimuth
          angles in degrees and lists the particle types available for each
          coordinate pair.
        """
        from astropy.io.misc.hdf5 import read_table_hdf5

        zeniths = []
        azimuths = []

        for particle_type in ParticleType:
            _zeniths, _azimuths = self.project_directories.get_available_MC_directions(particle_type)
            # try:
            #     DL2_table = read_table_hdf5(
            #         self.direction_model.model_index_file,
            #         path=f"{self.direction_model.model_nickname}/DL2/MC/{particle_type.value}",
            #     )
            #     _zeniths = DL2_table[
            #         f"testing_DL2_{particle_type.value}_zenith_distances"
            #     ]
            #     _azimuths = DL2_table[f"testing_DL2_{particle_type.value}_azimuths"]
            # except:
            #     _zeniths = []
            #     _azimuths = []
            zeniths.append(_zeniths)
            azimuths.append(_azimuths)

        flat_zeniths = [item for sublist in zeniths for item in sublist]
        flat_azimuths = [item for sublist in azimuths for item in sublist]

        coords = set(zip(flat_zeniths, flat_azimuths))
        coords = sorted(coords, key=lambda x: x[0])
        if verbose:
            if len(coords) > 0:
                print("Available MC DL2 directions:")
            for zenith, azimuth in coords:
                available_particles = []
                for i, particle_type in enumerate(ParticleType):
                    particle_available = (zenith, azimuth) in set(
                        zip(zeniths[i], azimuths[i])
                    )
                    if particle_available:
                        available_particles.append(particle_type.value)
                if len(available_particles) > 0:
                    print(
                        f"(ZD, Az): ({zenith.value} * u.deg, {azimuth.value} * u.deg) \t {' | '.join(available_particles)}"
                    )
                else:
                    print(
                        f"(ZD, Az): ({zenith.value} * u.deg, {azimuth.value} * u.deg)"
                    )
        return coords

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def launch_testing(
        self,
        zenith: float,
        azimuth: float,
        # output_dirs: list[str],
        # config_dir: str | None = None,
        launch_particle_types: list[ParticleType] = [ParticleType.GAMMA_POINT],
        batch_size=64,
        dl2_subarray=True,
        force_dl1_lookup=False,
        overwrite=False,
        config=None,
    ):
        """
        Launch the testing process for CTLearn models.

        Parameters
        ----------
        zenith : float
            Zenith angle for the testing data.
        azimuth : float
            Azimuth angle for the testing data.
        output_dirs : list of str
            List of output directories corresponding to each particle type.
        config_dir : str or None, optional
            Directory for configuration files. Defaults to None.
        launch_particle_types : list of ParticleType, optional
            List of particle types to launch testing for. Defaults to [ParticleType.GAMMA_POINT].
        batch_size : int, optional
            Batch size for prediction. Defaults to 64.
        dl2_subarray : bool, optional
            Whether to include DL2 subarray predictions. Defaults to True.
        force_dl1_lookup : bool, optional
            Whether to force DL1 lookup during prediction. Defaults to False.
        overwrite : bool, optional
            Whether to overwrite existing output files. Defaults to False.
        config : dict or None, optional
            Additional configuration parameters. Defaults to None.

        Raises
        ------
        ValueError
            If the number of output directories does not match the number of particle types.
        ValueError
            If the cluster configuration has more than one node.
        ValueError
            If the testing directories for the models do not match.
        ValueError
            If no matching directory is found for the given zenith and azimuth.
        ValueError
            If the testing directories are empty.

        Notes
        -----
        This function handles the testing process for CTLearn models, including directory
        validation, file matching, and command execution for both stereo and mono models.
        It supports cluster-based execution using SLURM or local execution.
        """
        # assert len(output_dirs) == len(launch_particle_types), (
        #     "Output directories must match the number of launched particle types"
        # )

        if self.cluster_configuration.nodes > 1:
            raise ValueError("CTLearn prediction tool can only be ran on a single GPU")
        self.cluster_configuration.info()
        import glob
        import os

        from astropy.io.misc.hdf5 import read_table_hdf5

        testing_files = []
        output_files = []
        for particle_type in launch_particle_types:
            output_dir = self.project_directories.get_dl2_mc_directory(particle_type, zenith, azimuth)
            os.makedirs(output_dir, exist_ok=True)
            direction_testing_table = read_table_hdf5(
                self.project_directories.model_index_file,
                path=f"{self.direction_model.model_nickname}/testing/{particle_type.value}",
            )
            energy_testing_table = read_table_hdf5(
                self.project_directories.model_index_file,
                path=f"{self.energy_model.model_nickname}/testing/{particle_type.value}",
            )
            type_testing_table = read_table_hdf5(
                self.project_directories.model_index_file,
                path=f"{self.type_model.model_nickname}/testing/{particle_type.value}",
            )
            if (
                not (
                    direction_testing_table[f"testing_{particle_type.value}_dirs"]
                    == energy_testing_table[f"testing_{particle_type.value}_dirs"]
                ).all()
                and (
                    direction_testing_table[f"testing_{particle_type.value}_dirs"]
                    == type_testing_table[f"testing_{particle_type.value}_dirs"]
                ).all()
            ):
                raise ValueError(
                    f"All models must have the same testing {particle_type.value} directories, use set_testing_files to set them"
                )
            if len(direction_testing_table[f"testing_{particle_type.value}_dirs"]) == 0:
                raise ValueError(
                    f"Testing {particle_type.value} directories cannot be empty"
                )
            dirs = direction_testing_table[f"testing_{particle_type.value}_dirs"]
            zeniths = direction_testing_table[
                f"testing_{particle_type.value}_zenith_distances"
            ]
            azimuths = direction_testing_table[
                f"testing_{particle_type.value}_azimuths"
            ]
            patterns = direction_testing_table[
                f"testing_{particle_type.value}_patterns"
            ]

            matching_dirs = [
                dirs[i]
                for i in range(len(dirs))
                if zeniths[i] == zenith and azimuths[i] == azimuth
            ]
            if not matching_dirs:
                raise ValueError(
                    f"No matching {particle_type.value} directory found for zenith {zenith} and azimuth {azimuth}"
                )
            dir = matching_dirs[0]
            pattern = [
                patterns[i]
                for i in range(len(patterns))
                if zeniths[i] == zenith and azimuths[i] == azimuth
            ][0]
            data_sample = DataSample(
                directory=dir,
                zenith_distance=zenith,
                azimuth=azimuth,
                pattern=pattern,
                particle_type=particle_type,
            )
            _files = np.sort(glob.glob(f"{dir}/{pattern}"))
            _output_files = [
                f"{output_dir}/{Path(file).stem.replace('dl1', 'dl2')}.h5"
                for file in _files
            ]
            testing_files.extend(_files)
            output_files.extend(_output_files)
            # for model in [self.direction_model, self.energy_model, self.type_model]:
            #     for file in _output_files:
            #         model.update_model_manager_DL2_MC_file(
            #             testing_MC_DL2_file=file, testing_MC_DL2_data_sample=data_sample
            #         )
        channels_string = ""
        for channel in self.channels:
            channels_string += f"--DLImageReader.channels={channel} "
        type_model_dir = np.sort(
            glob.glob(
                f"{self.type_model.model_parameters_table['model_dir'][0]}/{self.type_model.model_nickname}*"
            )
        )[-1]
        energy_model_dir = np.sort(
            glob.glob(
                f"{self.energy_model.model_parameters_table['model_dir'][0]}/{self.energy_model.model_nickname}*"
            )
        )[-1]
        direction_model_dir = np.sort(
            glob.glob(
                f"{self.direction_model.model_parameters_table['model_dir'][0]}/{self.direction_model.model_nickname}*"
            )
        )[-1]

        dl2_subarray_string = (
            " --dl2-subarray" if dl2_subarray else " --no-dl2-subarray"
        )
        force_dl1_lookup_string = (
            "--DLImageReader.force_dl1_lookup=True" if force_dl1_lookup else ""
        )
        config_string = f"--config {config}" if config is not None else ""

        allowed_tels = ast.literal_eval(
            self.direction_model.model_parameters_table["telescope_ids"][0]
        )
        # config['TrainCTLearnModel']['DLImageReader']['allowed_tels'] = allowed_tels # TODO pass allowed tels in a config file

        for input_file, output_file in zip(testing_files, output_files):
            if os.path.exists(output_file) and not overwrite:
                print(
                    f"Output file {output_file} already exists, skipping, set overwrite=True to overwrite"
                )
                continue
            if self.stereo:
                cmd = f"ctlearn-predict-stereo-model --input_url {input_file} \
--PredictCTLearnModel.batch_size={batch_size} \
--type_model={type_model_dir}/ctlearn_model.cpk \
--energy_model={energy_model_dir}/ctlearn_model.cpk \
--skydirection_model={direction_model_dir}/ctlearn_model.cpk \
--use-HDF5Merger \
--no-dl1-images --no-true-images --output {output_file} \
--DLImageReader.mode=stereo --PredictCTLearnModel.stack_telescope_images=True --DLImageReader.min_telescopes={self.min_telescopes} \
--PredictCTLearnModel.overwrite_tables=True -v {channels_string} {force_dl1_lookup_string} \
{config_string}"# --overwrite={overwrite}"
            else:
                # cmd = f"ctlearn-predict-mono --input_url {input_file} --type_model={type_model_dir}/ctlearn_model.cpk --energy_model={energy_model_dir}/ctlearn_model.cpk --direction_model={direction_model_dir}/ctlearn_model.cpk --no-dl1-images --no-true-images --output {output_file} --overwrite -v {channels_string}"
                cmd = f"ctlearn-predict-mono-model --input_url {input_file} \
--PredictCTLearnModel.batch_size={batch_size} \
--type_model={type_model_dir}/ctlearn_model.cpk \
--energy_model={energy_model_dir}/ctlearn_model.cpk \
--cameradirection_model={direction_model_dir}/ctlearn_model.cpk \
--no-dl1-images --no-true-images --output {output_file} \
--use-HDF5Merger{dl2_subarray_string} \
--PredictCTLearnModel.overwrite_tables=True -v {channels_string} {force_dl1_lookup_string} \
{config_string}"# --overwrite={overwrite}"

            if self.cluster_configuration.use_cluster:
                # sbatch_file = write_sbatch_script(cluster_configuration.cluster, Path(input_file).stem, cmd, config_dir, env_name=cluster_configuration.python_env, account=cluster_configuration.account)
                config_dir = self.project_directories.prediction_logs_directory
                sbatch_file = self.cluster_configuration.write_sbatch_script(
                    Path(input_file).stem, cmd, config_dir
                )
                os.system(f"sbatch {sbatch_file}")
            else:
                print(cmd)
                os.system(cmd)

    def predict_lstchain_data(
        self,
        input_file,
        output_file,
        run=None,
        subrun=None,
        config_dir=None,
        overwrite=False,
        pointing_table="/dl1/event/telescope/parameters/LST_LSTCam",
        batch_size=64,
    ):
        """
        Predict DL2 data from DL1 input using CTLearn models for LST-1 data.

        Parameters
        ----------
        input_file : str
            Path to the input DL1 file.
        output_file : str
            Path to the output DL2 file.
        run : int, optional
            Run number to override observation ID. Default is None.
        subrun : int, optional
            Subrun number to override observation ID. Default is None.
        config_dir : str, optional
            Directory to save the configuration file. Default is None.
        overwrite : bool, optional
            Whether to overwrite the output file if it already exists. Default is False.
        pointing_table : str, optional
            Path to the pointing table in the input file. Default is
            "/dl1/event/telescope/parameters/LST_LSTCam".
        batch_size : int, optional
            Batch size for prediction. Default is 64.

        Raises
        ------
        ValueError
            If the cluster configuration specifies more than one node, as the
            CTLearn prediction tool can only run on a single GPU.

        Notes
        -----
        - This method generates a configuration file for the CTLearn prediction tool
          and executes the prediction command.
        - If a cluster configuration is enabled, the prediction command is submitted
          as a SLURM job using an SBATCH script.
        - The method assumes the presence of pre-trained CTLearn models for event
          type, energy, and direction predictions.
        """
        if self.cluster_configuration.nodes > 1:
            raise ValueError("CTLearn prediction tool can only be ran on a single GPU")

        import ast
        import glob
        import json
        import os

        os.system(f"mkdir -p {output_file.rsplit('/', 1)[0]}")
        channels_string = ""
        for channel in self.channels:
            channels_string += f"--DLImageReader.channels {channel} "
        # print(f"{self.project_directories.type_model_directory}/{self.project_directories.tri_model_nickname}_type/{self.project_directories.tri_model_nickname}_type_v*")
        # type_model_dir = np.sort(
        #     glob.glob(
        #         f"{self.project_directories.type_model_directory}/{self.type_model.model_nickname}_v*"
        #     )
        # )[-1]
        # energy_model_dir = np.sort(
        #     glob.glob(
        #         f"{self.project_directories.energy_model_directory}/{self.energy_model.model_nickname}_v*"
        #     )
        # )[-1]
        # direction_model_dir = np.sort(
        #     glob.glob(
        #         f"{self.project_directories.direction_model_directory}/{self.direction_model.model_nickname}_v*"
        #     )
        # )[-1]
        type_model_dir = self.project_directories.latest_type_model_directory
        energy_model_dir = self.project_directories.latest_energy_model_directory
        direction_model_dir = self.project_directories.latest_direction_model_directory
        allowed_tels = ast.literal_eval(
            self.direction_model.model_parameters_table["telescope_ids"][0]
        )
        stereo_mode = "stereo" if self.stereo else "mono"
        # stack_telescope_images = True if self.stereo else False
        config = {}
        config["LST1PredictionTool"] = {}

        # config['LST1PredictionTool']['allowed_tels'] = allowed_tels
        # config['LST1PredictionTool']['min_telescopes'] = int(len(allowed_tels))
        # config['LST1PredictionTool']['mode'] = stereo_mode
        # config['LST1PredictionTool']['stack_telescope_images'] = stack_telescope_images # Mono only
        config["LST1PredictionTool"]["channels"] = self.channels
        # config['LST1PredictionTool']['dl1dh_reader_type'] = "DLImageReader"
        if (run is not None) and (subrun is not None):
            config["LST1PredictionTool"]["override_obs_id"] = int(
                f"{run:05d}{subrun:04d}"
            )
        config["LST1PredictionTool"]["output_path"] = output_file
        config["LST1PredictionTool"]["log_file"] = output_file.replace(".h5", ".log")
        config["LST1PredictionTool"]["overwrite"] = overwrite

        if config_dir is None:
            config_dir = self.project_directories.prediction_logs_directory

        config_file = f"{config_dir}/pred_config_{Path(input_file).stem}.json"
        with open(config_file, "w") as file:
            json.dump(config, file)
        print(f"Configuration saved to {config_file}")

        # avg_data_ze, avg_data_az = get_avg_pointing(input_file, pointing_table=pointing_table)
        # for model in [self.direction_model, self.energy_model, self.type_model]:
        #     model.update_model_manager_DL2_data_files(
        #         [output_file],
        #         [avg_data_ze],
        #         [avg_data_az],
        #     )

        cmd = f"ctlearn-predict-LST1 --input_url {input_file} \
--type_model {type_model_dir}/ctlearn_model.cpk \
--energy_model {energy_model_dir}/ctlearn_model.cpk \
--cameradirection_model {direction_model_dir}/ctlearn_model.cpk \
--config '{config_file}' --LST1PredictionTool.batch_size={batch_size} \
-v"

        if self.cluster_configuration.use_cluster:
            sbatch_file = self.cluster_configuration.write_sbatch_script(
                Path(input_file).stem, cmd, config_dir
            )
            import os

            os.system(f"sbatch {sbatch_file}")

        else:
            print(cmd)
            os.system(cmd)

        print("")

    def predict_data(
        self,
        input_file,
        output_file,
        config_dir=None,
        overwrite=False,
        pointing_table="dl0/monitoring/subarray/pointing",
    ):
        """
        Predict data using CTLearn models and save the results.

        Parameters
        ----------
        input_file : str
            Path to the input file containing the data to be processed.
        output_file : str
            Path to the output file where the prediction results will be saved.
        config_dir : str, optional
            Directory where the configuration file will be saved. Defaults to None.
        overwrite : bool, optional
            Whether to overwrite existing output files. Defaults to False.
        pointing_table : str, optional
            Path to the pointing table within the input file. Defaults to
            "dl0/monitoring/subarray/pointing".

        Raises
        ------
        ValueError
            If the cluster configuration specifies more than one node, as CTLearn
            prediction can only run on a single GPU.

        Notes
        -----
        This method generates a configuration file for CTLearn, determines the
        appropriate model directories, and constructs a command to run the
        CTLearn prediction tool. If a cluster is configured, the command is
        submitted as a job using SLURM; otherwise, it is executed locally.
        """
        if self.cluster_configuration.nodes > 1:
            raise ValueError("CTLearn prediction tool can only be ran on a single GPU")

        import ast
        import glob
        import json
        import os

        from .utils.utils import get_avg_pointing

        os.system(f"mkdir -p {output_file.rsplit('/', 1)[0]}")
        channels_string = ""
        for channel in self.channels:
            channels_string += f"--DLImageReader.channels {channel} "
        type_model_dir = np.sort(
            glob.glob(
                f"{self.type_model.model_parameters_table['model_dir'][0]}/{self.type_model.model_nickname}_v*"
            )
        )[-1]
        energy_model_dir = np.sort(
            glob.glob(
                f"{self.energy_model.model_parameters_table['model_dir'][0]}/{self.energy_model.model_nickname}_v*"
            )
        )[-1]
        direction_model_dir = np.sort(
            glob.glob(
                f"{self.direction_model.model_parameters_table['model_dir'][0]}/{self.direction_model.model_nickname}_v*"
            )
        )[-1]
        allowed_tels = ast.literal_eval(
            self.direction_model.model_parameters_table["telescope_ids"][0]
        )
        stereo_mode = "stereo" if self.stereo else "mono"
        stack_telescope_images = True if self.stereo else False
        config = {}
        config["PredictCTLearnModel"] = {}
        config["PredictCTLearnModel"]["DLImageReader"] = {}

        config["PredictCTLearnModel"]["DLImageReader"]["allowed_tels"] = allowed_tels
        config["PredictCTLearnModel"]["DLImageReader"]["min_telescopes"] = int(
            len(allowed_tels)
        )
        config["PredictCTLearnModel"]["DLImageReader"]["mode"] = stereo_mode
        config["PredictCTLearnModel"]["stack_telescope_images"] = stack_telescope_images
        config["PredictCTLearnModel"]["DLImageReader"]["channels"] = self.channels
        config["PredictCTLearnModel"]["dl1dh_reader_type"] = "DLImageReader"
        config["PredictCTLearnModel"]["output_path"] = output_file
        config["PredictCTLearnModel"]["log_file"] = output_file.replace(".h5", ".log")
        config["PredictCTLearnModel"]["overwrite"] = overwrite

        config_file = f"{config_dir}/pred_config_{Path(input_file).stem}.json"
        with open(config_file, "w") as file:
            json.dump(config, file)
        print(f"Configuration saved to {config_file}")

        avg_data_ze, avg_data_az = get_avg_pointing(
            input_file, pointing_table=pointing_table
        )
        # for model in [self.direction_model, self.energy_model, self.type_model]:
        #     model.update_model_manager_DL2_data_files(
        #         [output_file],
        #         [avg_data_ze],
        #         [avg_data_az],
        #     )

        cmd = f"ctlearn-predict-model --input_url {input_file} \
--type_model {type_model_dir}/ctlearn_model.cpk \
--energy_model {energy_model_dir}/ctlearn_model.cpk \
--direction_model {direction_model_dir}/ctlearn_model.cpk \
--config '{config_file}' \
--no-dl1-images --no-true-images \
--dl1-features \
--PredictCTLearnModel.overwrite_tables True -v"

        if self.cluster_configuration.use_cluster:
            sbatch_file = self.cluster_configuration.write_sbatch_script(
                Path(input_file).stem, cmd, config_dir
            )
            os.system(f"sbatch {sbatch_file}")
        else:
            print(cmd)
            os.system(cmd)

        print("")

    # TODO add option to delete original files
    def merge_DL2_files(
        self,
        zenith: str,
        azimuth: str,
        particle_type: ParticleType,
        overwrite=False,
    ):
        """
        Merge DL2 files for a specific zenith, azimuth, and particle type.

        This method retrieves DL2 Monte Carlo (MC) files for the specified
        zenith, azimuth, and particle type. If multiple files are found,
        they are merged into a single output file using the `ctapipe-merge`
        command. The merged file is then registered with the direction,
        energy, and type models. If only one file exists, no merging is
        performed.

        Parameters
        ----------
        zenith : str
            The zenith angle of the observation.
        azimuth : str
            The azimuth angle of the observation.
        output_file : str
            The path to the output file where merged data will be saved.
        particle_type : ParticleType
            The type of particle (e.g., gamma, proton) for which DL2 files
            are being merged.
        overwrite : bool, optional
            Whether to overwrite the output file if it already exists
            (default is False).

        Raises
        ------
        RuntimeError
            If the merging process fails.

        Notes
        -----
        - The `ctapipe-merge` command is used for merging files.
        - Original files are not deleted after merging.
        - If only one file exists for the given parameters, merging is skipped.
        """
        import os
        import glob

        output_directory = self.project_directories.get_dl2_mc_merged_directory(particle_type, zenith, azimuth)
        os.makedirs(output_directory, exist_ok=True)
        output_file = f"{output_directory}/merged_{particle_type.value}_zenith_{zenith.value}_azimuth_{azimuth.value}.dl2.h5"
        # files = self.project_directories.get_dl2_mc_files(
        #     zenith, azimuth, merged=False, particle_types=[particle_type]
        # )[particle_type.value]
        files = glob.glob(f"{self.project_directories.get_dl2_mc_directory(particle_type, zenith, azimuth)}/*.h5")
        if len(files) > 1:
            print(
                f"🔀 Merging DL2 {particle_type.value} files for zenith {zenith} and azimuth {azimuth}"
            )
            cmd = f"ctapipe-merge {' '.join(files)} --output={output_file} --progress --MergeTool.skip_broken_files=True {'--overwrite' if overwrite else ''}"
            print(f"Running : {cmd}")
            result = os.system(cmd)
            if result == 0:
                # for model in [
                #     self.direction_model,
                #     self.energy_model,
                #     self.type_model,
                # ]:
                #     model.update_merged_DL2_MC_files(
                #         zenith, azimuth, output_file, particle_type
                #     )
                # self.direction_model.update_merged_DL2_MC_files(
                #     zenith, azimuth, output_file, particle_type
                # )
                # self.energy_model.update_merged_DL2_MC_files(
                #     zenith, azimuth, output_file, particle_type
                # )
                # self.type_model.update_merged_DL2_MC_files(
                #     zenith, azimuth, output_file, particle_type
                # )
                print("Original files still exist and were not erased.")
            else:
                print(
                    f"Error: Failed to merge gamma files for zenith {zenith} and azimuth {azimuth}"
                )
        elif len(files) == 1:
            # for model in [
            #         self.direction_model,
            #         self.energy_model,
            #         self.type_model,
            #     ]:
            #         model.update_merged_DL2_MC_files(
            #             zenith, azimuth, files[0], particle_type
            #         )
            cmd = f"cp {files[0]} {output_file}"
            result = os.system(cmd)
            assert result == 0, (
                f"Error: Failed to copy file {files[0]} to {output_file}"
            )
            print(
                f"✅ There is a single {particle_type.value} file for zenith {zenith} and azimuth {azimuth}, file copied to 'merged'."
            )
        else:
            raise ValueError(
                f"No DL2 MC files found for zenith {zenith} and azimuth {azimuth} for particle type {particle_type.value}"
            )

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def plot_DL2_classification(
        self,
        zenith: float,
        azimuth: float,
        particle_types: list[ParticleType] = [
            ParticleType.GAMMA_POINT,
            ParticleType.PROTON,
        ],
    ):
        """
        Plot the DL2 classification results for given zenith and azimuth angles.

        Parameters
        ----------
        zenith : float
            The zenith angle of the simulated events.
        azimuth : float
            The azimuth angle of the simulated events.
        particle_types : list of ParticleType, optional
            A list of particle types to include in the plot. Defaults to
            [ParticleType.GAMMA_POINT, ParticleType.PROTON].

        Notes
        -----
        This method retrieves DL2 Monte Carlo (MC) files for the specified
        zenith and azimuth angles and particle types. It then loads the DL2
        data, extracts the `gammaness` values, and plots their distribution
        as histograms. The histograms are normalized to represent densities.
        The `gammaness` key is used to assess the classification performance
        of the model, where higher values typically indicate a higher
        likelihood of the event being a gamma-ray.
        The plot is displayed using `matplotlib.pyplot.show()`.

        See Also
        --------
        direction_model.get_DL2_MC_files : Retrieves the DL2 MC files for
            the specified parameters.
        load_DL2_data_MC : Loads DL2 data from a given file.
        """
        import matplotlib.pyplot as plt
        from astropy.table import vstack

        DL2_MC_files = self.project_directories.get_dl2_mc_files(
            zenith, azimuth, particle_types=particle_types
        )
        for particle_type in particle_types:
            testing_DL2_files = DL2_MC_files[particle_type.value]
            dl2_data = []
            tel_id = None if self.stereo else self.telescope_ids[0]
            for file in testing_DL2_files:
                dl2_data.append(load_DL2_data_MC(file, tel_id=tel_id))
            dl2_data = vstack(dl2_data)
            plt.hist(
                dl2_data[self.gammaness_key],
                bins=100,
                range=(0, 1),
                histtype="step",
                density=True,
                label=particle_type.value,
            )
        plt.xlabel("Gammaness")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def plot_DL2_energy(
        self,
        zenith: float,
        azimuth: float,
        particle_types: list[ParticleType] = [
            ParticleType.GAMMA_POINT,
            ParticleType.PROTON,
        ],
    ):
        """
        Plot the distribution of reconstructed DL2 energy for given zenith, azimuth, and particle types.

        Parameters
        ----------
        zenith : float
            Zenith angle of the simulated events in degrees.
        azimuth : float
            Azimuth angle of the simulated events in degrees.
        particle_types : list of ParticleType, optional
            List of particle types to include in the plot. Defaults to
            [ParticleType.GAMMA_POINT, ParticleType.PROTON].

        Notes
        -----
        This method retrieves the DL2 Monte Carlo (MC) files for the specified
        zenith, azimuth, and particle types, loads the data, and plots the
        reconstructed energy distribution as a histogram. The energy is displayed
        on a logarithmic scale for both axes.
        The method uses the `self.reco_energy_key` attribute to access the
        reconstructed energy values in the data.
        If `self.stereo` is True, all telescopes are used; otherwise, only the
        telescope specified by `self.telescope_ids[0]` is used.

        Raises
        ------
        KeyError
            If the particle type is not found in the DL2 MC files.
        """
        import matplotlib.pyplot as plt
        from astropy.table import vstack

        DL2_MC_files = self.project_directories.get_dl2_mc_files(
            zenith, azimuth, particle_types=particle_types
        )
        for particle_type in particle_types:
            testing_DL2_files = DL2_MC_files[particle_type.value]
            dl2_data = []
            tel_id = None if self.stereo else self.telescope_ids[0]
            for file in testing_DL2_files:
                dl2_data.append(load_DL2_data_MC(file, tel_id=tel_id))
            dl2_data = vstack(dl2_data)
            log_bins = np.logspace(
                np.log10(dl2_data[self.reco_energy_key].min()),
                np.log10(dl2_data[self.reco_energy_key].max()),
                100,
            )
            plt.hist(
                dl2_data[self.reco_energy_key],
                bins=log_bins,
                range=(0, 1),
                histtype="step",
                density=True,
                label=particle_type.value,
            )
        plt.xlabel("Energy [TeV]")
        plt.ylabel("Density")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.show()

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def plot_DL2_AltAz(
        self,
        zenith: float,
        azimuth: float,
        particle_types: list[ParticleType] = [ParticleType.GAMMA_POINT],
        cuts: Cuts = DefaultCuts.NO_CUTS.value,
    ):
        """
        Plot the reconstructed Altitude-Azimuth distribution for DL2 data.

        Parameters
        ----------
        zenith : float
            Zenith angle of the simulated observation in degrees.
        azimuth : float
            Azimuth angle of the simulated observation in degrees.
        particle_types : list of ParticleType, optional
            List of particle types to include in the plot. Defaults to [ParticleType.GAMMA_POINT].
        cuts : Cuts, optional
            Cuts to apply to the data. Must be of type `CutType.GLOBAL`. Defaults to `DefaultCuts.NO_CUTS.value`.

        Raises
        ------
        ValueError
            If the provided cuts are not of type `CutType.GLOBAL`.

        Notes
        -----
        This method visualizes the reconstructed altitude and azimuth of events
        for the specified particle types. It uses a 2D histogram to represent
        the density of reconstructed events and overlays the array pointing
        direction as a scatter point.
        The method assumes that the `direction_model` attribute provides access
        to the DL2 Monte Carlo files and that the `load_DL2_data_MC` function
        is available for loading the data.
        The plot is displayed using Matplotlib and includes a color bar to
        indicate the event counts in the 2D histogram.
        """
        import matplotlib.pyplot as plt
        from astropy.table import vstack

        if cuts.cut_type != CutType.GLOBAL:
            raise ValueError("Cuts must be global")

        fig, axs = plt.subplots(
            1, len(particle_types), figsize=(5 * len(particle_types), 4)
        )
        DL2_MC_files = self.project_directories.get_dl2_mc_files(
            zenith, azimuth, particle_types=particle_types
        )
        for i, particle_type in enumerate(particle_types):
            testing_DL2_files = DL2_MC_files[particle_type.value]
            dl2_data = []
            tel_id = None if self.stereo else self.telescope_ids[0]
            for file in testing_DL2_files:
                dl2_data.append(load_DL2_data_MC(file, tel_id=tel_id))
            dl2_data = vstack(dl2_data)
            dl2_data = dl2_data[dl2_data[self.gammaness_key] > cuts.gammaness_cut]
            if len(particle_types) > 1:
                ax = axs[i]
            else:
                ax = axs
            ax.scatter(
                dl2_data[self.pointing_alt_key][0] / np.pi * 180,
                dl2_data[self.pointing_az_key][0] / np.pi * 180,
                color=get_color("ctlearn_accent_1"),
                label="Array pointing",
                marker="o",
                s=80,
                edgecolor=get_color("ctlearn_accent_1"),
                facecolor="none",
            )
            ax.hist2d(
                dl2_data[self.reco_alt_key],
                dl2_data[self.reco_az_key],
                bins=100,
                zorder=0,
                # cmap="viridis",
                norm=plt.cm.colors.LogNorm(),
            )
            ax.set_xlabel("Altitude [deg]")
            ax.set_ylabel("Azimuth [deg]")
            ax.legend()
            ax.set_title(particle_type.value)
            cbar = plt.colorbar(ax.collections[1], ax=ax)
            cbar.set_label("Counts")
        plt.tight_layout()
        plt.show()

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def plot_migration_matrix(
        self,
        zenith: float,
        azimuth: float,
        particle_types: list[ParticleType] = [ParticleType.GAMMA_POINT],
        cuts: Cuts = DefaultCuts.NO_CUTS.value,
    ):
        """
        Plot the migration matrix for reconstructed vs true energy.

        Parameters
        ----------
        zenith : float
            Zenith angle of the observation in degrees.
        azimuth : float
            Azimuth angle of the observation in degrees.
        particle_types : list[ParticleType], optional
            List of particle types to include in the plot. Default is
            [ParticleType.GAMMA_POINT].
        cuts : Cuts, optional
            Cuts to apply to the data. Must be of type `CutType.GLOBAL`.
            Default is `DefaultCuts.NO_CUTS.value`.

        Raises
        ------
        ValueError
            If the provided cuts are not of type `CutType.GLOBAL`.

        Notes
        -----
        This method generates a 2D histogram (migration matrix) comparing
        reconstructed energy to true energy for the specified particle types.
        The plot is displayed using logarithmic scales for both axes. The
        method also overlays a diagonal line representing perfect reconstruction
        and applies the specified cuts to the data.
        The method uses Matplotlib for plotting and Astropy for data manipulation.
        """
        import matplotlib.pyplot as plt
        from astropy.table import join, vstack

        if cuts.cut_type != CutType.GLOBAL:
            raise ValueError("Cuts must be global")

        fig, axs = plt.subplots(
            1, len(particle_types), figsize=(5 * len(particle_types), 4)
        )
        DL2_MC_files = self.project_directories.get_dl2_mc_files(
            zenith, azimuth, particle_types=particle_types
        )
        for i, particle_type in enumerate(particle_types):
            testing_DL2_files = DL2_MC_files[particle_type.value]
            dl2_data = []
            shower_parameters = []
            tel_id = None if self.stereo else self.telescope_ids[0]
            for file in testing_DL2_files:
                dl2_data.append(load_DL2_data_MC(file, tel_id=tel_id))
                shower_parameters.append(load_true_shower_parameters(file))
            dl2_data = vstack(dl2_data)
            shower_parameters = vstack(shower_parameters)
            dl2_data = join(dl2_data, shower_parameters, keys=["obs_id", "event_id"])[
                dl2_data[self.gammaness_key] > cuts.gammaness_cut
            ]

            log_bins = np.logspace(
                np.log10(
                    min(
                        (
                            min(dl2_data[self.reco_energy_key]),
                            min(dl2_data[self.true_energy_key]),
                        )
                    )
                ),
                np.log10(
                    max(
                        max(dl2_data[self.reco_energy_key]),
                        max(dl2_data[self.true_energy_key]),
                    )
                ),
                100,
            )
            if len(particle_types) > 1:
                ax = axs[i]
            else:
                ax = axs
            cuts.plot_cuts_info_plt(ax)
            ax.plot(
                [log_bins[0], log_bins[-1]],
                [log_bins[0], log_bins[-1]],
                color=get_color("ctlearn_accent_1"),
                ls="--",
            )
            ax.hist2d(
                dl2_data[self.reco_energy_key],
                dl2_data[self.true_energy_key],
                bins=log_bins,
                # cmap="viridis",
                norm=plt.cm.colors.LogNorm(),
            )
            ax.set_xlabel("CTLean Energy [TeV]")
            ax.set_ylabel("True Energy [TeV]")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(log_bins[0], log_bins[-1])
            ax.set_ylim(log_bins[0], log_bins[-1])
            ax.axis("equal")
            ax.set_title(f"{particle_type.value}")
            cbar = plt.colorbar(ax.collections[0], ax=ax)
            cbar.set_label("Counts")

        plt.tight_layout()
        plt.show()

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def produce_irfs(
        self,
        zenith: float,
        azimuth: float,
        config: str,
        pointlike=True,
        electrons=False,
        protons=True,
        overwrite=False,
    ):
        """
        Produce Instrument Response Functions (IRFs) for given observational parameters.

        This method generates IRFs, cuts files, and benchmark files for a specified
        zenith and azimuth angle using the provided configuration. It supports
        processing gamma, electron, and proton particle types.

        Parameters
        ----------
        zenith : float
            Zenith angle of the observation in degrees.
        azimuth : float
            Azimuth angle of the observation in degrees.
        config : str
            Path to the configuration file for IRF generation.
        pointlike : bool, optional
            If True, use point-like gamma files. If False, use diffuse gamma files.
            Default is True.
        electrons : bool, optional
            If True, include electron files in the IRF generation. Default is False.
        protons : bool, optional
            If True, include proton files in the IRF generation. Default is True.
        overwrite : bool, optional
            If True, overwrite existing output files. Default is False.

        Raises
        ------
        ValueError
            If multiple files are found for a particle type or if required parameters
            are missing.
        RuntimeError
            If the system commands for generating cuts or IRFs fail.

        Notes
        -----
        - Ensure that the configuration file and input files are not moved or deleted
          as they are extensively used in the code for plotting and analysis.
        - Use `CTLearnTriModelManager.merge_DL2_files()` to merge multiple files
          before calling this method if necessary.
        """
        import os
        from pathlib import Path

        if pointlike:
            gamma_files = self.project_directories.get_dl2_mc_files(
                zenith, azimuth, particle_types=[ParticleType.GAMMA_POINT], merged=True
            )[ParticleType.GAMMA_POINT.value]
        else:
            gamma_files = self.project_directories.get_dl2_mc_files(
                zenith, azimuth, particle_types=[ParticleType.GAMMA_DIFFUSE], merged=True
            )[ParticleType.GAMMA_DIFFUSE.value]
        if len(gamma_files) > 1:
            raise ValueError(
                f"Multiple files found for gamma, zenith {zenith} and azimuth {azimuth}, please merge them first with CTLearnTriModelManager.merge_DL2_files()"
            )
        gamma_file = gamma_files[0]
        if electrons:
            electrons_files = self.project_directories.get_dl2_mc_files(
                zenith, azimuth, particle_types=[ParticleType.ELECTRON], merged=True
            )[ParticleType.ELECTRON.value]
            if len(electrons_files) > 1:
                raise ValueError(
                    f"Multiple files found for electrons, zenith {zenith} and azimuth {azimuth}, please merge them first with CTLearnTriModelManager.merge_DL2_files()"
                )
            electron_file = electrons_files[0]

        if protons:
            proton_files = self.project_directories.get_dl2_mc_files(
                zenith, azimuth, particle_types=[ParticleType.PROTON], merged=True
            )[ParticleType.PROTON.value]
            if len(proton_files) > 1:
                raise ValueError(
                    f"Multiple files found for proton, zenith {zenith} and azimuth {azimuth}, please merge them first with CTLearnTriModelManager.merge_DL2_files()"
                )
            proton_file = proton_files[0]

        
        irf_type, gammaness_efficiency, theta_efficiency = get_irf_type_from_config(config)

        match irf_type:
            case IRFType.EFFICIENCY_OPTIMIZED:
                cuts_type = CutType.EFFICIENCY_OPTIMIZED
                cuts = Cuts(
                    cuts_type,
                    efficiency_gammaness=gammaness_efficiency,
                    efficiency_theta=theta_efficiency,
                )
            case IRFType.SENSITIVITY_OPTIMIZED:
                cuts_type = CutType.SENSITIVITY_OPTIMIZED
                cuts = Cuts(
                    cuts_type,
                    gammaness_cut=None,
                    theta_cut=None,
                    efficiency_gammaness=None,
                    efficiency_theta=None,
                )
        

        output_directory = self.project_directories.get_irf_directory(zenith, azimuth, cuts)
        os.makedirs(output_directory, exist_ok=True)

        output_cuts_file = output_directory + f"/cuts_{zenith.value}_{azimuth.value}.fits"
        output_irf_file = output_directory + f"/irf_{zenith.value}_{azimuth.value}.fits"
        compatible_output_irf_file = output_directory + f"/gammapy_irf_{zenith.value}_{azimuth.value}.fits"
        output_benchmark_file = output_directory + f"/benchmark_{zenith.value}_{azimuth.value}.fits"

        cmd = f"scp {config} {output_directory}"
        result = os.system(cmd)
        assert result == 0, f"Failed to copy config file to output directory : {result}"

        cmd = f"mv {output_directory}/{Path(config).name} {output_directory}/config_{zenith.value}_{azimuth.value}.yaml"
        result = os.system(cmd)
        assert result == 0, f"Failed to rename config file to config_{zenith.value}_{azimuth.value}.yaml : {result}"

        config = f"{output_directory}/config_{zenith.value}_{azimuth.value}.yaml"

        electron_string = f"--electron-file {electron_file}" if electrons else ""
        proton_string = f"--proton-file {proton_file}" if protons else ""
        do_background_string = "--do-background" if protons else "--no-do-background"
        cmd = f"ctapipe-optimize-event-selection \
-c {config} \
--gamma-file {gamma_file} \
{proton_string} \
{electron_string} \
--output {output_cuts_file} \
--overwrite True \
--Tool.log_level DEBUG"
        print(cmd)
        result_cuts = os.system(cmd)
        if result_cuts != 0:
            raise RuntimeError(
                f"Error: Failed to produce cuts file for zenith {zenith} and azimuth {azimuth}"
            )
        cmd = f"ctapipe-compute-irf \
-c {config} --IrfTool.cuts_file {output_cuts_file} \
--gamma-file {gamma_file} \
{proton_string} \
{electron_string} \
{do_background_string} \
--output {output_irf_file} \
--benchmark-output {output_benchmark_file} \
--no-spatial-selection-applied --overwrite\
--Tool.log_level DEBUG"
        print(cmd)
        result_irfs = os.system(cmd)
        if result_irfs != 0:
            raise RuntimeError(
                f"Error: Failed to produce IRF file for zenith {zenith} and azimuth {azimuth}"
            )
        
        convert_irf_format(output_irf_file, output_cuts_file, compatible_output_irf_file)
        if not self.stereo:

            cmd = f"manager_create_irf_files \
-g {gamma_file} \
{proton_string} \
-o {compatible_output_irf_file} \
--energy-dependent-gh \
--energy-dependent-theta \
--gh-efficiency 0.7 \
--theta-containment 0.7 \
--overwrite "
            print(cmd)
            result_irfs = os.system(cmd)
            if result_irfs != 0:
                raise RuntimeError(
                    f"Error: Failed to produce IRF file for zenith {zenith} and azimuth {azimuth}"
            )


    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def produce_irfs_with_uncertainties(
        self,
        zenith: float,
        azimuth: float,
        config: str,
        pointlike=True,
        electrons=False,
        unblinded=True,
        resume_file_index=None,
        resume_cut_index=None,
        overwrite=False,
    ):
        """
        Produce Instrument Response Functions (IRFs) for given observational parameters.

        This method generates IRFs, cuts files, and benchmark files for a specified
        zenith and azimuth angle using the provided configuration. It supports
        processing gamma, electron, and proton particle types.

        Parameters
        ----------
        zenith : float
            Zenith angle of the observation in degrees.
        azimuth : float
            Azimuth angle of the observation in degrees.
        config : str
            Path to the configuration file for IRF generation.
        pointlike : bool, optional
            If True, use point-like gamma files. If False, use diffuse gamma files.
            Default is True.
        electrons : bool, optional
            If True, include electron files in the IRF generation. Default is False.
        overwrite : bool, optional
            If True, overwrite existing output files. Default is False.

        Raises
        ------
        ValueError
            If multiple files are found for a particle type or if required parameters
            are missing.
        RuntimeError
            If the system commands for generating cuts or IRFs fail.

        Notes
        -----
        - Ensure that the configuration file and input files are not moved or deleted
          as they are extensively used in the code for plotting and analysis.
        - Use `CTLearnTriModelManager.merge_DL2_files()` to merge multiple files
          before calling this method if necessary.
        """
        import os
        from pathlib import Path

        if pointlike:
            gamma_files = self.project_directories.get_dl2_mc_files(
                zenith, azimuth, particle_types=[ParticleType.GAMMA_POINT], merged=False
            )[ParticleType.GAMMA_POINT.value]
        else:
            gamma_files = self.project_directories.get_dl2_mc_files(
                zenith, azimuth, particle_types=[ParticleType.GAMMA_DIFFUSE], merged=False
            )[ParticleType.GAMMA_DIFFUSE.value]
        if len(gamma_files) == 1:
            raise ValueError(
                f"Only one file found for gamma, zenith {zenith} and azimuth {azimuth}, please use produce_irfs() instead."
            )
        if electrons:
            electrons_files = self.project_directories.get_dl2_mc_files(
                zenith, azimuth, particle_types=[ParticleType.ELECTRON], merged=True
            )[ParticleType.ELECTRON.value]
            if len(electrons_files) > 1:
                raise ValueError(
                    f"Multiple files found for electrons, zenith {zenith} and azimuth {azimuth}, please merge them first with CTLearnTriModelManager.merge_DL2_files()"
                )
            electron_file = electrons_files[0]

        proton_files = self.project_directories.get_dl2_mc_files(
            zenith, azimuth, particle_types=[ParticleType.PROTON], merged=True
        )[ParticleType.PROTON.value]
        if len(proton_files) > 1:
            raise ValueError(
                f"Multiple files found for proton, zenith {zenith} and azimuth {azimuth}, please merge them first with CTLearnTriModelManager.merge_DL2_files()"
            )
        proton_file = proton_files[0]
        
        irf_type, gammaness_efficiency, theta_efficiency = get_irf_type_from_config(config)

        # Check that irf_type is sensitivity optimized otherwise raise error
        if irf_type.value != "sensitivity_optimized":
            raise ValueError(
                "For unblinded IRFs with uncertainties, only sensitivity optimized IRFs are supported. "
                f"Current IRF type in config is {irf_type.value}. Please update the config file."
            )
        # Init a Cuts object for optimization the sensitivity
        cuts = Cuts(
            CutType.SENSITIVITY_OPTIMIZED,
            gammaness_cut=None,
            theta_cut=None,
            efficiency_gammaness=None,
            efficiency_theta=None,
        )

        irf_directory = self.project_directories.get_irf_directory(zenith, azimuth, cuts)
        os.makedirs(irf_directory, exist_ok=True)
        output_directory = f"{irf_directory}/IRFs_with_uncertainties/"
        os.makedirs(output_directory, exist_ok=True)

        cmd = f"scp {config} {output_directory}"
        result = os.system(cmd)
        assert result == 0, f"Failed to copy config file to output directory : {result}"

        cmd = f"mv {output_directory}/{Path(config).name} {output_directory}/config_{zenith.value}_{azimuth.value}.yaml"
        result = os.system(cmd)
        assert result == 0, f"Failed to rename config file to config_{zenith.value}_{azimuth.value}.yaml : {result}"

        config = f"{output_directory}/config_{zenith.value}_{azimuth.value}.yaml"

        electron_string = f"--electron-file {electron_file}" if electrons else ""
        if resume_file_index is None and resume_cut_index is None:
            for g, gamma_file in enumerate(gamma_files):
                output_cuts_file = output_directory + f"/cuts_{zenith.value}_{azimuth.value}_{g}.fits"
                cmd = f"ctapipe-optimize-event-selection \
-c {config} \
--gamma-file {gamma_file} \
--proton-file {proton_file} \
{electron_string} \
--output {output_cuts_file} \
--overwrite True"
                print(cmd)
                result_cuts = os.system(cmd)
                if result_cuts != 0:
                    raise RuntimeError(
                        f"Error: Failed to produce cuts file for zenith {zenith} and azimuth {azimuth}"
                )
        for g, gamma_file in enumerate(gamma_files):
            # Skip the previous files if resuming
            if resume_file_index is not None and g < resume_file_index:
                continue
            for i in range(len(gamma_files)):
                # Skip the previous cuts if resuming
                if resume_cut_index is not None and i < resume_cut_index:
                    continue
                # Skip the current gamma file if unblinded
                if i == g and unblinded:
                    continue
                cuts_file = output_directory + f"/cuts_{zenith.value}_{azimuth.value}_{i}.fits"
                output_irf_file = output_directory + f"/irf_{zenith.value}_{azimuth.value}_{g}_with_cuts_{i}.fits"
                compatible_output_irf_file = output_directory + f"/gammapy_irf_{zenith.value}_{azimuth.value}_{g}_with_cuts_{i}.fits"
                output_benchmark_file = output_directory + f"/benchmark_{zenith.value}_{azimuth.value}_{g}_with_cuts_{i}.fits"
                cmd = f"ctapipe-compute-irf \
-c {config} --IrfTool.cuts_file {cuts_file} \
--gamma-file {gamma_file} \
--proton-file {proton_file} \
{electron_string} \
--do-background \
--output {output_irf_file} \
--benchmark-output {output_benchmark_file} \
--no-spatial-selection-applied --overwrite"
                print(cmd)
                result_irfs = os.system(cmd)
                if result_irfs != 0:
                    raise RuntimeError(
                        f"Error: Failed to produce IRF file for zenith {zenith} and azimuth {azimuth}"
                    )
                convert_irf_format(output_irf_file, cuts_file, compatible_output_irf_file)



    def plot_sensitivity_benchmark(
        self,
        zenith: float,
        azimuth: float,
        cuts: list[Cuts] = [DefaultCuts.EFF_70.value],
        title: str = None,
        ax=None,
        label=None,
    ):
        import matplotlib.pyplot as plt
        from astropy.io import fits

        if ax is None:
            fig, ax = plt.subplots()
        if len(cuts) == 1:
            cuts[0].plot_cuts_info_plt(ax)
        for cut in cuts:
            irf_file = self.get_IRF_data(zenith, azimuth, cut)["benchmark_file"]
            with fits.open(irf_file) as hudl:
                energy_center = hudl["SENSITIVITY"].data["ENERG_LO"] + 0.5 * (
                    hudl["SENSITIVITY"].data["ENERG_HI"] - hudl["SENSITIVITY"].data["ENERG_LO"]
                )
                if len(cuts) > 1:
                    ax.plot(
                        energy_center[0],
                        hudl["SENSITIVITY"].data["ENERGY_FLUX_SENSITIVITY"][0, 0, :],
                        label=cut.get_label(),
                    )
                else:
                    ax.plot(
                        energy_center[0],
                        hudl["SENSITIVITY"].data["ENERGY_FLUX_SENSITIVITY"][0, 0, :],
                    )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Energy [TeV]")
        ax.set_ylabel("Sensitivity [erg s$^{-1}$ cm$^{-2}$]")
        if len(cuts) > 1:
            ax.legend()
        if title is not None:
            ax.set_title(title)
        if ax is None:
            plt.show()

    def plot_angular_resolution_benchmark(
        self,
        zenith: float,
        azimuth: float,
        cuts: list[Cuts] = [DefaultCuts.EFF_70.value],
        containments: list[int] = [68, 95],
        title: str = None,
        ax=None,
        label=None,
    ):
        import matplotlib.pyplot as plt
        from astropy.io import fits

        if ax is None:
            fig, ax = plt.subplots()
        if len(cuts) == 1:
            cuts[0].plot_cuts_info_plt(ax)
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        line_styles = ["-", "--", "-.", ":"]
        for cut, color in zip(cuts, default_colors[: len(cuts)]):
            irf_file = self.get_IRF_data(zenith, azimuth, cut)["benchmark_file"]
            with fits.open(irf_file) as hudl:
                energy_center = hudl["ANGULAR RESOLUTION "].data["ENERG_LO"] + 0.5 * (
                    hudl["ANGULAR RESOLUTION "].data["ENERG_HI"]
                    - hudl["ANGULAR RESOLUTION "].data["ENERG_LO"]
                )
                for containment, line_style in zip(containments, line_styles):
                    ax.plot(
                        energy_center[0],
                        hudl["ANGULAR RESOLUTION"].data[f"ANGULAR_RESOLUTION_{containment}"][0, 0, :],
                        color=color,
                        ls=line_style,
                    )
        ax.set_xscale("log")
        ax.set_xlabel("Energy [TeV]")
        ax.set_ylabel("Angular resolution [deg]")
        cut_labels = [cut.get_label() for cut in cuts]
        containment_labels = [f"{containment}%" for containment in containments]
        cut_legend = ax.legend(
            handles=[plt.Line2D([0], [0], color=color, lw=2) for color in default_colors[: len(cuts)]],
            labels=cut_labels,
            loc="best",
        )
        containment_legend = ax.legend(
            handles=[plt.Line2D([0], [0], color="black", ls=ls, lw=2) for ls in line_styles[: len(containments)]],
            labels=containment_labels,
            loc="lower left",
            title="Containment",
        )
        if len(cuts) > 1:
            ax.add_artist(cut_legend)
        if title is not None:
            ax.set_title(title)
        if ax is None:
            plt.show()

    def plot_energy_resolution_benchmark(
        self,
        zenith: float,
        azimuth: float,
        cuts: list[Cuts] = [DefaultCuts.EFF_70.value],
        title: str = None,
        ax=None,
        label=None,
    ):
        import matplotlib.pyplot as plt
        from astropy.io import fits

        if ax is None:
            fig, ax = plt.subplots()
        if len(cuts) == 1:
            cuts[0].plot_cuts_info_plt(ax)
        for cut in cuts:
            irf_file = self.get_IRF_data(zenith, azimuth, cut)["benchmark_file"]
            with fits.open(irf_file) as hudl:
                energy_center = hudl["ENERGY BIAS RESOLUTION"].data["ENERG_LO"] + 0.5 * (
                    hudl["ENERGY BIAS RESOLUTION"].data["ENERG_HI"]
                    - hudl["ENERGY BIAS RESOLUTION"].data["ENERG_LO"]
                )
                if len(cuts) > 1:
                    ax.plot(
                        energy_center[0],
                        hudl["ENERGY BIAS RESOLUTION"].data["RESOLUTION"][0, 0, :],
                        label=cut.get_label(),
                    )
                else:
                    ax.plot(
                        energy_center[0],
                        hudl["ENERGY BIAS RESOLUTION"].data["RESOLUTION"][0, 0, :],
                    )
        ax.set_xscale("log")
        ax.set_xlabel("Energy [TeV]")
        ax.set_ylabel("Energy resolution")
        if len(cuts) > 1:
            ax.legend()
        if title is not None:
            ax.set_title(title)
        if ax is None:
            plt.show()

    def plot_energy_bias_benchmark(
        self,
        zenith: float,
        azimuth: float,
        cuts: list[Cuts] = [DefaultCuts.EFF_70.value],
        title: str = None,
        ax=None,
        label=None,
    ):
        import matplotlib.pyplot as plt
        from astropy.io import fits

        if ax is None:
            fig, ax = plt.subplots()
        if len(cuts) == 1:
            cuts[0].plot_cuts_info_plt(ax)
        for cut in cuts:
            irf_file = self.get_IRF_data(zenith, azimuth, cut)["benchmark_file"]
            with fits.open(irf_file) as hudl:
                energy_center = hudl["ENERGY BIAS RESOLUTION"].data["ENERG_LO"] + 0.5 * (
                    hudl["ENERGY BIAS RESOLUTION"].data["ENERG_HI"]
                    - hudl["ENERGY BIAS RESOLUTION"].data["ENERG_LO"]
                )
                if len(cuts) > 1:
                    ax.plot(
                        energy_center[0],
                        hudl["ENERGY BIAS RESOLUTION"].data["BIAS"][0, 0, :],
                        label=cut.get_label(),
                    )
                else:
                    ax.plot(
                        energy_center[0],
                        hudl["ENERGY BIAS RESOLUTION"].data["BIAS"][0, 0, :],
                    )
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Energy [TeV]")
        ax.set_ylabel("Energy bias")
        if len(cuts) > 1:
            ax.legend()
        if title is not None:
            ax.set_title(title)
        if ax is None:
            plt.show()

    # @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    # def plot_benchmark(
    #     self,
    #     zenith: float,
    #     azimuth: float,
    #     cuts: list[Cuts] = [DefaultCuts.EFF_70.value],
    #     containments: list[int] = [68, 95],
    #     title: str = None,
    #     axs=None,
    #     label=None,
    # ):
    #     """
    #     Plot benchmark performance metrics for a given zenith and azimuth.

    #     This function generates multiple plots to visualize sensitivity, angular resolution,
    #     energy resolution, and energy bias for the specified zenith and azimuth angles.
    #     The plots are generated based on the provided cuts and containment percentages.

    #     Parameters
    #     ----------
    #     zenith : float
    #         Zenith angle in degrees.
    #     azimuth : float
    #         Azimuth angle in degrees.
    #     cuts : list[Cuts], optional
    #         List of cut configurations to apply. Defaults to [DefaultCuts.EFF_70.value].
    #     containments : list[int], optional
    #         List of containment percentages to plot angular resolution. Defaults to [68, 95].
    #     title : str, optional
    #         Title for the plots. Defaults to None.
    #     axs : list[matplotlib.axes.Axes], optional
    #         List of axes to plot on. If None, new figures and axes are created. Defaults to None.
    #     label : str, optional
    #         Label for the plots. Defaults to None.

    #     Notes
    #     -----
    #     - The function uses Matplotlib for plotting and Astropy for reading FITS files.
    #     - If multiple cuts are provided, legends are created for both cuts and containment percentages.
    #     - The function generates four separate plots:
    #         1. Sensitivity vs. Energy
    #         2. Angular resolution vs. Energy
    #         3. Energy resolution vs. Energy
    #         4. Energy bias vs. Energy

    #     Raises
    #     ------
    #     KeyError
    #         If the required data keys are not found in the FITS files.
    #     """
    #     import matplotlib.pyplot as plt
    #     from astropy.io import fits

    #     if axs is None:
    #         fig, ax = plt.subplots()
    #     else:
    #         ax = axs[0]
    #     if len(cuts) == 1:
    #         cuts[0].plot_cuts_info_plt(ax)
    #     for cut in cuts:
    #         irf_file = self.get_IRF_data(zenith, azimuth, cut)["benchmark_file"]
    #         print(irf_file)
    #         hudl = fits.open(irf_file)
    #         energy_center = hudl["SENSITIVITY"].data["ENERG_LO"] + 0.5 * (
    #             hudl["SENSITIVITY"].data["ENERG_HI"]
    #             - hudl["SENSITIVITY"].data["ENERG_LO"]
    #         )
    #         if len(cuts) > 1:
    #             plt.plot(
    #                 energy_center[0],
    #                 hudl["SENSITIVITY"].data["ENERGY_FLUX_SENSITIVITY"][0, 0, :],
    #                 label=cut.get_label(),
    #             )
    #         else:
    #             plt.plot(
    #                 energy_center[0],
    #                 hudl["SENSITIVITY"].data["ENERGY_FLUX_SENSITIVITY"][0, 0, :],
    #             )
    #     plt.xscale("log")
    #     plt.yscale("log")
    #     plt.xlabel("Energy [TeV]")
    #     plt.ylabel("Sensitivity [erg s$^{-1}$ cm$^{-2}$]")
    #     if len(cuts) > 1:
    #         plt.legend()
    #     if title is not None:
    #         plt.title(title)
    #     if axs is None:
    #         plt.show()

    #     fig, ax = plt.subplots()
    #     if len(cuts) == 1:
    #         cuts[0].plot_cuts_info_plt(ax)
    #     default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    #     for cut, color in zip(cuts, default_colors[: len(cuts)]):
    #         irf_file = self.get_IRF_data(zenith, azimuth, cut)["benchmark_file"]
    #         hudl = fits.open(irf_file)
    #         energy_center = hudl["ANGULAR RESOLUTION "].data["ENERG_LO"] + 0.5 * (
    #             hudl["ANGULAR RESOLUTION "].data["ENERG_HI"]
    #             - hudl["ANGULAR RESOLUTION "].data["ENERG_LO"]
    #         )
    #         line_styles = ["-", "--", "-.", ":"]
    #         for containment, line_style in zip(containments, line_styles):
    #             plt.plot(
    #                 energy_center[0],
    #                 hudl["ANGULAR RESOLUTION"].data[
    #                     f"ANGULAR_RESOLUTION_{containment}"
    #                 ][0, 0, :],
    #                 color=color,
    #                 ls=line_style,
    #             )

    #     # plt.plot(energy_center[0], hudl['ANGULAR RESOLUTION'].data['ANGULAR_RESOLUTION_25'][0,0,:], label='25%')
    #     # plt.plot(energy_center[0], hudl['ANGULAR RESOLUTION'].data['ANGULAR_RESOLUTION_50'][0,0,:], label='50%')
    #     # plt.plot(energy_center[0], hudl['ANGULAR RESOLUTION'].data['ANGULAR_RESOLUTION_68'][0,0,:], label='68%')
    #     # plt.plot(energy_center[0], hudl['ANGULAR RESOLUTION'].data['ANGULAR_RESOLUTION_95'][0,0,:], label='95%')
    #     plt.xscale("log")
    #     plt.xlabel("Energy [TeV]")
    #     plt.ylabel("Angular resolution [deg]")
    #     # Create separate legends for cuts and containment percentages
    #     # Create separate legends for cuts and containment percentages
    #     cut_labels = [cut.get_label() for cut in cuts]
    #     containment_labels = [f"{containment}%" for containment in containments]
    #     cut_legend = ax.legend(
    #         handles=[
    #             plt.Line2D([0], [0], color=color, lw=2)
    #             for color in default_colors[: len(cuts)]
    #         ],
    #         labels=cut_labels,
    #         loc="best",
    #     )
    #     containment_legend = ax.legend(
    #         handles=[
    #             plt.Line2D([0], [0], color="black", ls=ls, lw=2)
    #             for ls in line_styles[: len(containments)]
    #         ],
    #         labels=containment_labels,
    #         loc="lower left",
    #         title="Containment",
    #     )
    #     if len(cuts) > 1:
    #         ax.add_artist(cut_legend)

    #     if title is not None:
    #         plt.title(title)
    #     if axs is None:
    #         plt.show()

    #     fig, ax = plt.subplots()
    #     if len(cuts) == 1:
    #         cuts[0].plot_cuts_info_plt(ax)
    #     for cut in cuts:
    #         irf_file = self.get_IRF_data(zenith, azimuth, cut)["benchmark_file"]
    #         hudl = fits.open(irf_file)
    #         energy_center = hudl["ENERGY BIAS RESOLUTION"].data["ENERG_LO"] + 0.5 * (
    #             hudl["ENERGY BIAS RESOLUTION"].data["ENERG_HI"]
    #             - hudl["ENERGY BIAS RESOLUTION"].data["ENERG_LO"]
    #         )
    #         if len(cuts) > 1:
    #             plt.plot(
    #                 energy_center[0],
    #                 hudl["ENERGY BIAS RESOLUTION"].data["RESOLUTION"][0, 0, :],
    #                 label=cut.get_label(),
    #             )
    #         else:
    #             plt.plot(
    #                 energy_center[0],
    #                 hudl["ENERGY BIAS RESOLUTION"].data["RESOLUTION"][0, 0, :],
    #             )
    #     plt.xscale("log")
    #     plt.xlabel("Energy [TeV]")
    #     plt.ylabel("Energy resolution")
    #     if len(cuts) > 1:
    #         plt.legend()
    #     if title is not None:
    #         plt.title(title)
    #     if axs is None:
    #         plt.show()

    #     fig, ax = plt.subplots()
    #     if len(cuts) == 1:
    #         cuts[0].plot_cuts_info_plt(ax)
    #     for cut in cuts:
    #         irf_file = self.get_IRF_data(zenith, azimuth, cut)["benchmark_file"]
    #         hudl = fits.open(irf_file)
    #         energy_center = hudl["ENERGY BIAS RESOLUTION"].data["ENERG_LO"] + 0.5 * (
    #             hudl["ENERGY BIAS RESOLUTION"].data["ENERG_HI"]
    #             - hudl["ENERGY BIAS RESOLUTION"].data["ENERG_LO"]
    #         )
    #         if len(cuts) > 1:
    #             plt.plot(
    #                 energy_center[0],
    #                 hudl["ENERGY BIAS RESOLUTION"].data["BIAS"][0, 0, :],
    #                 label=cut.get_label(),
    #             )
    #         else:
    #             plt.plot(
    #                 energy_center[0],
    #                 hudl["ENERGY BIAS RESOLUTION"].data["BIAS"][0, 0, :],
    #             )
    #     plt.xscale("log")
    #     plt.xlabel("Energy [TeV]")
    #     plt.ylabel("Energy bias")
    #     if len(cuts) > 1:
    #         plt.legend()
    #     if title is not None:
    #         plt.title(title)
    #     if axs is None:
    #         plt.show()
    #     hudl.close()

    def plot_cuts(
        self,
        zeniths: list[float] = None,
        azimuths: list[float] = None,
        cuts: list[Cuts] = [DefaultCuts.EFF_70.value],
        axs=None,
        label=None,
        export_to_h5: str=None,
        import_from_h5: str = None,
        import_label: str = None,
    ):
        """
        Plot the cuts for given zenith and azimuth directions.

        Parameters
        ----------
        zeniths : list of float, optional
            List of zenith angles in degrees. If None, use all available Monte Carlo directions.
        azimuths : list of float, optional
            List of azimuth angles in degrees. Must match the length of `zeniths`.
        cuts : list of Cuts, optional
            List of cut configurations to plot. Defaults to a single cut with 70% efficiency.
        axs : matplotlib.axes.Axes, optional
            Axes to plot on. If None, create new subplots.
        label : str, optional
            Label for the plot legend. If None, use the cut's label if multiple cuts are provided.

        Notes
        -----
        This method retrieves the IRF data for the specified directions and plots the
        gammaness and theta cuts as a function of energy. If multiple cuts are provided,
        they are plotted on the same axes with a legend.
        """
        import matplotlib.pyplot as plt
        from astropy.io import fits

        
        export_curves = ExportCurves(export_to_h5)
        if import_from_h5 is not None:
            import_curves = ExportCurves(import_from_h5, export_mode=False, import_label=import_label)
            for curve_type in import_curves.curve_types:
                if curve_type not in [CurveType.GH_CUTS.value, CurveType.THETA_CUTS.value]:
                    raise ValueError(f"Imported curves are not of type GH-cuts or theta-cuts : {curve_type}")
            # assert ((import_curves.curve_types == CurveType.GH_CUTS) | (import_curves.curve_types == CurveType.THETA_CUTS)), "Imported curves are not of type GH_CUTS or THETA_CUTS"

        if zeniths is None:
            coords = self.get_available_MC_directions(verbose=False)
        else:
            coords = list(zip(zeniths, azimuths))

        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            if (len(cuts) == 1) and (import_from_h5 is None):
                cuts[0].plot_cuts_info_plt(axs[0])
                cuts[0].plot_cuts_info_plt(axs[1])

        for i, coord in enumerate(coords):
            zenith, azimuth = coord
            for cut in cuts:
                cuts_file = self.get_IRF_data(zenith, azimuth, cut)["cuts_file"]
                if label is None:
                    if (len(cuts) > 1) or (import_from_h5 is not None):
                        l = cut.get_label()
                    else:
                        l = None
                else:
                    l = label
                with fits.open(cuts_file) as hdul:
                    
                    export_curves.add_curve(hdul["GH_CUTS"].data["center"], hdul["GH_CUTS"].data["cut"], CurveType.GH_CUTS, cuts=cut)
                    export_curves.add_curve(hdul["RAD_MAX"].data["center"], hdul["RAD_MAX"].data["cut"], CurveType.THETA_CUTS, cuts=cut)
                    axs[0].plot(
                        hdul["GH_CUTS"].data["center"],
                        hdul["GH_CUTS"].data["cut"],
                        label=l,
                    )
                    axs[0].set_xlabel("Energy [TeV]")
                    axs[0].set_ylabel("Gammaness cut")
                    axs[0].set_xscale("log")

                    
                    axs[1].plot(
                        hdul["RAD_MAX"].data["center"],
                        hdul["RAD_MAX"].data["cut"],
                        label=l,
                    )
                    axs[1].set_xlabel("Energy [TeV]")
                    axs[1].set_ylabel("Theta cut [deg]")
                    axs[1].set_xscale("log")

            
        if len(cuts) > 1:
            axs[0].legend()
            axs[1].legend()
        if export_to_h5 is not None:
            export_curves.export()
        if import_from_h5 is not None:
            import_curves.plot_curves(axs = [axs[0], axs[1]] * int(len(import_curves.x_values)/2))
            axs[0].legend()
            axs[1].legend()
        plt.tight_layout()
        if axs is None:
            plt.show()

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def plot_irfs(self, zenith, azimuth, cuts: Cuts = DefaultCuts.EFF_70.value):
        """
        Plot Instrument Response Functions (IRFs) for a given zenith and azimuth angle.

        Parameters
        ----------
        zenith : float
            Zenith angle in degrees for which the IRFs are to be plotted.
        azimuth : float
            Azimuth angle in degrees for which the IRFs are to be plotted.
        cuts : Cuts, optional
            Selection cuts to apply when retrieving the IRF data. Defaults to `DefaultCuts.EFF_70.value`.

        Notes
        -----
        This method reads the IRF data from the direction model using the specified
        zenith and azimuth angles and plots the effective area, background, and energy
        dispersion using the `peek` method of the respective IRF classes from `gammapy.irf`.
        """
        from gammapy.irf import (
            Background2D,
            EffectiveAreaTable2D,
            EnergyDispersion2D,
        )

        irf_file = self.get_IRF_data(zenith, azimuth, cuts)["irf_file"]
        # rad_max = RadMax2D.read(irf_file, hdu="RAD MAX")
        aeff = EffectiveAreaTable2D.read(irf_file, hdu="EFFECTIVE AREA")
        bkg = Background2D.read(irf_file, hdu="BACKGROUND")
        edisp = EnergyDispersion2D.read(irf_file, hdu="ENERGY DISPERSION")
        edisp.peek()
        aeff.peek()
        bkg.peek()

    def plot_loss(self):
        """
        Plot the training and validation loss for multiple models.

        This method visualizes the training and validation loss curves for
        three models: direction, energy, and type models. It reads the
        training logs from CSV files, aggregates the loss values, and
        plots them against the epochs.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Displays the loss plots for the models.

        Notes
        -----
        - The method searches for training log files in directories
          specified by the model's parameters.
        - If no training logs are found in the first search pattern,
          an alternative pattern is used.
        - The method uses Matplotlib for plotting and Pandas for
          reading the CSV files.
        """
        import glob

        import matplotlib.pyplot as plt
        import pandas as pd

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, model in zip(
            axs, [self.direction_model, self.energy_model, self.type_model]
        ):
            # print(f"{model.model_parameters_table['model_dir'][0]}/{model.model_nickname}*/training_log.csv")
            training_logs = np.sort(
                glob.glob(
                    f"{model.model_parameters_table['model_dir'][0]}/{model.model_nickname}*/training_log.csv"
                )
            )
            if len(training_logs) == 0:
                # print(f"{model.model_parameters_table['model_dir'][0]}/{model.model_nickname}/{model.model_nickname}*/training_log.csv")
                training_logs = np.sort(
                    glob.glob(
                        f"{model.model_parameters_table['model_dir'][0]}/{model.model_nickname}/{model.model_nickname}*/training_log.csv"
                    )
                )
            # print(training_logs)
            losses_train = []
            losses_val = []
            for training_log in training_logs:
                df = pd.read_csv(training_log)
                losses_train = np.concatenate((losses_train, df["loss"].to_numpy()))
                losses_val = np.concatenate((losses_val, df["val_loss"].to_numpy()))
            epochs = np.arange(1, len(losses_train) + 1)
            if len(epochs) > 1:
                ax.plot(epochs, losses_train, label="Training", lw=2)
                ax.plot(epochs, losses_val, label="Validation", ls="--")
            else:
                ax.scatter(epochs, losses_train, label="Training", lw=2)
                ax.scatter(epochs, losses_val, label="Validation", ls="--")
            ax.set_title(f"{model.model_parameters_table['reco'][0]} training".title())
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_xticks(np.arange(1, len(epochs) + 1, 2))
            ax.legend()
        plt.tight_layout()
        plt.show()

    @u.quantity_input(zeniths=u.deg, azimuths=u.deg)
    def plot_angular_resolution_DL2(
        self,
        zeniths: list[float] = None,
        azimuths: list[float] = None,
        cuts: list[Cuts] = [DefaultCuts.NO_CUTS.value],
        ylim=None,
        particle_type: ParticleType = ParticleType.GAMMA_POINT,
        figsize=None,
        ax=None,
        label=None,
        export_to_h5: str = None,
        import_from_h5: str = None,
        import_label: str = None,
    ):
        """
        Plot the angular resolution as a function of true energy for DL2 data.

        Parameters
        ----------
        zeniths : list[float], optional
            List of zenith angles in degrees. If None, use available Monte Carlo directions.
        azimuths : list[float], optional
            List of azimuth angles in degrees. Must have the same length as `zeniths`.
        cuts : list[Cuts], optional
            List of cuts to apply. Defaults to `[DefaultCuts.NO_CUTS.value]`.
        ylim : tuple, optional
            Tuple specifying the y-axis limits as (min, max). If None, use default limits.
        particle_type : ParticleType, optional
            Type of particle to analyze. Defaults to `ParticleType.GAMMA_POINT`.
        figsize : tuple, optional
            Figure size as (width, height). If None, use default size.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If None, create a new figure and axes.
        label : str, optional
            Label for the plot legend. If None, generate a default label.

        Raises
        ------
        AssertionError
            If `zeniths` and `azimuths` are provided but have different lengths.
            If both `zeniths/azimuths` and `cuts` have lengths greater than 1.

        Notes
        -----
        - The function calculates the angular resolution for the given zenith and azimuth
          angles or cuts and plots it as a function of true energy.
        - If multiple zenith/azimuth pairs are provided, the closest pair to the training
          data is highlighted in the plot.
        - The function supports plotting on an existing Axes object or creating a new plot.
        """
        import astropy.units as u
        import matplotlib.pyplot as plt

        
        export_curves = ExportCurves(export_to_h5)
        if import_from_h5 is not None:
            import_curves = ExportCurves(import_from_h5, export_mode=False, import_label=import_label)
            for curve_type in import_curves.curve_types:
                if curve_type not in [CurveType.ANGULAR_RESOLUTION.value]:
                    raise ValueError(f"Imported curves are not of type angular-resolution : {curve_type}")

        if zeniths is None:
            coords = self.get_available_MC_directions(verbose=False)
        else:
            assert len(zeniths) == len(azimuths), (
                "zeniths and azimuths must have the same length"
            )
            coords = list(zip(zeniths, azimuths))

        assert len(coords) == 1 or len(cuts) == 1, (
            "Either zeniths/azimuths or 'cuts' must have a length of 1"
        )

        avg_model_az = np.mean(self.direction_model.validity.azimuth_range).to(u.deg)
        avg_model_ze = np.mean(self.direction_model.validity.zenith_range).to(u.deg)
        testing_azs = np.empty(len(coords)) * u.deg
        testing_zes = np.empty(len(coords)) * u.deg
        i = 0
        for zenith, azimuth in coords:
            testing_azs[i] = azimuth.to(u.deg)
            testing_zes[i] = zenith.to(u.deg)
            i += 1
        closest_coord_index = np.argmin(
            angular_distance(avg_model_ze, avg_model_az, testing_zes, testing_azs)
        )

        # DL2_gamma_table = read_table_hdf5(self.direction_model.model_index_file, path=f'{self.direction_model.model_nickname}/DL2/MC/{particle_type.value}')

        if ax is None:
            if figsize is not None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig, ax = plt.subplots()

            if len(cuts) == 1 and (import_from_h5 is None):
                stored_efficiency_theta = cuts[0].efficiency_theta
                cuts[0].efficiency_theta = None
                cuts[0].plot_cuts_info_plt(ax)
                cuts[0].efficiency_theta = stored_efficiency_theta
            if len(coords) == 1:
                plot_pointing_on_ax(ax, coords[0][0], coords[0][1])

        for i, coord in enumerate(coords):
            for cut in cuts:
                zenith, azimuth = coord
                try:
                    e_bins, ang_res_err = self.get_angular_resolution_DL2(
                        zenith, azimuth, cut, particle_type
                    )
                except:
                    continue
                e = (e_bins[:-1].value + e_bins[1:].value) / 2
                ang_res = [e_r[0].value for e_r in ang_res_err]
                ang_res_minus = [e_r[0].value - e_r[1].value for e_r in ang_res_err]
                ang_res_plus = [e_r[2].value - e_r[0].value for e_r in ang_res_err]
                ang_res_min = [e_r[1].value for e_r in ang_res_err]
                ang_res_max = [e_r[2].value for e_r in ang_res_err]
                if len(cuts) == 1 and (import_from_h5 is None):
                    if i == closest_coord_index:
                        if label is None:
                            l = (
                                f"Closest to training data\n{particle_type.value} ({zenith.value:.1f}, {azimuth.value:.1f})°"
                                if len(coords) > 1
                                else f"{particle_type.value} ({zenith.value:.1f}, {azimuth.value:.1f})°"
                            )
                        else:
                            l = label
                        ax.errorbar(
                            e,
                            ang_res,
                            yerr=[ang_res_minus, ang_res_plus],
                            label=l,
                            markersize=8,
                            marker="o",
                            ls="--",
                        )
                        ax.fill_between(e, ang_res_min, ang_res_max, alpha=0.2)
                        # ctaplot.plot_angular_resolution_per_energy(true_alt, reco_alt, true_az, reco_az, true_energy, bins=log_bins, label=l, markersize=8, ax=ax)
                    else:
                        if label is None:
                            l = f"{particle_type.value} ({zenith.value:.1f}, {azimuth.value:.1f})°"
                        else:
                            l = label
                        ax.errorbar(
                            e,
                            ang_res,
                            yerr=[ang_res_minus, ang_res_plus],
                            label=l,
                            markersize=8,
                            marker="o",
                            ls="--",
                        )
                        ax.fill_between(e, ang_res_min, ang_res_max, alpha=0.2)
                        # ctaplot.plot_angular_resolution_per_energy(true_alt, reco_alt, true_az, reco_az, true_energy, bins=log_bins, label=l, alpha=0.5, marker='v', ax=ax)
                else:
                    if label is None:
                        stored_efficiency_theta = cut.efficiency_theta
                        cut.efficiency_theta = None
                        l = cut.get_label()
                        cut.efficiency_theta = stored_efficiency_theta
                    else:
                        l = label
                    ax.errorbar(
                        e,
                        ang_res,
                        yerr=[ang_res_minus, ang_res_plus],
                        label=l,
                        markersize=8,
                        marker="o",
                        ls="--",
                    )
                    ax.fill_between(e, ang_res_min, ang_res_max, alpha=0.2)
                    # ctaplot.plot_angular_resolution_per_energy(true_alt, reco_alt, true_az, reco_az, true_energy, bins=log_bins, label=l, markersize=8, ax=ax)
                # except IndexError as e:
                #     print(e)
                #     print("Skipping this zenith/azimuth pair")
                export_curves.add_curve(
                    e,
                    ang_res,
                    CurveType.ANGULAR_RESOLUTION,
                    cuts=cut,
                )
        if export_to_h5 is not None:
            export_curves.export()
        if import_from_h5 is not None:
            import_curves.plot_curves(axs = [ax] * int(len(import_curves.x_values)))
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.set_xscale("log")
        ax.set_xlabel("True Energy [TeV]")
        ax.set_ylabel("Angular resolution [deg]")
        ax.legend()
        ax.grid(False, which="both")
        if ax is None:
            plt.show()

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def get_DL2_tables(
        self,
        zenith: float,
        azimuth: float,
        cuts: Cuts = DefaultCuts.NO_CUTS.value,
        particle_type: ParticleType = ParticleType.GAMMA_POINT,
        apply_theta_cut=True,
    ):
        """
        Retrieve DL2 tables for a given zenith, azimuth, and particle type, applying specified cuts.

        Parameters
        ----------
        zenith : float
            Zenith angle of the simulated events in degrees.
        azimuth : float
            Azimuth angle of the simulated events in degrees.
        cuts : Cuts, optional
            Cuts to apply to the data. Defaults to `DefaultCuts.NO_CUTS.value`.
        particle_type : ParticleType, optional
            Type of particle to filter (e.g., gamma, proton). Defaults to `ParticleType.GAMMA_POINT`.
        apply_theta_cut : bool, optional
            Whether to apply a theta cut during energy-dependent cuts. Defaults to True.

        Returns
        -------
        tuple
            A tuple containing:
            - true_energy (astropy.units.Quantity): True energies of the events.
            - reco_energy (astropy.units.Quantity): Reconstructed energies of the events.
            - true_alt (astropy.units.Quantity): True altitudes of the events.
            - reco_alt (astropy.units.Quantity): Reconstructed altitudes of the events.
            - true_az (astropy.units.Quantity): True azimuths of the events.
            - reco_az (astropy.units.Quantity): Reconstructed azimuths of the events.
            - log_bins (astropy.units.Quantity): Logarithmic energy bins.

        Raises
        ------
        ValueError
            If an unknown cut type is provided.
        """
        import astropy.units as u
        from astropy.io.misc.hdf5 import read_table_hdf5
        from astropy.table import join, vstack

        testing_DL2_gamma_files = self.project_directories.get_dl2_mc_files(
            zenith,
            azimuth,
            particle_types=[particle_type],
        )[particle_type.value]
        if len(testing_DL2_gamma_files) == 0:
            return
        dl2_gamma = []
        shower_parameters_gamma = []
        tel_id = None if self.stereo else self.telescope_ids[0]
        for file in testing_DL2_gamma_files:
            dl2_gamma.append(load_DL2_data_MC(file, tel_id=tel_id))
            shower_parameters_gamma.append(load_true_shower_parameters(file))
        dl2_gamma = vstack(dl2_gamma)
        shower_parameters_gamma = vstack(shower_parameters_gamma)
        dl2_gamma = join(
            dl2_gamma, shower_parameters_gamma, keys=["obs_id", "event_id"]
        )

        match cuts.cut_type:
            case CutType.GLOBAL:
                mask = dl2_gamma[self.gammaness_key] > cuts.gammaness_cut
                reco_alt = dl2_gamma[self.reco_alt_key].to(u.deg)[mask]
                reco_az = dl2_gamma[self.reco_az_key].to(u.deg)[mask]
                true_alt = dl2_gamma[self.true_alt_key].to(u.deg)[mask]
                true_az = dl2_gamma[self.true_az_key].to(u.deg)[mask]
                reco_energy = dl2_gamma[self.reco_energy_key][mask]
                true_energy = dl2_gamma[self.true_energy_key][mask]
                true_energy_min = np.min(true_energy)
                true_energy_max = np.max(true_energy)
                reco_energy_min = np.min(reco_energy)
                reco_energy_max = np.max(reco_energy)
                bins_per_decade = 5
                log_bins = (
                    np.logspace(
                        np.log10(true_energy_min),
                        np.log10(true_energy_max),
                        num=int(
                            np.log10(true_energy_max / true_energy_min)
                            * bins_per_decade
                        )
                        + 1,
                    )
                    * u.TeV
                )

            case CutType.EFFICIENCY_OPTIMIZED | CutType.SENSITIVITY_OPTIMIZED:
                cuts_file = self.get_IRF_data(zenith, azimuth, cuts)["cuts_file"]
                dl2_gamma, log_bins = self.apply_energy_dependent_cuts_MC(
                    dl2_gamma, cuts_file, theta_cut=apply_theta_cut
                )
                reco_alt = dl2_gamma[self.reco_alt_key].to(u.deg)
                reco_az = dl2_gamma[self.reco_az_key].to(u.deg)
                true_alt = dl2_gamma[self.true_alt_key].to(u.deg)
                true_az = dl2_gamma[self.true_az_key].to(u.deg)
                reco_energy = dl2_gamma[self.reco_energy_key]
                true_energy = dl2_gamma[self.true_energy_key]
            case _:
                raise ValueError(f"Unknown cut type: {cuts.cut_type}")
        return true_energy, reco_energy, true_alt, reco_alt, true_az, reco_az, log_bins

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def get_angular_resolution_DL2(
        self,
        zenith: float = None,
        azimuth: float = None,
        cuts: Cuts = DefaultCuts.NO_CUTS.value,
        particle_type: ParticleType = ParticleType.GAMMA_POINT,
    ):
        """
        Compute the angular resolution as a function of energy for DL2 data.

        Parameters
        ----------
        zenith : float, optional
            Zenith angle of the observation in degrees. If None, use all available data.
        azimuth : float, optional
            Azimuth angle of the observation in degrees. If None, use all available data.
        cuts : Cuts, optional
            Selection cuts to apply to the data. Defaults to `DefaultCuts.NO_CUTS.value`.
        particle_type : ParticleType, optional
            Type of particle to consider. Defaults to `ParticleType.GAMMA_POINT`.

        Returns
        -------
        tuple
            A tuple containing:
            - e : numpy.ndarray
                Energy bins used for the angular resolution calculation.
            - ang_res : numpy.ndarray
                Angular resolution values corresponding to the energy bins.

        Notes
        -----
        This method uses the `ctaplot` library to calculate the angular resolution
        based on true and reconstructed event parameters. The `get_DL2_tables` method
        is used to retrieve the necessary data.
        """
        import ctaplot

        true_energy, reco_energy, true_alt, reco_alt, true_az, reco_az, log_bins = (
            self.get_DL2_tables(
                zenith, azimuth, cuts, particle_type, apply_theta_cut=False
            )
        )
        e, ang_res = ctaplot.angular_resolution_per_energy(
            true_alt, reco_alt, true_az, reco_az, true_energy, bins=log_bins
        )
        return e, ang_res

    @u.quantity_input(zeniths=u.deg, azimuths=u.deg)
    def plot_energy_resolution_DL2(
        self,
        zeniths: list[float] = None,
        azimuths: list[float] = None,
        cuts: list[Cuts] = [DefaultCuts.NO_CUTS.value],
        ylim=None,
        particle_type: ParticleType = ParticleType.GAMMA_POINT,
        figsize=None,
        ax=None,
        label=None,
        export_to_h5: str = None,
        import_from_h5: str = None,
        import_label: str = None,
    ):
        """
        Plot the energy resolution as a function of true energy for DL2 data.

        Parameters
        ----------
        zeniths : list of float, optional
            List of zenith angles in degrees. If None, use all available MC directions.
        azimuths : list of float, optional
            List of azimuth angles in degrees. Must have the same length as `zeniths`.
        cuts : list of Cuts, optional
            List of cuts to apply. Default is `[DefaultCuts.NO_CUTS.value]`.
        ylim : tuple of float, optional
            Limits for the y-axis as (ymin, ymax). If None, use default limits.
        particle_type : ParticleType, optional
            Type of particle to consider. Default is `ParticleType.GAMMA_POINT`.
        figsize : tuple of float, optional
            Size of the figure as (width, height). If None, use default size.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, create a new figure and axes.
        label : str, optional
            Label for the plot legend. If None, generate labels automatically.

        Raises
        ------
        AssertionError
            If `zeniths` and `azimuths` have different lengths.
            If both `zeniths/azimuths` and `cuts` have lengths greater than 1.

        Notes
        -----
        - If multiple zenith/azimuth pairs are provided, the closest pair to the
          training data is highlighted.
        - If multiple cuts are provided, each cut is plotted separately.

        See Also
        --------
        get_energy_resolution_DL2 : Compute energy resolution for given parameters.
        """
        import matplotlib.pyplot as plt

        
        export_curves = ExportCurves(export_to_h5)
        if import_from_h5 is not None:
            import_curves = ExportCurves(import_from_h5, export_mode=False, import_label=import_label)
            for curve_type in import_curves.curve_types:
                if curve_type not in [CurveType.ENERGY_RESOLUTION.value]:
                    raise ValueError(f"Imported curves are not of type energy-resolution : {curve_type}")

        if zeniths is None:
            coords = self.get_available_MC_directions(verbose=False)
        else:
            assert len(zeniths) == len(azimuths), (
                "zeniths and azimuths must have the same length"
            )
            coords = list(zip(zeniths, azimuths))
        assert len(coords) == 1 or len(cuts) == 1, (
            "Either zeniths/azimuths or 'cuts' must have a length of 1"
        )
        avg_model_az = np.mean(self.direction_model.validity.azimuth_range).to(u.deg)
        avg_model_ze = np.mean(self.direction_model.validity.zenith_range).to(u.deg)
        testing_azs = np.empty(len(coords)) * u.deg
        testing_zes = np.empty(len(coords)) * u.deg
        i = 0
        for zenith, azimuth in coords:
            testing_azs[i] = azimuth.to(u.deg)
            testing_zes[i] = zenith.to(u.deg)
            i += 1
        closest_coord_index = np.argmin(
            angular_distance(avg_model_ze, avg_model_az, testing_zes, testing_azs)
        )
        if ax is None:
            if figsize is not None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig, ax = plt.subplots()

            if len(cuts) == 1 and (import_from_h5 is None):
                cuts[0].plot_cuts_info_plt(ax)
            if len(coords) == 1:
                plot_pointing_on_ax(ax, coords[0][0], coords[0][1])

        for i, coord in enumerate(coords):
            for cut in cuts:
                try:
                    e_bins, e_res_err = self.get_energy_resolution_DL2(
                        zenith=coord[0],
                        azimuth=coord[1],
                        cuts=cut,
                        particle_type=particle_type,
                    )
                except:
                    continue
                e = (e_bins[:-1].value + e_bins[1:].value) / 2
                e_res = [e_r[0] for e_r in e_res_err]
                e_res_minus = [e_r[0] - e_r[1] for e_r in e_res_err]
                e_res_plus = [e_r[2] - e_r[0] for e_r in e_res_err]
                e_res_min = [e_r[1] for e_r in e_res_err]
                e_res_max = [e_r[2] for e_r in e_res_err]

                if len(cuts) == 1 and (import_from_h5 is None):
                    if i == closest_coord_index:
                        if label is None:
                            l = (
                                f"Closest to training data\n{particle_type.value} ({zenith.value:.1f}, {azimuth.value:.1f})°"
                                if len(coords) > 1
                                else f"{particle_type.value} ({zenith.value:.1f}, {azimuth.value:.1f})°"
                            )
                        else:
                            l = label
                        ax.errorbar(
                            e,
                            e_res,
                            yerr=[e_res_minus, e_res_plus],
                            label=l,
                            markersize=8,
                            marker="o",
                            ls="--",
                        )
                        ax.fill_between(e, e_res_min, e_res_max, alpha=0.2)
                    else:
                        if label is None:
                            l = f"{particle_type.value} ({zenith.value:.1f}, {azimuth.value:.1f})°"
                        else:
                            l = label
                        ax.errorbar(
                            e,
                            e_res,
                            yerr=[e_res_minus, e_res_plus],
                            label=l,
                            alpha=0.5,
                            marker="v",
                            ls="--",
                        )
                        ax.fill_between(e, e_res_min, e_res_max, alpha=0.2)
                else:
                    if label is None:
                        l = cut.get_label()
                    else:
                        l = label
                    ax.errorbar(
                        e,
                        e_res,
                        yerr=[e_res_minus, e_res_plus],
                        label=l,
                        markersize=8,
                        marker="o",
                        ls="--",
                    )
                    ax.fill_between(e, e_res_min, e_res_max, alpha=0.2)
                export_curves.add_curve(
                    e,
                    e_res,
                    CurveType.ENERGY_RESOLUTION,
                    cuts=cut,
                )
        if export_to_h5 is not None:
            export_curves.export()
        if import_from_h5 is not None:
            import_curves.plot_curves(axs = [ax] * int(len(import_curves.x_values)))
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.set_xscale("log")
        ax.set_xlabel("True Energy [TeV]")
        ax.set_ylabel("Energy resolution")
        ax.legend()
        ax.grid(False, which="both")
        if ax is None:
            plt.show()

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def get_energy_resolution_DL2(
        self,
        zenith: float = None,
        azimuth: float = None,
        cuts: Cuts = DefaultCuts.NO_CUTS.value,
        particle_type: ParticleType = ParticleType.GAMMA_POINT,
    ):
        """
        Compute the energy resolution for DL2 data.

        Parameters
        ----------
        zenith : float, optional
            The zenith angle of the observations. If None, use all available data.
        azimuth : float, optional
            The azimuth angle of the observations. If None, use all available data.
        cuts : Cuts, optional
            The selection cuts to apply to the data. Defaults to `DefaultCuts.NO_CUTS.value`.
        particle_type : ParticleType, optional
            The type of particle to consider. Defaults to `ParticleType.GAMMA_POINT`.

        Returns
        -------
        e : numpy.ndarray
            The energy bin centers.
        e_res : numpy.ndarray
            The energy resolution for each energy bin.

        Notes
        -----
        This method uses `ctaplot.energy_resolution_per_energy` to compute the energy
        resolution based on true and reconstructed energies.
        """
        import ctaplot

        true_energy, reco_energy, true_alt, reco_alt, true_az, reco_az, log_bins = (
            self.get_DL2_tables(zenith, azimuth, cuts, particle_type)
        )
        e, e_res = ctaplot.energy_resolution_per_energy(
            true_energy, reco_energy, bins=log_bins
        )
        return e, e_res

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def plot_ROC_curve_DL2(self, zenith: float, azimuth: float, nbins: int = 3):
        """
        Plot the ROC curve for DL2 data based on gammaness and true energy.

        Parameters
        ----------
        zenith : float
            Zenith angle of the simulated data in degrees.
        azimuth : float
            Azimuth angle of the simulated data in degrees.
        nbins : int, optional
            Number of energy bins for the ROC curve, by default 3.

        Notes
        -----
        This function uses DL2 data for gamma and proton events to compute the
        Receiver Operating Characteristic (ROC) curve. It requires the `ctaplot`
        library for plotting and assumes the presence of specific keys for
        gammaness and true energy in the DL2 data.
        The function retrieves the DL2 data for gamma and proton events, combines
        them, and calculates the ROC curve for different energy bins. The resulting
        plot is displayed using matplotlib.

        Raises
        ------
        KeyError
            If the required keys (`gammaness_key` or `true_energy_key`) are not
            present in the DL2 data.
        ValueError
            If no DL2 files are found for the given zenith and azimuth angles.
        """
        import astropy.units as u
        import ctaplot
        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.table import join, vstack

        # if export_to_h5 is not None:
        #     export_curves = ExportCurves(export_to_h5)

        testing_DL2_gamma_files = self.project_directories.get_dl2_mc_files(
            zenith, azimuth
        )[ParticleType.GAMMA_POINT.value]
        testing_DL2_proton_files = self.project_directories.get_dl2_mc_files(
            zenith, azimuth
        )[ParticleType.PROTON.value]

        tel_id = None if self.stereo else self.telescope_ids[0]

        if len(testing_DL2_gamma_files) > 0:
            dl2_gamma = []
            shower_parameters_gamma = []
            for file in testing_DL2_gamma_files:
                # print(file)
                dl2_gamma.append(load_DL2_data_MC(file, tel_id=tel_id))
                shower_parameters_gamma.append(load_true_shower_parameters(file))
            dl2_gamma = vstack(dl2_gamma)
            shower_parameters_gamma = vstack(shower_parameters_gamma)
            dl2_gamma = join(
                dl2_gamma, shower_parameters_gamma, keys=["obs_id", "event_id"]
            )
        else:
            dl2_gamma = []
        mc_type_gamma = np.zeros(len(dl2_gamma))

        if len(testing_DL2_proton_files) > 0:
            dl2_protons = []
            shower_parameters_protons = []
            for file in testing_DL2_proton_files:
                # print(file)
                dl2_protons.append(load_DL2_data_MC(file, tel_id=tel_id))
                shower_parameters_protons.append(load_true_shower_parameters(file))
            dl2_proton = vstack(dl2_protons)
            shower_parameters_protons = vstack(shower_parameters_protons)
            dl2_proton = join(
                dl2_proton, shower_parameters_protons, keys=["obs_id", "event_id"]
            )
        else:
            dl2_proton = []
        mc_type_proton = np.ones(len(dl2_proton))

        mc_type = np.concatenate((mc_type_gamma, mc_type_proton))
        gammaness = np.concatenate(
            (dl2_gamma[self.gammaness_key], dl2_proton[self.gammaness_key])
        )
        mc_gamma_energies = (
            np.concatenate(
                (dl2_gamma[self.true_energy_key], dl2_proton[self.true_energy_key])
            )
            * u.TeV
        )
        # plt.figure(figsize=(14,8))
        energy_bins = (
            np.logspace(
                np.log10(min(mc_gamma_energies.value)),
                np.log10(max(mc_gamma_energies.value)),
                nbins + 1,
            )
            * u.TeV
        )

        ctaplot.plot_roc_curve_gammaness_per_energy(
            mc_type,
            gammaness,
            mc_gamma_energies,
            energy_bins=energy_bins,  # u.Quantity([0.01,0.1,1,3,10], u.TeV),
            linestyle="--",
            alpha=1,
            linewidth=2,
        )
        plt.legend()
        # plt.xlim(-0.05, 1.05)
        # plt.ylim(-0.05, 1.05)
        plt.show()
        
    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def get_IRF_data(self, zenith=None, azimuth=None, cuts: Cuts = None):
        """
        Retrieve Instrument Response Function (IRF) data based on specified parameters.

        Parameters
        ----------
        zenith : float, optional
            The zenith angle for which to retrieve IRF data. If None, the average zenith
            from the validity range is used.
        azimuth : float, optional
            The azimuth angle for which to retrieve IRF data. If None, the average azimuth
            from the validity range is used.
        cuts : Cuts
            The cuts object specifying the IRF type and efficiency parameters.

        Returns
        -------
        tuple
            A tuple containing:
            - config (str): The configuration string for the matched IRF data.
            - cuts_file (str): The file path to the cuts file.
            - irf_file (str): The file path to the IRF file.
            - benchmark_file (str): The file path to the benchmark file.

        Raises
        ------
        IndexError
            If no IRF data is found for the specified cuts or direction, or if multiple
            matches are found for the given parameters.

        Notes
        -----
        If both `zenith` and `azimuth` are None, the method calculates the average zenith
        and azimuth from the validity range and retrieves the closest IRF data.
        """
        from astropy.io.misc.hdf5 import read_table_hdf5

        if zenith is None or azimuth is None:
            average_zenith = (
                self.direction_model.validity.zenith_range[0] + self.validity.zenith_range[1]
            ) / 2
            average_azimuth = (
                self.direction_model.validity.azimuth_range[0] + self.validity.azimuth_range[1]
            ) / 2
            # return self.get_closest_IRF_data(average_zenith, average_azimuth, cuts)
            return self.project_directories.get_closest_irf_files(
            zenith, azimuth, cuts
        )

        irf_files = self.project_directories.get_irf_files(zenith, azimuth, cuts)
        return irf_files
    
    def compare_irfs_to_RF(self, zenith: float, azimuth=None):
        """
        Compare Instrument Response Functions (IRFs) to Random Forest (RF) benchmarks.

        This method visualizes and compares the performance metrics of IRFs generated
        by CTLearn with RF benchmark data for a given zenith angle and optional azimuth angle.
        It generates plots for flux sensitivity, angular resolution, and energy resolution.

        Parameters
        ----------
        zenith : float
            The zenith angle in degrees for which the comparison is performed.
        azimuth : float, optional
            The azimuth angle in degrees for which the comparison is performed.
            If not provided, defaults to None.

        Raises
        ------
        ImportError
            If required modules cannot be imported.
        FileNotFoundError
            If the required IRF or RF benchmark files are not found.
        KeyError
            If expected data keys are missing in the IRF or RF benchmark files.

        Notes
        -----
        - The method uses HDF5 and FITS files to load performance data.
        - Plots are displayed using Matplotlib for visual comparison.
        - The RF benchmark data is dynamically imported based on the provided zenith angle
          and telescope configuration.

        See Also
        --------
        self.direction_model.get_IRF_data : Method to retrieve IRF data for a given zenith
            and azimuth angle.
        """
        import importlib
        import importlib.resources as pkg_resources

        import matplotlib.pyplot as plt
        from astropy.io import fits
        from astropy.table import Table

        tel_path = "SST1M"
        tel_string = "stereo" if self.stereo else "tel_001"
        stereo_path = "stereo" if self.stereo else "mono"

        module_name = f"ctlearn_manager.resources.irfs.{tel_path}.performance.{stereo_path}_performance_med4_{zenith}deg"
        RF_bechmpark = importlib.import_module(module_name)

        with pkg_resources.path(
            RF_bechmpark, f"angular_resolution_{tel_string}.h5"
        ) as angular_resolution_file:
            angular_resolution_table = Table.read(
                angular_resolution_file, format="hdf5", path="res"
            )
            angular_resolution_table_bins = Table.read(
                angular_resolution_file, format="hdf5", path="bins"
            )

        with pkg_resources.path(
            RF_bechmpark, f"energy_resolution_{tel_string}.h5"
        ) as energy_resolution_file:
            energy_resolution_table = Table.read(
                energy_resolution_file, format="hdf5", path="res"
            )
            energy_resolution_table_bins = Table.read(
                energy_resolution_file, format="hdf5", path="bins"
            )

        with pkg_resources.path(
            RF_bechmpark, f"flux_sensitivity_{tel_string}.h5"
        ) as flux_sensitivity_file:
            flux_sensitivity_table = Table.read(
                flux_sensitivity_file, format="hdf5", path="sensitivity"
            )

        irf_file = self.get_IRF_data(zenith, azimuth)["benchmark_file"]
        hudl = fits.open(irf_file)

        energy_center = hudl["SENSITIVITY"].data["ENERG_LO"] + 0.5 * (
            hudl["SENSITIVITY"].data["ENERG_HI"] - hudl["SENSITIVITY"].data["ENERG_LO"]
        )
        plt.plot(
            flux_sensitivity_table["energy"],
            flux_sensitivity_table["flux_sensitivity"],
            label="RF",
        )
        plt.fill_between(
            flux_sensitivity_table["energy"],
            flux_sensitivity_table["flux_sensitivity"]
            - flux_sensitivity_table["flux_sensitivity_err_minus"],
            flux_sensitivity_table["flux_sensitivity"]
            + flux_sensitivity_table["flux_sensitivity_err_plus"],
            alpha=0.5,
            color= get_color("ctlearn_1"),
        )
        plt.plot(
            energy_center[0],
            hudl["SENSITIVITY"].data["ENERGY_FLUX_SENSITIVITY"][0, 0, :],
            label="CTLearn",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Energy [TeV]")
        plt.ylabel("Sensitivity [erg s$^{-1}$ cm$^{-2}$]")
        plt.legend()
        plt.show()

        energy_center = hudl["ANGULAR RESOLUTION "].data["ENERG_LO"] + 0.5 * (
            hudl["ANGULAR RESOLUTION "].data["ENERG_HI"]
            - hudl["ANGULAR RESOLUTION "].data["ENERG_LO"]
        )
        energy_center_RF = angular_resolution_table_bins["energy_bins"][
            1:
        ] - 0.5 * np.diff(angular_resolution_table_bins["energy_bins"])
        plt.plot(
            energy_center_RF, angular_resolution_table["angular_res"], label="RF 68%"
        )
        plt.fill_between(
            energy_center_RF,
            angular_resolution_table["angular_res_err_lo"],
            angular_resolution_table["angular_res_err_hi"],
            alpha=0.5,
            color=get_color("ctlearn_1"),
        )
        plt.plot(
            energy_center[0],
            hudl["ANGULAR RESOLUTION"].data["ANGULAR_RESOLUTION_68"][0, 0, :],
            label="CTLearn 68%",
        )
        plt.xscale("log")
        plt.xlabel("Energy [TeV]")
        plt.ylabel("Angular resolution [deg]")
        plt.legend()
        plt.show()
        plt.show()

        energy_center = hudl["ENERGY BIAS RESOLUTION"].data["ENERG_LO"] + 0.5 * (
            hudl["ENERGY BIAS RESOLUTION"].data["ENERG_HI"]
            - hudl["ENERGY BIAS RESOLUTION"].data["ENERG_LO"]
        )
        energy_center_RF = energy_resolution_table_bins["energy_bins"][
            1:
        ] - 0.5 * np.diff(energy_resolution_table_bins["energy_bins"])
        plt.plot(energy_center_RF, energy_resolution_table["energy_res"], label="RF")
        plt.fill_between(
            energy_center_RF,
            energy_resolution_table["energy_res_err_lo"],
            energy_resolution_table["energy_res_err_hi"],
            alpha=0.5,
            color=get_color("ctlearn_1"),
        )
        plt.plot(
            energy_center[0],
            hudl["ENERGY BIAS RESOLUTION"].data["RESOLUTION"][0, 0, :],
            label="CTLearn",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Energy [TeV]")
        plt.ylabel("Energy resolution")
        plt.legend()
        plt.show()

        hudl.close()

    def plot_everything_dl2(
        self,
        output_directory: str,
        dl2_files: list[str],
        gammaness_cut: float = 0.9,
        edep_cuts: bool = False,
    ):
        """
        Generate DL2 plots using the trained tri-model and submit the job to the cluster.

        Parameters
        ----------
        output_directory : str
            The directory where the output files and plots will be saved.
        dl2_files : list[str]
            A list of paths to the DL2 data files to be processed.
        gammaness_cut : float, optional
            The gammaness cut value to filter events, by default 0.9.
        edep_cuts : bool, optional
            Whether to apply energy-dependent cuts, by default False.

        Notes
        -----
        This method serializes the current instance of the tri-model to a pickle file
        and generates a command to create DL2 plots. It temporarily disables the cluster
        configuration to process any unprocessed DL2 files in the same job. The plotting
        job is submitted to the cluster using an SBATCH script.
        """
        import os
        import pickle

        tri_model_file = f"{output_directory}/tri_model.pkl"
        self.dl2_data_files = dl2_files

        use_cluster = self.cluster_configuration.use_cluster
        self.cluster_configuration.use_cluster = False  # if some DL2 files were not processed, they will be processed in the same job as the plotting job, and not submit multiple new jobs

        with open(tri_model_file, "wb") as f:
            pickle.dump(self, f)
        self.cluster_configuration.use_cluster = use_cluster

        cmd = f"plot_dl2 --stereo_tri_model {tri_model_file} --output_directory {output_directory} --gammaness_cut {gammaness_cut} --edep_cuts {edep_cuts}"

        sbatch_file = self.cluster_configuration.write_sbatch_script(
            "dl2_plots", cmd, output_directory, use_gpu_cscs=False
        )
        os.system(f"sbatch {sbatch_file}")

    def plot_zenith_azimuth_ranges(self):
        """
        Plot the zenith and azimuth ranges for the direction model.

        This method delegates the plotting of zenith and azimuth ranges
        to the `plot_zenith_azimuth_ranges` method of the `direction_model`
        attribute.

        Notes
        -----
        Ensure that the `direction_model` attribute is properly initialized
        and has a `plot_zenith_azimuth_ranges` method before calling this
        method to avoid runtime errors.
        """
        self.direction_model.plot_zenith_azimuth_ranges()

    def apply_energy_dependent_cuts_MC(self, data, cuts_file, theta_cut=True):
        """
        Apply energy-dependent cuts to Monte Carlo (MC) data.

        Parameters
        ----------
        data : pandas.DataFrame or astropy.table.Table
            The input data containing reconstructed and true event information.
        cuts_file : str
            Path to the FITS file containing the cut values for gammaness and theta.
        theta_cut : bool, optional
            Whether to apply the theta cut based on angular separation, by default True.

        Returns
        -------
        dl2 : pandas.DataFrame or astropy.table.Table
            The data after applying the energy-dependent cuts.
        E_bins : astropy.units.Quantity
            The energy bins used for the cuts, in units of TeV.

        Notes
        -----
        - The function reads cut values for gammaness and theta from the provided FITS file.
        - It ensures that the energy ranges for gammaness and theta cuts match.
        - If `theta_cut` is enabled, the angular separation between true and reconstructed
          coordinates is calculated and used for filtering.
        - The function creates masks for each energy bin and combines them to filter the data.
        """
        # Apply cuts to the data
        from astropy.coordinates import SkyCoord
        from astropy.io import fits

        with fits.open(cuts_file) as hdul:
            gammaness_cuts = hdul["GH_CUTS"].data["cut"]
            energy_low_gamma = hdul["GH_CUTS"].data["low"]
            energy_high_gamma = hdul["GH_CUTS"].data["high"]
            theta_cuts = hdul["RAD_MAX"].data["cut"]
            energy_low_theta = hdul["RAD_MAX"].data["low"]
            energy_high_theta = hdul["RAD_MAX"].data["high"]
            E_bins = np.concatenate((energy_low_gamma, [energy_high_gamma[-1]])) * u.TeV
            assert (energy_low_gamma == energy_low_theta).all(), (
                "Energy low values for gammaness and theta cuts do not match"
            )
            assert (energy_high_gamma == energy_high_theta).all(), (
                "Energy high values for gammaness and theta cuts do not match"
            )

            if theta_cut:
                true_coords = SkyCoord(
                    alt=data[self.true_alt_key],
                    az=data[self.true_az_key],
                    frame="altaz",
                    unit="deg",
                )
                reco_coords = SkyCoord(
                    alt=data[self.reco_alt_key],
                    az=data[self.reco_az_key],
                    frame="altaz",
                    unit="deg",
                )
                angular_separation = true_coords.separation(reco_coords).deg
                data["angular_separation"] = angular_separation

            masks = []
            for E_min, E_max, gcut, tcut in zip(
                energy_low_gamma, energy_high_gamma, gammaness_cuts, theta_cuts
            ):
                energy_mask = (data[self.reco_energy_key] > E_min) & (
                    data[self.reco_energy_key] < E_max
                )
                gammaness_mask = data[self.gammaness_key] > gcut
                if theta_cut:
                    theta_mask = data["angular_separation"] < tcut
                    mask = energy_mask & gammaness_mask & theta_mask
                else:
                    mask = energy_mask & gammaness_mask
                masks.append(mask)

            full_mask = np.zeros(len(data), dtype=bool)
            for mask in masks:
                full_mask |= mask
            dl2 = data[full_mask]
        return dl2, E_bins
