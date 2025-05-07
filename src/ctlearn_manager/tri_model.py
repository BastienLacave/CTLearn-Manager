import ast
from pathlib import Path

import astropy.units as u
import ctadata
import numpy as np

from .io.io import load_DL2_data_MC, load_true_shower_parameters
from .model_manager import CTLearnModelManager, DataSample
from .utils.utils import (
    ClusterConfiguration,
    ParticleType,
    angular_distance,
    set_mpl_style,
    get_irf_type_from_config,
    Cuts,
    IRFType,
    CutType,
    DefaultCuts,
    CTLearnManagerStyle,
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

    def __init__(self, direction_model: CTLearnModelManager, energy_model: CTLearnModelManager, type_model: CTLearnModelManager, cluster_configuration=ClusterConfiguration()):
        """
        Initialize the CTLearnTriModelManager with direction, energy, and type models.
        :param direction_model: A CTLearnModelManager instance for direction reconstruction.
        :type direction_model: CTLearnModelManager
        :param energy_model: A CTLearnModelManager instance for energy reconstruction.
        :type energy_model: CTLearnModelManager
        :param type_model: A CTLearnModelManager instance for type reconstruction.
        :type type_model: CTLearnModelManager
        :param cluster_configuration: Configuration for the cluster, defaults to ClusterConfiguration().
        :type cluster_configuration: ClusterConfiguration, optional
        :raises ValueError: If the direction_model is not a direction model.
        :raises ValueError: If the energy_model is not an energy model.
        :raises ValueError: If the type_model is not a type model.
        :raises ValueError: If all models do not have the same channels.
        :raises ValueError: If all models do not have the same stereo value.
        :raises ValueError: If all models do not have the same telescope_ids.
        :return: None
        """
        if direction_model.model_parameters_table['reco'][0] in ['direction', 'cameradirection', 'skydirection']:
            self.direction_model = direction_model
        else:
            raise ValueError('direction_model must be a direction model')
        if energy_model.model_parameters_table['reco'][0] == 'energy':
            self.energy_model = energy_model
        else:
            raise ValueError('energy_model must be an energy model')
        if type_model.model_parameters_table['reco'][0] == 'type':
            self.type_model = type_model
        else:
            raise ValueError('type_model must be a type model')
        import ast
        direction_channels = ast.literal_eval(self.direction_model.model_parameters_table['channels'][0])
        energy_channels = ast.literal_eval(self.energy_model.model_parameters_table['channels'][0])
        type_channels = ast.literal_eval(self.type_model.model_parameters_table['channels'][0])
        if not (direction_channels == energy_channels == type_channels):
            raise ValueError('All models must have the same channels')
        else:
            self.channels = direction_channels
        
        if not (self.direction_model.stereo == self.energy_model.stereo == self.type_model.stereo):
            raise ValueError('All models must have the same stereo value')
        else:
            self.stereo = self.direction_model.stereo
        if not (self.direction_model.telescope_ids == self.energy_model.telescope_ids == self.type_model.telescope_ids):
            raise ValueError('All models must have the same telescope_ids')
        
        if not (self.direction_model.min_telescopes == self.energy_model.min_telescopes == self.type_model.min_telescopes):
            raise ValueError('All models must have the same min_telescopes')
        else:
            self.min_telescopes = self.direction_model.min_telescopes
        self.telescope_ids = self.direction_model.telescope_ids
        self.telescope_names = self.direction_model.telescope_names
        self.cluster_configuration = cluster_configuration
        self.reconstruction_method = "CTLearn"
        self.reco_field_suffix = self.reconstruction_method if self.stereo else f"{self.reconstruction_method}_tel"
        self.set_keys()
        print(f"🧠🧠🧠 CTLearnTriModelManager ▮ {self.direction_model.model_nickname} ▮ {self.energy_model.model_nickname} ▮ {self.type_model.model_nickname} ▮")
        self.get_available_MC_directions()
        set_mpl_style()

    def set_keys(self):
        """
        Set the keys for various attributes used in the model.
        This method initializes several attributes with specific keys based on the 
        `reco_field_suffix` and `stereo` properties of the instance.

        Attributes
        ----------
        gammaness_key : str
            Key for the gammaness prediction.
        reco_energy_key : str
            Key for the reconstructed energy.
        intensity_key : str
            Key for the hillas intensity.
        reco_alt_key : str
            Key for the reconstructed altitude.
        reco_az_key : str
            Key for the reconstructed azimuth.
        true_alt_key : str
            Key for the true altitude.
        true_az_key : str
            Key for the true azimuth.
        true_energy_key : str
            Key for the true energy.
        pointing_alt_key : str
            Key for the pointing altitude, varies based on `stereo`.
        pointing_az_key : str
            Key for the pointing azimuth, varies based on `stereo`.
        time_key : str
            Key for the time.
        """
        self.gammaness_key = f"{self.reco_field_suffix}_prediction" #if self.CTLearn else "gammaness"
        self.reco_energy_key = f"{self.reco_field_suffix}_energy" #if self.CTLearn else "reco_energy"
        self.intensity_key = "hillas_intensity" #if self.CTLearn else "intensity"
        self.reco_alt_key = f"{self.reco_field_suffix}_alt" #if self.CTLearn else "reco_alt"
        self.reco_az_key = f"{self.reco_field_suffix}_az" #if self.CTLearn else "reco_az"
        self.true_alt_key = "true_alt" #if self.CTLearn else "alt"
        self.true_az_key = "true_az" #if self.CTLearn else "az"
        self.true_energy_key = "true_energy" #if self.CTLearn else "energy"
        # self.true_type_key = "true_type" #if self.CTLearn else "type"
        self.pointing_alt_key = "array_altitude" if self.stereo else "altitude" #if self.CTLearn else "alt_tel"
        self.pointing_az_key = "array_azimuth" if self.stereo else "azimuth" #if self.CTLearn else "az_tel"
        self.time_key = "time" #if self.CTLearn else "dragon_time"
            
    def set_testing_data(self, testing_samples: list[DataSample]):
        """
        Set the directories and associated parameters for testing data.
        This method updates the testing data for the direction, energy, and type models
        with the provided gamma and proton directories and their corresponding parameters.
        :param testing_gamma_dirs: List of directories containing gamma testing data.
        :type testing_gamma_dirs: list
        :param testing_proton_dirs: List of directories containing proton testing data.
        :type testing_proton_dirs: list
        :param testing_gamma_zenith_distances: List of zenith distances for gamma testing data.
        :type testing_gamma_zenith_distances: list
        :param testing_gamma_azimuths: List of azimuths for gamma testing data.
        :type testing_gamma_azimuths: list
        :param testing_proton_zenith_distances: List of zenith distances for proton testing data.
        :type testing_proton_zenith_distances: list
        :param testing_proton_azimuths: List of azimuths for proton testing data.
        :type testing_proton_azimuths: list
        :param testing_gamma_patterns: List of patterns for gamma testing data.
        :type testing_gamma_patterns: list
        :param testing_proton_patterns: List of patterns for proton testing data.
        :type testing_proton_patterns: list
        :raises ValueError: If the lengths of the gamma lists are not equal.
        :raises ValueError: If the lengths of the proton lists are not equal.
        """
        for model in [self.direction_model, self.energy_model, self.type_model]:
            for data_sample in testing_samples:  
                model.update_model_manager_testing_data(data_sample)
        self.get_available_testing_directions()
    
    def set_DL2_MC_file(self, testing_MC_DL2_file: str, testing_MC_DL2_data_sample: DataSample):
        """
        Set the DL2 Monte Carlo (MC) files for testing.
        This method updates the DL2 MC files for the direction, energy, and type models.
        :param testing_DL2_gamma_files: List of file paths for testing DL2 gamma files.
        :type testing_DL2_gamma_files: list
        :param testing_DL2_proton_files: List of file paths for testing DL2 proton files.
        :type testing_DL2_proton_files: list
        :param testing_DL2_gamma_zenith_distances: List of zenith distances for testing DL2 gamma files.
        :type testing_DL2_gamma_zenith_distances: list
        :param testing_DL2_gamma_azimuths: List of azimuths for testing DL2 gamma files.
        :type testing_DL2_gamma_azimuths: list
        :param testing_DL2_proton_zenith_distances: List of zenith distances for testing DL2 proton files.
        :type testing_DL2_proton_zenith_distances: list
        :param testing_DL2_proton_azimuths: List of azimuths for testing DL2 proton files.
        :type testing_DL2_proton_azimuths: list
        """
        for model in [self.direction_model, self.energy_model, self.type_model]:
            model.update_model_manager_DL2_MC_file(
                testing_MC_DL2_file=testing_MC_DL2_file,
                testing_MC_DL2_data_sample=testing_MC_DL2_data_sample
            )

    def delete_table_from_index(self, path: str):
        """
        Erase the table from the index file.
        This method removes the specified table from the HDF5 index file of the direction model.
        :param path: Path to the table to be erased.
        :type path: str
        """
        import h5py
        with h5py.File(self.direction_model.model_index_file, 'r+') as f:
            del f[path]
            print(f"Table {path} erased from {self.direction_model.model_index_file}")

    def get_available_testing_directions(self):
        """
        Retrieve and print available testing directions from the direction model's HDF5 file.
        This method reads the testing directions (zenith and azimuth angles) from the specified
        HDF5 file associated with the direction model. It prints each pair of zenith and azimuth
        angles in the format "(ZD, Az): (zenith, azimuth)".
        :raises KeyError: If the required keys are not found in the HDF5 file.
        :raises IOError: If there is an issue reading the HDF5 file.
        """
        from astropy.io.misc.hdf5 import read_table_hdf5
        zeniths = []
        azimuths = []

        for particle_type in ParticleType:
            try:
                DL2_table = read_table_hdf5(self.direction_model.model_index_file, path=f'{self.direction_model.model_nickname}/testing/{particle_type.value}')
                _zeniths = DL2_table[f'testing_{particle_type.value}_zenith_distances']
                _azimuths = DL2_table[f'testing_{particle_type.value}_azimuths']
            except:
                _zeniths = []
                _azimuths = []
            zeniths.append(_zeniths)
            azimuths.append(_azimuths)

        flat_zeniths = [item for sublist in zeniths for item in sublist]
        flat_azimuths = [item for sublist in azimuths for item in sublist]

        coords = set(zip(flat_zeniths, flat_azimuths))
        if len(coords) > 0:
            print("Available testing directions:")
        for zenith, azimuth in coords:
            available_particles = []
            for i, particle_type in enumerate(ParticleType):
                particle_available = (zenith, azimuth) in set(zip(zeniths[i], azimuths[i]))
                if particle_available:
                    available_particles.append(particle_type.value)
            if len(available_particles) > 0:
                print(f"(ZD, Az): ({zenith.value}, {azimuth.value})°\t{' | '.join(available_particles)}")
            else:
                print(f"(ZD, Az): ({zenith.value}, {azimuth.value})°")

    def get_available_MC_directions(self, verbose=True):
        """
        Retrieve and print available Monte Carlo (MC) directions from HDF5 files.
        This method reads the zenith and azimuth distances for gamma and proton 
        events from the specified HDF5 file and prints the available directions 
        for both types of events.
        The method attempts to read the following datasets from the HDF5 file:
        - `testing_DL2_gamma_zenith_distances` and `testing_DL2_gamma_azimuths` 
          for gamma events.
        - `testing_DL2_proton_zenith_distances` and `testing_DL2_proton_azimuths` 
          for proton events.
        If the datasets are not found, empty lists are used instead.
        The available directions are printed in the format:
        (ZD, Az): (zenith_distance, azimuth)    gamma | proton
        Where `gamma` and `proton` indicate the availability of the respective 
        event type for the given direction.

        Raises
        ------
            Any exceptions raised during the reading of the HDF5 file are caught 
            and result in empty lists for the respective event type.
        """
        from astropy.io.misc.hdf5 import read_table_hdf5

        zeniths = []
        azimuths = []

        for particle_type in ParticleType:
            try:
                DL2_table = read_table_hdf5(self.direction_model.model_index_file, path=f'{self.direction_model.model_nickname}/DL2/MC/{particle_type.value}')
                _zeniths = DL2_table[f'testing_DL2_{particle_type.value}_zenith_distances']
                _azimuths = DL2_table[f'testing_DL2_{particle_type.value}_azimuths']
            except:
                _zeniths = []
                _azimuths = []
            zeniths.append(_zeniths)
            azimuths.append(_azimuths)

        flat_zeniths = [item for sublist in zeniths for item in sublist]
        flat_azimuths = [item for sublist in azimuths for item in sublist]

        coords = set(zip(flat_zeniths, flat_azimuths))
        if verbose:
            if len(coords) > 0:
                print("Available MC DL2 directions:")
            for zenith, azimuth in coords:
                available_particles = []
                for i, particle_type in enumerate(ParticleType):
                    particle_available = (zenith, azimuth) in set(zip(zeniths[i], azimuths[i]))
                    if particle_available:
                        available_particles.append(particle_type.value)
                if len(available_particles) > 0:
                    print(f"(ZD, Az): ({zenith}, {azimuth}) \t {' | '.join(available_particles)}")
                else:
                    print(f"(ZD, Az): ({zenith}, {azimuth})")
        return coords
        
    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def launch_testing(self, zenith: float, azimuth: float, output_dirs: list[str], config_dir: str | None = None, launch_particle_types:list[ParticleType]=[ParticleType.GAMMA_POINT], batch_size=64, dl2_subarray=True, overwrite=False, config=None):
        """
        Launch testing for the given zenith and azimuth angles.
        This function checks the testing files for gamma and proton particles, ensures they match across models,
        and launches the testing process using the specified models.
        :param zenith: Zenith angle for the testing.
        :type zenith: float
        :param azimuth: Azimuth angle for the testing.
        :type azimuth: float
        :param output_dirs: List of directories to store the output files. If length is 1, both gamma and proton outputs
                            will be stored in the same directory. If length is 2, the first directory will be used for
                            gamma outputs and the second for proton outputs.
        :type output_dirs: list
        :param config_dir: Directory for configuration files, defaults to None.
        :type config_dir: str, optional
        :param launch_particle_type: Type of particles to launch testing for. Must be 'gamma', 'proton', or 'both'.
                                        Defaults to 'both'.
        :type launch_particle_type: str
        :raises ValueError: If `launch_particle_type` is not 'gamma', 'proton', or 'both'.
        :raises ValueError: If the testing directories for gamma or proton particles do not match across models.
        :raises ValueError: If no matching directory is found for the given zenith and azimuth angles.
        :raises ValueError: If `output_dirs` does not have length 1 or 2.
        """
        assert len(output_dirs) == len(launch_particle_types), "Output directories must match the number of launched particle types"

        if self.cluster_configuration.nodes > 1:
            raise ValueError("CTLearn prediction tool can only be ran on a single GPU")
        self.cluster_configuration.info()
        import glob
        import os

        from astropy.io.misc.hdf5 import read_table_hdf5
        testing_files = []
        output_files = []
        for particle_type, output_dir in zip(launch_particle_types, output_dirs):
            direction_testing_table =  read_table_hdf5(self.direction_model.model_index_file, path=f'{self.direction_model.model_nickname}/testing/{particle_type.value}')
            energy_testing_table =  read_table_hdf5(self.energy_model.model_index_file, path=f'{self.energy_model.model_nickname}/testing/{particle_type.value}')
            type_testing_table =  read_table_hdf5(self.type_model.model_index_file, path=f'{self.type_model.model_nickname}/testing/{particle_type.value}')
            if not (direction_testing_table[f'testing_{particle_type.value}_dirs'] == energy_testing_table[f'testing_{particle_type.value}_dirs']).all() and (direction_testing_table[f'testing_{particle_type.value}_dirs'] == type_testing_table[f'testing_{particle_type.value}_dirs']).all():
                raise ValueError(f"All models must have the same testing {particle_type.value} directories, use set_testing_files to set them")
            if len(direction_testing_table[f'testing_{particle_type.value}_dirs']) == 0:
                raise ValueError(f"Testing {particle_type.value} directories cannot be empty")
            dirs = direction_testing_table[f'testing_{particle_type.value}_dirs']
            zeniths = direction_testing_table[f'testing_{particle_type.value}_zenith_distances']
            azimuths = direction_testing_table[f'testing_{particle_type.value}_azimuths']
            patterns = direction_testing_table[f'testing_{particle_type.value}_patterns']
            
            matching_dirs = [dirs[i] for i in range(len(dirs)) if zeniths[i] == zenith and azimuths[i] == azimuth]
            if not matching_dirs:
                raise ValueError(f"No matching {particle_type.value} directory found for zenith {zenith} and azimuth {azimuth}")
            dir = matching_dirs[0]
            pattern = [patterns[i] for i in range(len(patterns)) if zeniths[i] == zenith and azimuths[i] == azimuth][0]
            data_sample = DataSample(
                directory=dir,
                zenith_distance=zenith,
                azimuth=azimuth,
                pattern=pattern,
                particle_type=particle_type,
                )
            _files = np.sort(glob.glob(f"{dir}/{pattern}"))
            _output_files = [f"{output_dir}/{Path(file).stem.replace('dl1', 'dl2')}.h5" for file in _files]
            testing_files.extend(_files)
            output_files.extend(_output_files)
            for model in [self.direction_model, self.energy_model, self.type_model]:
                for file in _output_files:
                    model.update_model_manager_DL2_MC_file(
                        testing_MC_DL2_file=file,
                        testing_MC_DL2_data_sample=data_sample
                    )
        channels_string = ""
        for channel in self.channels:
            channels_string += f"--DLImageReader.channels={channel} "
        type_model_dir = np.sort(glob.glob(f"{self.type_model.model_parameters_table['model_dir'][0]}/{self.type_model.model_nickname}*"))[-1]
        energy_model_dir = np.sort(glob.glob(f"{self.energy_model.model_parameters_table['model_dir'][0]}/{self.energy_model.model_nickname}*"))[-1]
        direction_model_dir = np.sort(glob.glob(f"{self.direction_model.model_parameters_table['model_dir'][0]}/{self.direction_model.model_nickname}*"))[-1]

        dl2_subarray_string = " --dl2-subarray" if dl2_subarray else " --no-dl2-subarray"
        config_string = f"--config {config}" if config is not None else ""

        allowed_tels = ast.literal_eval(self.direction_model.model_parameters_table['telescope_ids'][0])
        # config['TrainCTLearnModel']['DLImageReader']['allowed_tels'] = allowed_tels # TODO pass allowed tels in a config file
         
        for input_file, output_file in zip(testing_files, output_files):
            if os.path.exists(output_file) and not overwrite:
                print(f"Output file {output_file} already exists, skipping, set overwrite=True to overwrite")
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
--PredictCTLearnModel.overwrite_tables=True -v {channels_string} \
{config_string}"
            else:
                # cmd = f"ctlearn-predict-mono --input_url {input_file} --type_model={type_model_dir}/ctlearn_model.cpk --energy_model={energy_model_dir}/ctlearn_model.cpk --direction_model={direction_model_dir}/ctlearn_model.cpk --no-dl1-images --no-true-images --output {output_file} --overwrite -v {channels_string}"
                cmd = f"ctlearn-predict-mono-model --input_url {input_file} \
--PredictCTLearnModel.batch_size={batch_size} \
--type_model={type_model_dir}/ctlearn_model.cpk \
--energy_model={energy_model_dir}/ctlearn_model.cpk \
--cameradirection_model={direction_model_dir}/ctlearn_model.cpk \
--no-dl1-images --no-true-images --output {output_file} \
--use-HDF5Merger{dl2_subarray_string} \
--PredictCTLearnModel.overwrite_tables=True -v {channels_string} \
{config_string}"
            
            if self.cluster_configuration.use_cluster:
                # sbatch_file = write_sbatch_script(cluster_configuration.cluster, Path(input_file).stem, cmd, config_dir, env_name=cluster_configuration.python_env, account=cluster_configuration.account)
                sbatch_file = self.cluster_configuration.write_sbatch_script(Path(input_file).stem, cmd, config_dir)
                os.system(f"sbatch {sbatch_file}")  
            else:
                print(cmd)
                os.system(cmd)
        
    
    def predict_lstchain_data(self, input_file, output_file, run=None, subrun=None, config_dir=None, overwrite=False, pointing_table='/dl1/event/telescope/parameters/LST_LSTCam', batch_size=64):
        """
        Predicts data using lstchain models and saves the output to a specified file.
        :param input_file: Path to the input file containing data to be predicted.
        :type input_file: str
        :param output_file: Path to the output file where predictions will be saved.
        :type output_file: str
        :param run: Run number to override observation ID, defaults to None.
        :type run: int, optional
        :param subrun: Subrun number to override observation ID, defaults to None.
        :type subrun: int, optional
        :param config_dir: Directory to save the configuration file, defaults to None.
        :type config_dir: str, optional
        :param overwrite: Flag to indicate whether to overwrite existing output file, defaults to False.
        :type overwrite: bool, optional
        :param pointing_table: Path to the pointing table in the input file, defaults to '/dl1/event/telescope/parameters/LST_LSTCam'.
        :type pointing_table: str, optional
        :return: None
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
        type_model_dir = np.sort(glob.glob(f"{self.type_model.model_parameters_table['model_dir'][0]}/{self.type_model.model_nickname}_v*"))[-1]
        energy_model_dir = np.sort(glob.glob(f"{self.energy_model.model_parameters_table['model_dir'][0]}/{self.energy_model.model_nickname}_v*"))[-1]
        direction_model_dir = np.sort(glob.glob(f"{self.direction_model.model_parameters_table['model_dir'][0]}/{self.direction_model.model_nickname}_v*"))[-1]
        allowed_tels = ast.literal_eval(self.direction_model.model_parameters_table['telescope_ids'][0])
        stereo_mode = 'stereo' if self.stereo else "mono"
        # stack_telescope_images = True if self.stereo else False
        config = {}
        config['LST1PredictionTool'] = {}

        # config['LST1PredictionTool']['allowed_tels'] = allowed_tels
        # config['LST1PredictionTool']['min_telescopes'] = int(len(allowed_tels))
        # config['LST1PredictionTool']['mode'] = stereo_mode
        # config['LST1PredictionTool']['stack_telescope_images'] = stack_telescope_images # Mono only
        config['LST1PredictionTool']['channels'] = self.channels
        # config['LST1PredictionTool']['dl1dh_reader_type'] = "DLImageReader"
        if (run is not None) and (subrun is not None):
            config['LST1PredictionTool']['override_obs_id'] = int(f"{run:05d}{subrun:04d}")
        config['LST1PredictionTool']['output_path'] = output_file
        config['LST1PredictionTool']['log_file'] = output_file.replace('.h5', '.log')
        config['LST1PredictionTool']['overwrite'] = overwrite

        config_file = f"{config_dir}/pred_config_{Path(input_file).stem}.json"
        with open(config_file, 'w') as file:
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
            sbatch_file = self.cluster_configuration.write_sbatch_script(Path(input_file).stem, cmd, config_dir)
            import os
            os.system(f"sbatch {sbatch_file}")
    
        else:
            print(cmd)
            os.system(cmd)
     

        print("")
        
    
    def predict_data(self, input_file, output_file, config_dir=None, overwrite=False, pointing_table='dl0/monitoring/subarray/pointing'):
        """
        Predict data using CTLearn models and save the results to the specified output file.
        :param input_file: str
            Path to the input file containing the data to be predicted.
        :param output_file: str
            Path to the output file where the prediction results will be saved.
        :param config_dir: str, optional
            Directory where the configuration file will be saved. Default is None.
        :param overwrite: bool, optional
            Whether to overwrite the existing output file. Default is False.
        :param pointing_table: str, optional
            Path to the pointing table in the input file. Default is 'dl0/monitoring/subarray/pointing'.
        :returns: None
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
        type_model_dir = np.sort(glob.glob(f"{self.type_model.model_parameters_table['model_dir'][0]}/{self.type_model.model_nickname}_v*"))[-1]
        energy_model_dir = np.sort(glob.glob(f"{self.energy_model.model_parameters_table['model_dir'][0]}/{self.energy_model.model_nickname}_v*"))[-1]
        direction_model_dir = np.sort(glob.glob(f"{self.direction_model.model_parameters_table['model_dir'][0]}/{self.direction_model.model_nickname}_v*"))[-1]
        allowed_tels = ast.literal_eval(self.direction_model.model_parameters_table['telescope_ids'][0])
        stereo_mode = 'stereo' if self.stereo else "mono"
        stack_telescope_images = True if self.stereo else False
        config = {}
        config['PredictCTLearnModel'] = {}
        config['PredictCTLearnModel']['DLImageReader'] = {}

        config['PredictCTLearnModel']['DLImageReader']['allowed_tels'] = allowed_tels
        config['PredictCTLearnModel']['DLImageReader']['min_telescopes'] = int(len(allowed_tels))
        config['PredictCTLearnModel']['DLImageReader']['mode'] = stereo_mode
        config['PredictCTLearnModel']['stack_telescope_images'] = stack_telescope_images
        config['PredictCTLearnModel']['DLImageReader']['channels'] = self.channels
        config['PredictCTLearnModel']['dl1dh_reader_type'] = "DLImageReader"
        config['PredictCTLearnModel']['output_path'] = output_file
        config['PredictCTLearnModel']['log_file'] = output_file.replace('.h5', '.log')
        config['PredictCTLearnModel']['overwrite'] = overwrite
    
        config_file = f"{config_dir}/pred_config_{Path(input_file).stem}.json"
        with open(config_file, 'w') as file:
            json.dump(config, file)
        print(f"Configuration saved to {config_file}")

        avg_data_ze, avg_data_az = get_avg_pointing(input_file, pointing_table=pointing_table)
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
            sbatch_file = self.cluster_configuration.write_sbatch_script(Path(input_file).stem, cmd, config_dir)
            os.system(f"sbatch {sbatch_file}")
        else:
            print(cmd)
            os.system(cmd)

        print("")
    
    
    # TODO add option to delete original files
    def merge_DL2_files(self, zenith: str, azimuth: str, output_file: str, particle_type: ParticleType, overwrite=False):
        """
        Merge DL2 files for given zenith and azimuth angles.
        This method merges DL2 gamma and proton files for the specified zenith and azimuth angles
        using the `ctapipe-merge` command. If there are multiple files to merge, the merged file
        is saved to the specified output file. If there is only one file, no merging is performed.
        The merged file paths are then updated in the direction, energy, and type models.
        :param zenith: Zenith angle for which to merge DL2 files.
        :type zenith: float
        :param azimuth: Azimuth angle for which to merge DL2 files.
        :type azimuth: float
        :param output_file_gammas: Path to the output file for merged gamma files. If None, no merging is performed for gamma files.
        :type output_file_gammas: str, optional
        :param output_file_protons: Path to the output file for merged proton files. If None, no merging is performed for proton files.
        :type output_file_protons: str, optional
        :param overwrite: Whether to overwrite existing merged files.
        :type overwrite: bool
        :raises RuntimeError: If the merging process fails for either gamma or proton files.
        """
        import os
        files = self.direction_model.get_DL2_MC_files(zenith, azimuth, particle_types = [particle_type])[particle_type.value]
        if len(files) > 1:
            print(f"🔀 Merging DL2 {particle_type.value} files for zenith {zenith} and azimuth {azimuth}")
            cmd = f"ctapipe-merge {' '.join(files)} --output={output_file} --progress --MergeTool.skip_broken_files=True {'--overwrite' if overwrite else ''}"
            print(f"Running : {cmd}")
            result = os.system(cmd)
            if result == 0:
                self.direction_model.update_merged_DL2_MC_files(zenith, azimuth, output_file, particle_type)
                self.energy_model.update_merged_DL2_MC_files(zenith, azimuth, output_file, particle_type)
                self.type_model.update_merged_DL2_MC_files(zenith, azimuth, output_file, particle_type)
                print("Original files still exist and were not erased.")
            else:
                print(f"Error: Failed to merge gamma files for zenith {zenith} and azimuth {azimuth}")
        else:
            print(f"✅ There already is a single {particle_type.value} file for zenith {zenith} and azimuth {azimuth}")

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def plot_DL2_classification(self, zenith: float, azimuth: float, particle_types: list[ParticleType]=[ParticleType.GAMMA_POINT, ParticleType.PROTON]):
        """
        Plots the DL2 classification results for gamma and proton events.
        This function generates a histogram plot showing the distribution of 
        CTLearn predictions for gamma and proton events based on the given 
        zenith and azimuth angles. The plot displays the density of predictions 
        for both classes.
        :param zenith: Zenith angle for which to retrieve DL2 MC files.
        :type zenith: float
        :param azimuth: Azimuth angle for which to retrieve DL2 MC files.
        :type azimuth: float
        """
        import matplotlib.pyplot as plt
        from astropy.table import vstack
        
        DL2_MC_files = self.direction_model.get_DL2_MC_files(zenith, azimuth, particle_types = particle_types)
        for particle_type in particle_types:
            testing_DL2_files = DL2_MC_files[particle_type.value]
            dl2_data = []
            tel_id = None if self.stereo else self.telescope_ids[0]
            for file in testing_DL2_files:
                dl2_data.append(load_DL2_data_MC(file, tel_id=tel_id))
            dl2_data = vstack(dl2_data)
            plt.hist(dl2_data[self.gammaness_key], bins=100, range=(0, 1), histtype="step", density=True, label=particle_type.value)
        plt.xlabel("Gammaness")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)  
    def plot_DL2_energy(self, zenith: float, azimuth: float, particle_types: list[ParticleType]=[ParticleType.GAMMA_POINT, ParticleType.PROTON]):
        """
        Plot the DL2 energy distribution for gamma and proton events.
        This function generates a histogram plot of the DL2 energy distribution for 
        gamma and proton events based on the given zenith and azimuth angles. The 
        energy values are plotted on a logarithmic scale.
        :param zenith: Zenith angle for which the DL2 data is to be plotted.
        :type zenith: float
        :param azimuth: Azimuth angle for which the DL2 data is to be plotted.
        :type azimuth: float
        :returns: None
        """
        import matplotlib.pyplot as plt
        from astropy.table import vstack
        
        DL2_MC_files = self.direction_model.get_DL2_MC_files(zenith, azimuth, particle_types = particle_types)
        for particle_type in particle_types:
            testing_DL2_files = DL2_MC_files[particle_type.value]
            dl2_data = []
            tel_id = None if self.stereo else self.telescope_ids[0]
            for file in testing_DL2_files:
                dl2_data.append(load_DL2_data_MC(file, tel_id=tel_id))
            dl2_data = vstack(dl2_data)
            plt.hist(dl2_data[self.reco_energy_key], bins=100, range=(0, 1), histtype="step", density=True, label=particle_type.value)
        plt.xlabel("Energy [TeV]")
        plt.ylabel("Density")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.show()

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)   
    def plot_DL2_AltAz(self, zenith: float, azimuth: float, particle_types: list[ParticleType]=[ParticleType.GAMMA_POINT], cuts: Cuts=DefaultCuts.NO_CUTS.value):
        """
        Plot the reconstructed Altitude and Azimuth for DL2 data.
        This function generates two subplots: one for gamma events and one for proton events.
        It visualizes the reconstructed altitude and azimuth using a 2D histogram and marks the array pointing direction.

        Parameters
        ----------
        zenith : float
            The zenith angle for which to get the DL2 MC files.
        azimuth : float
            The azimuth angle for which to get the DL2 MC files.

        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt
        from astropy.table import vstack
        if cuts.cut_type != CutType.GLOBAL:
            raise ValueError("Cuts must be global")
        
        fig, axs = plt.subplots(1, len(particle_types), figsize=(5*len(particle_types), 4))
        DL2_MC_files = self.direction_model.get_DL2_MC_files(zenith, azimuth, particle_types = particle_types)
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
            ax.scatter(dl2_data[self.pointing_alt_key][0]/np.pi*180, dl2_data[self.pointing_az_key][0]/np.pi*180, color=CTLearnManagerStyle.ctlearn_accent_1.value, label="Array pointing", marker="x", s=80)
            ax.hist2d(dl2_data[self.reco_alt_key], dl2_data[self.reco_az_key], bins=100, zorder=0, cmap="viridis", norm=plt.cm.colors.LogNorm())
            ax.set_xlabel("Altitude [deg]")
            ax.set_ylabel("Azimuth [deg]")
            ax.legend()
            ax.set_title(particle_type.value)
            cbar = plt.colorbar(ax.collections[1], ax=ax)
            cbar.set_label("Counts")
        plt.tight_layout()
        plt.show()

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)   
    def plot_migration_matrix(self, zenith: float, azimuth: float, particle_types: list[ParticleType]=[ParticleType.GAMMA_POINT], cuts: Cuts=DefaultCuts.NO_CUTS.value):    
        """
        Plot the migration matrix for gamma and proton events.
        This function generates a 2D histogram plot of the reconstructed energy 
        versus the true energy for both gamma and proton events. The plots are 
        displayed side by side for comparison.

        Parameters
        ----------
        zenith : float
            The zenith angle of the observation.
        azimuth : float
            The azimuth angle of the observation.

        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt
        from astropy.table import join, vstack
        if cuts.cut_type != CutType.GLOBAL:
            raise ValueError("Cuts must be global")
        
        fig, axs = plt.subplots(1, len(particle_types) , figsize=(5*len(particle_types), 4))
        DL2_MC_files = self.direction_model.get_DL2_MC_files(zenith, azimuth, particle_types = particle_types)
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
            dl2_data = join(dl2_data, shower_parameters, keys=["obs_id", "event_id"])[dl2_data[self.gammaness_key] > cuts.gammaness_cut]

            log_bins = np.logspace(
                    np.log10(min((min(dl2_data[self.reco_energy_key]), min(dl2_data[self.true_energy_key])))), 
                    np.log10(max(max(dl2_data[self.reco_energy_key]), max(dl2_data[self.true_energy_key]))),
                    100)
            if len(particle_types) > 1:
                ax = axs[i]
            else:
                ax = axs
            cuts.plot_cuts_info_plt(ax)
            ax.plot([log_bins[0], log_bins[-1]], [log_bins[0], log_bins[-1]], color=CTLearnManagerStyle.ctlearn_accent_1.value, ls="--")
            ax.hist2d(dl2_data[self.reco_energy_key], dl2_data[self.true_energy_key], bins=log_bins, cmap="viridis", norm=plt.cm.colors.LogNorm())
            ax.set_xlabel("CTLean Energy [TeV]")
            ax.set_ylabel("True Energy [TeV]")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(log_bins[0], log_bins[-1])
            ax.set_ylim(log_bins[0], log_bins[-1])
            ax.axis('equal')
            ax.set_title(f"{particle_type.value}")
            cbar = plt.colorbar(ax.collections[0], ax=ax)
            cbar.set_label("Counts")

        plt.tight_layout()
        plt.show()

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def produce_irfs(self, zenith: float, azimuth: float, config: str, output_cuts_file: str, output_irf_file: str, output_benchmark_file: str, pointlike=True, electrons=False, protons=True, overwrite=False):
        """
        Produce Instrument Response Functions (IRFs) for given zenith and azimuth angles.
        This method generates IRFs by running external commands and updating the model manager with the necessary data.
        If configuration files are not provided, it attempts to retrieve them from the direction model.
        :param zenith: Zenith angle for which to produce IRFs.
        :type zenith: float
        :param azimuth: Azimuth angle for which to produce IRFs.
        :type azimuth: float
        :param config: Path to the configuration file. If None, it will be retrieved from the direction model.
        :type config: str, optional
        :param output_cuts_file: Path to the output cuts file. If None, it will be retrieved from the direction model.
        :type output_cuts_file: str, optional
        :param output_irf_file: Path to the output IRF file. If None, it will be retrieved from the direction model.
        :type output_irf_file: str, optional
        :param output_benchmark_file: Path to the output benchmark file. If None, it will be retrieved from the direction model.
        :type output_benchmark_file: str, optional
        :raises ValueError: If any of the required files (config, output_cuts_file, output_irf_file, output_benchmark_file) are not provided and cannot be retrieved.
        :raises ValueError: If multiple gamma or proton files are found for the given zenith and azimuth angles.
        """

        import os
        # irf_type, gammaness_efficiency, theta_efficiency = get_irf_type_from_config(config)
        # match irf_type:
        #     case IRFType.EFFICIENCY_OPTIMIZED:
        #         cuts_type = CutsType.EFFICIENCY_OPTIMIZED
        #     case IRFType.SENSITIVITY_OPTIMIZED:
        #         cuts_type = CutsType.SENSITIVITY_OPTIMIZED
        # cuts = Cuts(cuts_type, gammaness_efficiency = gammaness_efficiency, theta_efficiency = theta_efficiency)
        # if config is None:
        #     try:
        #         config = self.direction_model.get_IRF_data(zenith, azimuth)[0]
        #     except:
        #         raise ValueError("A configuration file must be provided, at least the first time.")
        # if output_cuts_file is None:
        #     try:
        #         output_cuts_file = self.direction_model.get_IRF_data(zenith, azimuth)[1]
        #     except:
        #         raise ValueError("An output cuts file must be provided, at least the first time.")
        # if output_irf_file is None:
        #     try:
        #         output_irf_file = self.direction_model.get_IRF_data(zenith, azimuth)[2]
        #     except:
        #         raise ValueError("An output IRF file must be provided, at least the first time.")
        # if output_benchmark_file is None:
        #     try:
        #         output_benchmark_file = self.direction_model.get_IRF_data(zenith, azimuth)[3]
        #     except:
        #         raise ValueError("An output benchmark file must be provided, at least the first time.")
        
        if pointlike:
            gamma_files = self.direction_model.get_DL2_MC_files(zenith, azimuth, particle_types=[ParticleType.GAMMA_POINT])[ParticleType.GAMMA_POINT.value]
        else:
            gamma_files = self.direction_model.get_DL2_MC_files(zenith, azimuth, particle_types=[ParticleType.GAMMA_DIFFUSE])[ParticleType.GAMMA_DIFFUSE.value]
        if len(gamma_files) > 1:
            raise ValueError(f"Multiple files found for gamma, zenith {zenith} and azimuth {azimuth}, please merge them first with CTLearnTriModelManager.merge_DL2_files()")
        gamma_file = gamma_files[0]
        if electrons:
            electrons_files = self.direction_model.get_DL2_MC_files(zenith, azimuth, particle_types=[ParticleType.ELECTRON])[ParticleType.ELECTRON.value]
            if len(electrons_files) > 1:
                raise ValueError(f"Multiple files found for electrons, zenith {zenith} and azimuth {azimuth}, please merge them first with CTLearnTriModelManager.merge_DL2_files()")
            electron_file = electrons_files[0]

        if protons:
            proton_files = self.direction_model.get_DL2_MC_files(zenith, azimuth, particle_types=[ParticleType.PROTON])[ParticleType.PROTON.value]
            if len(proton_files) > 1:
                raise ValueError(f"Multiple files found for proton, zenith {zenith} and azimuth {azimuth}, please merge them first with CTLearnTriModelManager.merge_DL2_files()")
            proton_file = proton_files[0]

        os.makedirs(output_cuts_file.rsplit('/', 1)[0], exist_ok=True)
        os.makedirs(output_irf_file.rsplit('/', 1)[0], exist_ok=True)
        os.makedirs(output_benchmark_file.rsplit('/', 1)[0], exist_ok=True)
        
        electron_string = f" --electron-file {electron_file}" if electrons else ""
        proton_string = f" --proton-file {proton_file}" if protons else ""
        cmd = f"ctapipe-optimize-event-selection \
-c {config} \
--gamma-file {gamma_file} \
{proton_string} \
{electron_string} \
--output {output_cuts_file} \
--overwrite True"
        print(cmd)
            # --EventSelectionOptimizer.optimization_algorithm=PercentileCuts"
        result_cuts = os.system(cmd)
        if result_cuts != 0:
            raise RuntimeError(f"Error: Failed to produce cuts file for zenith {zenith} and azimuth {azimuth}")
        cmd = f"ctapipe-compute-irf \
-c {config} --IrfTool.cuts_file {output_cuts_file} \
--gamma-file {gamma_file} \
{proton_string} \
{electron_string} \
--do-background \
--output {output_irf_file} \
--benchmark-output {output_benchmark_file} \
--no-spatial-selection-applied --overwrite --spatial-selection-applied"
        print(cmd)
        result_irfs = os.system(cmd)
        if result_irfs != 0:
            raise RuntimeError(f"Error: Failed to produce IRF file for zenith {zenith} and azimuth {azimuth}")
        self.direction_model.update_model_manager_IRF_data(config, output_cuts_file, output_irf_file, output_benchmark_file, zenith, azimuth)
        self.energy_model.update_model_manager_IRF_data(config, output_cuts_file, output_irf_file, output_benchmark_file, zenith, azimuth)
        self.type_model.update_model_manager_IRF_data(config, output_cuts_file, output_irf_file, output_benchmark_file, zenith, azimuth)
    

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def plot_benchmark(self, zenith: float, azimuth: float, cuts: list[Cuts]=[DefaultCuts.EFF_70.value], containments: list[int]=[68, 95], title: str=None):
        """
        Plot benchmark graphs for sensitivity, angular resolution, energy resolution, and energy bias 
        based on the given zenith and azimuth angles.

        Parameters
        ----------
        zenith : float
            The zenith angle for which the IRF data is to be retrieved.
        azimuth : float
            The azimuth angle for which the IRF data is to be retrieved.

        Returns
        -------
        None
        """
        
        import matplotlib.pyplot as plt
        from astropy.io import fits
        fig, ax = plt.subplots()
        if len(cuts) == 1:
            cuts[0].plot_cuts_info_plt(ax)
        for cut in cuts:
            irf_file = self.direction_model.get_IRF_data(zenith, azimuth, cut)[3]
            hudl = fits.open(irf_file)
            energy_center = hudl['SENSITIVITY'].data['ENERG_LO'] + 0.5 * (hudl['SENSITIVITY'].data['ENERG_HI'] - hudl['SENSITIVITY'].data['ENERG_LO'])
            if len(cuts) > 1:
                plt.plot(energy_center[0], hudl['SENSITIVITY'].data['ENERGY_FLUX_SENSITIVITY'][0,0,:], label=cut.get_label())
            else:
                plt.plot(energy_center[0], hudl['SENSITIVITY'].data['ENERGY_FLUX_SENSITIVITY'][0,0,:])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Energy [TeV]')
        plt.ylabel('Sensitivity [erg s$^{-1}$ cm$^{-2}$]')
        if len(cuts) > 1:
            plt.legend()
        if title is not None:
            plt.title(title)
        plt.show()
        
        fig, ax = plt.subplots()
        if len(cuts) == 1:
            cuts[0].plot_cuts_info_plt(ax)
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for cut, color in zip(cuts, default_colors[:len(cuts)]):
            irf_file = self.direction_model.get_IRF_data(zenith, azimuth, cut)[3]
            hudl = fits.open(irf_file)
            energy_center = hudl['ANGULAR RESOLUTION '].data['ENERG_LO'] + 0.5 * (hudl['ANGULAR RESOLUTION '].data['ENERG_HI'] - hudl['ANGULAR RESOLUTION '].data['ENERG_LO'])
            line_styles = ['-', '--', '-.', ':']
            for containment, line_style in zip(containments, line_styles):
                plt.plot(energy_center[0], hudl['ANGULAR RESOLUTION'].data[f'ANGULAR_RESOLUTION_{containment}'][0,0,:], color=color, ls=line_style)

        # plt.plot(energy_center[0], hudl['ANGULAR RESOLUTION'].data['ANGULAR_RESOLUTION_25'][0,0,:], label='25%')
        # plt.plot(energy_center[0], hudl['ANGULAR RESOLUTION'].data['ANGULAR_RESOLUTION_50'][0,0,:], label='50%')
        # plt.plot(energy_center[0], hudl['ANGULAR RESOLUTION'].data['ANGULAR_RESOLUTION_68'][0,0,:], label='68%')
        # plt.plot(energy_center[0], hudl['ANGULAR RESOLUTION'].data['ANGULAR_RESOLUTION_95'][0,0,:], label='95%')
        plt.xscale('log')
        plt.xlabel('Energy [TeV]')
        plt.ylabel('Angular resolution [deg]')
        # Create separate legends for cuts and containment percentages
        # Create separate legends for cuts and containment percentages
        cut_labels = [cut.get_label() for cut in cuts]
        containment_labels = [f"{containment}%" for containment in containments]
        cut_legend = ax.legend(handles=[plt.Line2D([0], [0], color=color, lw=2) for color in default_colors[:len(cuts)]],
                       labels=cut_labels, loc='best')
        containment_legend = ax.legend(handles=[plt.Line2D([0], [0], color='black', ls=ls, lw=2) for ls in line_styles[:len(containments)]],
                           labels=containment_labels, loc='lower left', title="Containment")
        if len(cuts) > 1:
            ax.add_artist(cut_legend)
        
        if title is not None:
            plt.title(title)
        plt.show()

        fig, ax = plt.subplots()
        if len(cuts) == 1:
            cuts[0].plot_cuts_info_plt(ax)
        for cut in cuts:
            irf_file = self.direction_model.get_IRF_data(zenith, azimuth, cut)[3]
            hudl = fits.open(irf_file)
            energy_center = hudl['ENERGY BIAS RESOLUTION'].data['ENERG_LO'] + 0.5 * (hudl['ENERGY BIAS RESOLUTION'].data['ENERG_HI'] - hudl['ENERGY BIAS RESOLUTION'].data['ENERG_LO'])
            if len(cuts) > 1:
                plt.plot(energy_center[0], hudl['ENERGY BIAS RESOLUTION'].data['RESOLUTION'][0,0,:], label=cut.get_label())
            else:
                plt.plot(energy_center[0], hudl['ENERGY BIAS RESOLUTION'].data['RESOLUTION'][0,0,:])
        plt.xscale('log')
        plt.xlabel('Energy [TeV]')
        plt.ylabel('Energy resolution')
        if len(cuts) > 1:
            plt.legend()
        if title is not None:
            plt.title(title)
        plt.show()
        

        fig, ax = plt.subplots()
        if len(cuts) == 1:
            cuts[0].plot_cuts_info_plt(ax)
        for cut in cuts:
            irf_file = self.direction_model.get_IRF_data(zenith, azimuth, cut)[3]
            hudl = fits.open(irf_file)
            energy_center = hudl['ENERGY BIAS RESOLUTION'].data['ENERG_LO'] + 0.5 * (hudl['ENERGY BIAS RESOLUTION'].data['ENERG_HI'] - hudl['ENERGY BIAS RESOLUTION'].data['ENERG_LO'])
            if len(cuts) > 1:
                plt.plot(energy_center[0], hudl['ENERGY BIAS RESOLUTION'].data['BIAS'][0,0,:], label=cut.get_label())
            else:
                plt.plot(energy_center[0], hudl['ENERGY BIAS RESOLUTION'].data['BIAS'][0,0,:])
        plt.xscale('log')
        plt.xlabel('Energy [TeV]')
        plt.ylabel('Energy bias')
        if len(cuts) > 1:
            plt.legend()
        if title is not None:
            plt.title(title)
        plt.show()
        hudl.close() 

    def plot_cuts(self, zenith: float, azimuth: float, cuts: list[Cuts]=[DefaultCuts.EFF_70.value]):
        """
        Plot the cuts for given zenith and azimuth angles.
        This method reads the cuts data from the specified IRF file and plots the cuts
        using the `peek` method from the `gammapy.irf` module.
        :param zenith: Zenith angle for which to retrieve and plot the cuts.
        :type zenith: float
        :param azimuth: Azimuth angle for which to retrieve and plot the cuts.
        :type azimuth: float
        """
        
        from astropy.io import fits
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        
        for cut in cuts:
            cuts_file = self.direction_model.get_IRF_data(zenith, azimuth, cut)[1]
            print(cuts_file)
            if len(cuts) > 1:
                label = cut.get_label()
            else:
                label = ""
            with fits.open(cuts_file) as hdul:
                axs[0].plot(hdul['GH_CUTS'].data['center'], hdul['GH_CUTS'].data['cut'], label=label)
                axs[0].set_xlabel("Energy [TeV]")
                axs[0].set_ylabel("Gammaness cut")
                axs[0].set_xscale('log')

                axs[1].plot(hdul['RAD_MAX'].data['center'], hdul['RAD_MAX'].data['cut'], label=label)
                axs[1].set_xlabel("Energy [TeV]")
                axs[1].set_ylabel("Theta cut [deg]")
                axs[1].set_xscale('log')
        if len(cuts) == 1:
            cuts[0].plot_cuts_info_plt(axs[0])
            cuts[0].plot_cuts_info_plt(axs[1])
        else:
            axs[0].legend()
            axs[1].legend()
            

        plt.tight_layout()
        plt.show()

    
    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def plot_irfs(self, zenith, azimuth):
        """
        Plot the Instrument Response Functions (IRFs) for given zenith and azimuth angles.
        This method reads the IRF data for the specified zenith and azimuth angles, and then
        plots the Effective Area, Background, and Energy Dispersion using the `peek` method
        from the `gammapy.irf` module.
        :param zenith: Zenith angle for which to retrieve and plot the IRFs.
        :type zenith: float
        :param azimuth: Azimuth angle for which to retrieve and plot the IRFs.
        :type azimuth: float
        """
        
        from gammapy.irf import (
            Background2D,
            EffectiveAreaTable2D,
            EnergyDispersion2D,
        )
        irf_file = self.direction_model.get_IRF_data(zenith, azimuth)[2]
        # rad_max = RadMax2D.read(irf_file, hdu="RAD MAX")
        aeff = EffectiveAreaTable2D.read(irf_file, hdu="EFFECTIVE AREA")
        bkg = Background2D.read(irf_file, hdu="BACKGROUND")
        edisp = EnergyDispersion2D.read(irf_file, hdu="ENERGY DISPERSION")
        edisp.peek()
        aeff.peek()
        bkg.peek()
        
    def plot_loss(self):
        """
        Plot the training and validation loss for direction, energy, and type models.
        This method reads the training logs for the direction, energy, and type models,
        concatenates the loss values, and plots them using matplotlib.
        The plot will display three subplots, one for each model, showing the training
        and validation loss over epochs.
        The method assumes that the training logs are stored in CSV files with columns
        'loss' and 'val_loss' for training and validation loss respectively.
        The CSV files are expected to be located in directories specified by the
        'model_dir' and 'model_nickname' attributes of each model's 'model_parameters_table'.
        The method uses the `set_mpl_style` function to set the matplotlib style.

        Raises
        ------
            FileNotFoundError: If no training log files are found for any of the models.
        """
        
        import glob

        import matplotlib.pyplot as plt
        import pandas as pd
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, model in zip(axs, [self.direction_model, self.energy_model, self.type_model]):
            # print(f"{model.model_parameters_table['model_dir'][0]}/{model.model_nickname}*/training_log.csv")
            training_logs = np.sort(glob.glob(f"{model.model_parameters_table['model_dir'][0]}/{model.model_nickname}*/training_log.csv"))
            if len(training_logs) == 0:
                # print(f"{model.model_parameters_table['model_dir'][0]}/{model.model_nickname}/{model.model_nickname}*/training_log.csv")
                training_logs = np.sort(glob.glob(f"{model.model_parameters_table['model_dir'][0]}/{model.model_nickname}/{model.model_nickname}*/training_log.csv"))
            # print(training_logs)
            losses_train = []
            losses_val = []
            for training_log in training_logs:
                df = pd.read_csv(training_log)
                losses_train = np.concatenate((losses_train, df['loss'].to_numpy()))
                losses_val = np.concatenate((losses_val, df['val_loss'].to_numpy()))
            epochs = np.arange(1, len(losses_train)+1)
            if len(epochs) > 1:
                ax.plot(epochs, losses_train, label="Training", lw=2)
                ax.plot(epochs, losses_val, label="Validation", ls='--')
            else:
                ax.scatter(epochs, losses_train, label="Training", lw=2)
                ax.scatter(epochs, losses_val, label="Validation", ls='--')
            ax.set_title(f"{model.model_parameters_table['reco'][0]} training".title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_xticks(np.arange(1, len(epochs) + 1, 2))
            ax.legend()
        plt.tight_layout()
        plt.show()
    
    @u.quantity_input(zeniths=u.deg,azimuths=u.deg)
    def plot_angular_resolution_DL2(self, zeniths: list[float] = None, azimuths: list[float] = None, cuts: list[Cuts]=[DefaultCuts.NO_CUTS.value], ylim=None, particle_type: ParticleType=ParticleType.GAMMA_POINT, figsize=None):
        """
        Plot the angular resolution for DL2 data at a given zenith and azimuth angle.
        This function reads DL2 gamma-ray data from HDF5 files, processes the data to 
        obtain reconstructed and true shower parameters, and then plots the angular 
        resolution as a function of true energy using ctaplot.

        Parameters
        ----------
        zenith : float
            The zenith angle for which to plot the angular resolution.
        azimuth : float
            The azimuth angle for which to plot the angular resolution.

        Returns
        -------
        None
        """
        
        import astropy.units as u
        import ctaplot
        import matplotlib.pyplot as plt
        from astropy.io.misc.hdf5 import read_table_hdf5
        from astropy.table import join, vstack
        if zeniths is None:
            coords = self.get_available_MC_directions(verbose=False)
        else:
            assert len(zeniths) == len(azimuths), "zeniths and azimuths must have the same length"
            coords = list(zip(zeniths, azimuths))

        assert len(coords) == 1 or len(cuts) == 1, "Either zeniths/azimuths or 'cuts' must have a length of 1"

        avg_model_az = np.mean(self.direction_model.validity.azimuth_range).to(u.deg)
        avg_model_ze = np.mean(self.direction_model.validity.zenith_range).to(u.deg)
        testing_azs = np.empty(len(coords)) * u.deg
        testing_zes = np.empty(len(coords)) * u.deg
        i = 0
        for zenith, azimuth in coords:
            testing_azs[i] = azimuth.to(u.deg)
            testing_zes[i] = zenith.to(u.deg)
            i += 1
        closest_coord_index = np.argmin(angular_distance(avg_model_ze, avg_model_az, testing_zes, testing_azs))
        
        DL2_gamma_table = read_table_hdf5(self.direction_model.model_index_file, path=f'{self.direction_model.model_nickname}/DL2/MC/{particle_type.value}')

        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()

        if len(cuts) == 1:
            cuts[0].plot_cuts_info_plt(ax)

        for i, coord in enumerate(coords):
            for cut in cuts:
                
                zenith, azimuth = coord
                testing_DL2_gamma_files = DL2_gamma_table[f'testing_DL2_{particle_type.value}_files'][
                    (DL2_gamma_table[f'testing_DL2_{particle_type.value}_zenith_distances'] == zenith) &
                    (DL2_gamma_table[f'testing_DL2_{particle_type.value}_azimuths'] == azimuth)
                ]
                # testing_DL2_gamma_files = DL2_gamma_table['testing_DL2_gamma_files'][((DL2_gamma_table['testing_DL2_gamma_zenith_distances'] == zenith) and (DL2_gamma_table['testing_DL2_gamma_azimuths'] == azimuth)).all()]
                dl2_gamma = []
                shower_parameters_gamma = []
                tel_id = None if self.stereo else self.telescope_ids[0]
                for file in testing_DL2_gamma_files:
                    dl2_gamma.append(load_DL2_data_MC(file, tel_id=tel_id))
                    shower_parameters_gamma.append(load_true_shower_parameters(file))
                dl2_gamma = vstack(dl2_gamma)
                shower_parameters_gamma = vstack(shower_parameters_gamma)
                dl2_gamma = join(dl2_gamma, shower_parameters_gamma, keys=["obs_id", "event_id"])
                

                match cut.cut_type:
                    case CutType.GLOBAL:
                        mask = dl2_gamma[self.gammaness_key] > cut.gammaness_cut
                        reco_alt = dl2_gamma[self.reco_alt_key].to(u.deg) [mask]
                        reco_az = dl2_gamma[self.reco_az_key].to(u.deg) [mask]
                        true_alt = dl2_gamma[self.true_alt_key].to(u.deg) [mask]
                        true_az = dl2_gamma[self.true_az_key].to(u.deg) [mask]
                        reco_energy = dl2_gamma[self.reco_energy_key] [mask]
                        true_energy = dl2_gamma[self.true_energy_key] [mask]  

                    case CutType.EFFICIENCY_OPTIMIZED | CutType.SENSITIVITY_OPTIMIZED: 
                        cuts_file = self.direction_model.get_IRF_data(zenith, azimuth, cut)[1]
                        dl2_gamma = self.apply_energy_dependent_cuts_MC(dl2_gamma, cuts_file, theta_cut=False)  
                        reco_alt = dl2_gamma[self.reco_alt_key].to(u.deg)
                        reco_az = dl2_gamma[self.reco_az_key].to(u.deg)
                        true_alt = dl2_gamma[self.true_alt_key].to(u.deg)
                        true_az = dl2_gamma[self.true_az_key].to(u.deg)
                        reco_energy = dl2_gamma[self.reco_energy_key]
                        true_energy = dl2_gamma[self.true_energy_key]    
                    case _:
                        raise ValueError(f"Unknown cut type: {cut.cut_type}")                          
                # Define the range of true energy values
                true_energy_min = np.min(true_energy)
                true_energy_max = np.max(true_energy)
                reco_energy_min = np.min(reco_energy)
                reco_energy_max = np.max(reco_energy)

                plt.xlim(reco_energy_min, reco_energy_max)

                # Create bins with 5 bins per decade in log scale
                bins_per_decade = 5
                log_bins = np.logspace(np.log10(true_energy_min), np.log10(true_energy_max), 
                                    num=int(np.log10(true_energy_max/true_energy_min) * bins_per_decade) + 1) * u.TeV
                cut.efficiency_theta = None
                if len(cuts) == 1:
                    if i == closest_coord_index:
                        label = f"Closest to training data\n{particle_type.value} ({zenith.value:.1f}, {azimuth.value:.1f})°" if len(coords) > 1 else f"{particle_type.value} ({zenith.value:.1f}, {azimuth.value:.1f})°"
                        ctaplot.plot_angular_resolution_per_energy(true_alt, reco_alt, true_az, reco_az, true_energy, bins=log_bins, label=label, markersize=8)
                    else:
                        ctaplot.plot_angular_resolution_per_energy(true_alt, reco_alt, true_az, reco_az, true_energy, bins=log_bins, label=f"{particle_type.value} ({zenith.value:.1f}, {azimuth.value:.1f})°", alpha=0.5, marker='v')
                else:
                    ctaplot.plot_angular_resolution_per_energy(true_alt, reco_alt, true_az, reco_az, true_energy, bins=log_bins, label=cut.get_label(), markersize=8)


        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        plt.xlabel("True Energy [TeV]")
        plt.legend()
        plt.grid(False, which='both')
        plt.show()

    @u.quantity_input(zeniths=u.deg,azimuths=u.deg)    
    def plot_energy_resolution_DL2(self, zeniths: list[float] = None, azimuths: list[float] = None, cuts: list[Cuts]=[DefaultCuts.NO_CUTS.value], ylim=None, particle_type: ParticleType=ParticleType.GAMMA_POINT, figsize=None):
        """
        Plot the energy resolution for DL2 data at given zenith and azimuth angles.
        This function reads DL2 gamma data from HDF5 files, processes it to obtain
        reconstructed and true energy values, and then plots the energy resolution
        using ctaplot.

        Parameters
        ----------
        zenith : float
            The zenith angle for which the energy resolution is to be plotted.
        azimuth : float
            The azimuth angle for which the energy resolution is to be plotted.

        Returns
        -------
        None
        """
        
        import astropy.units as u
        import ctaplot
        import matplotlib.pyplot as plt
        from astropy.io.misc.hdf5 import read_table_hdf5
        from astropy.table import join, vstack

        if zeniths is None:
            coords = self.get_available_MC_directions(verbose=False)
        else:
            assert len(zeniths) == len(azimuths), "zeniths and azimuths must have the same length"
            coords = list(zip(zeniths, azimuths))

        assert len(coords) == 1 or len(cuts) == 1, "Either zeniths/azimuths or 'cuts' must have a length of 1"

        avg_model_az = np.mean(self.direction_model.validity.azimuth_range).to(u.deg)
        avg_model_ze = np.mean(self.direction_model.validity.zenith_range).to(u.deg)
        testing_azs = np.empty(len(coords)) * u.deg
        testing_zes = np.empty(len(coords)) * u.deg
        i = 0
        for zenith, azimuth in coords:
            testing_azs[i] = azimuth.to(u.deg)
            testing_zes[i] = zenith.to(u.deg)
            i += 1
        closest_coord_index = np.argmin(angular_distance(avg_model_ze, avg_model_az, testing_zes, testing_azs))
        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()

        if len(cuts) == 1:
            cuts[0].plot_cuts_info_plt(ax)
           
        DL2_gamma_table = read_table_hdf5(self.direction_model.model_index_file, path=f'{self.direction_model.model_nickname}/DL2/MC/{particle_type.value}')
        for i, coord in enumerate(coords):
            for cut in cuts:
                zenith, azimuth = coord
                testing_DL2_gamma_files = DL2_gamma_table[f'testing_DL2_{particle_type.value}_files'][
                    (DL2_gamma_table[f'testing_DL2_{particle_type.value}_zenith_distances'] == zenith) &
                    (DL2_gamma_table[f'testing_DL2_{particle_type.value}_azimuths'] == azimuth)
                ]
                # testing_DL2_gamma_files = DL2_gamma_table['testing_DL2_gamma_files'][DL2_gamma_table['testing_DL2_gamma_zenith_distances'] == zenith][DL2_gamma_table['testing_DL2_gamma_azimuths'] == azimuth]
                dl2_gamma = []
                shower_parameters_gamma = []
                tel_id = None if self.stereo else self.telescope_ids[0]
                for file in testing_DL2_gamma_files:
                    dl2_gamma.append(load_DL2_data_MC(file, tel_id))
                    shower_parameters_gamma.append(load_true_shower_parameters(file))
                dl2_gamma = vstack(dl2_gamma)
                shower_parameters_gamma = vstack(shower_parameters_gamma)
                dl2_gamma = join(dl2_gamma, shower_parameters_gamma, keys=["obs_id", "event_id"])

                match cut.cut_type:
                    case CutType.GLOBAL:
                        mask = dl2_gamma[self.gammaness_key] > cut.gammaness_cut
                        reco_energy = dl2_gamma[self.reco_energy_key] [mask]
                        true_energy = dl2_gamma[self.true_energy_key] [mask]   

                    case CutType.EFFICIENCY_OPTIMIZED | CutType.SENSITIVITY_OPTIMIZED:
                        cuts_file = self.direction_model.get_IRF_data(zenith, azimuth, cut)[1]
                        dl2_gamma = self.apply_energy_dependent_cuts_MC(dl2_gamma, cuts_file)
                        reco_energy = dl2_gamma[self.reco_energy_key]
                        true_energy = dl2_gamma[self.true_energy_key]  
                    case _:
                        raise ValueError(f"Unknown cut type: {cut.cut_type}")   
                     
                # Define the range of true energy values
                true_energy_min = np.min(true_energy)
                true_energy_max = np.max(true_energy)
                reco_energy_min = np.min(reco_energy)
                reco_energy_max = np.max(reco_energy)

                plt.xlim(reco_energy_min, reco_energy_max)

                # Create bins with 5 bins per decade in log scale
                bins_per_decade = 5
                log_bins = np.logspace(np.log10(true_energy_min), np.log10(true_energy_max), 
                                    num=int(np.log10(true_energy_max/true_energy_min) * bins_per_decade) + 1) * u.TeV
                if len(cuts) == 1:
                    if i == closest_coord_index:
                        label = f"Closest to training data\n{particle_type.value} ({zenith.value:.1f}, {azimuth.value:.1f})°" if len(coords) > 1 else f"{particle_type.value} ({zenith.value:.1f}, {azimuth.value:.1f})°"
                        ctaplot.plot_energy_resolution(true_energy, reco_energy, bins=log_bins, label=label, markersize=8)
                    else:
                        ctaplot.plot_energy_resolution(true_energy, reco_energy, bins=log_bins, label=f"{particle_type.value} ({zenith.value:.1f}, {azimuth.value:.1f})°", alpha=0.5, marker='v')
                else:
                    ctaplot.plot_energy_resolution(true_energy, reco_energy, bins=log_bins, label=cut.get_label(), markersize=8)
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        
        plt.legend()
        plt.grid(False, which='both')
        plt.show()
        
    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def plot_ROC_curve_DL2(self, zenith: float, azimuth: float, nbins: int=10):
        """
        Plot the ROC curve for DL2 data.
        This function generates and plots the ROC curve for Data Level 2 (DL2) 
        data for given zenith and azimuth angles. It uses gamma and proton Monte Carlo 
        (MC) files to compute the ROC curve based on the gammaness score and true 
        energy of the events.
        :param zenith: Zenith angle for the DL2 data.
        :type zenith: float
        :param azimuth: Azimuth angle for the DL2 data.
        :type azimuth: float
        :param nbins: Number of energy bins for the ROC curve, defaults to 10.
        :type nbins: int, optional
        :raises ValueError: If no DL2 gamma or proton files are found for the given 
                            zenith and azimuth angles.
        :returns: None
        """
        
        import astropy.units as u
        import ctaplot
        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.table import join, vstack

        testing_DL2_gamma_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[ParticleType.GAMMA_POINT.value]
        testing_DL2_proton_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[ParticleType.PROTON.value]

        tel_id = None if self.stereo else self.telescope_ids[0]

        if len(testing_DL2_gamma_files) > 0:
            dl2_gamma = []
            shower_parameters_gamma = []
            for file in testing_DL2_gamma_files:
                print(file)
                dl2_gamma.append(load_DL2_data_MC(file, tel_id=tel_id))
                shower_parameters_gamma.append(load_true_shower_parameters(file))
            dl2_gamma = vstack(dl2_gamma)
            shower_parameters_gamma = vstack(shower_parameters_gamma)
            dl2_gamma = join(dl2_gamma, shower_parameters_gamma, keys=["obs_id", "event_id"])
        else:
            dl2_gamma = []
        mc_type_gamma = np.zeros(len(dl2_gamma))
        
        
        if len(testing_DL2_proton_files) > 0:
            dl2_protons = []
            shower_parameters_protons = []
            for file in testing_DL2_proton_files:
                print(file)
                dl2_protons.append(load_DL2_data_MC(file, tel_id=tel_id))
                shower_parameters_protons.append(load_true_shower_parameters(file))
            dl2_proton = vstack(dl2_protons)
            shower_parameters_protons = vstack(shower_parameters_protons)
            dl2_proton = join(dl2_proton, shower_parameters_protons, keys=["obs_id", "event_id"])
        else:
            dl2_proton = []
        mc_type_proton = np.ones(len(dl2_proton))
            
        mc_type = np.concatenate((mc_type_gamma, mc_type_proton))
        gammaness = np.concatenate((dl2_gamma[self.gammaness_key], dl2_proton[self.gammaness_key]))
        mc_gamma_energies = np.concatenate((dl2_gamma[self.true_energy_key], dl2_proton[self.true_energy_key])) * u.TeV
        # plt.figure(figsize=(14,8))
        energy_bins = np.linspace(min(mc_gamma_energies), max(mc_gamma_energies), nbins+1)
        ctaplot.plot_roc_curve_gammaness_per_energy(mc_type, gammaness, mc_gamma_energies,
                                                        energy_bins=energy_bins, #u.Quantity([0.01,0.1,1,3,10], u.TeV),
                                                        linestyle='--',
                                                        alpha=1,
                                                        linewidth=2,
                                                        )
        plt.legend()
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.show()
        
    def compare_irfs_to_RF(self, zenith: float, azimuth=None):
        """
        Compare Instrument Response Functions (IRFs) to Random Forest (RF) benchmarks.
        This function compares the IRFs obtained from the CTLearn model to the RF benchmarks
        for a given zenith angle and optional azimuth angle. It plots the flux sensitivity,
        angular resolution, and energy resolution for both the CTLearn model and the RF benchmarks.

        Parameters
        ----------
        zenith : float
            The zenith angle in degrees.
        azimuth : float, optional
            The azimuth angle in degrees. If not provided, the default value is None.

        Returns
        -------
        None
            This function does not return any value. It generates and displays plots.
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
        
        with pkg_resources.path(RF_bechmpark, f'angular_resolution_{tel_string}.h5') as angular_resolution_file:
            angular_resolution_table = Table.read(angular_resolution_file, format='hdf5', path='res')
            angular_resolution_table_bins = Table.read(angular_resolution_file, format='hdf5', path='bins')
            
        with pkg_resources.path(RF_bechmpark, f'energy_resolution_{tel_string}.h5') as energy_resolution_file:
            energy_resolution_table = Table.read(energy_resolution_file, format='hdf5', path='res')
            energy_resolution_table_bins = Table.read(energy_resolution_file, format='hdf5', path='bins')
            
        with pkg_resources.path(RF_bechmpark, f'flux_sensitivity_{tel_string}.h5') as flux_sensitivity_file:
            flux_sensitivity_table = Table.read(flux_sensitivity_file, format='hdf5', path='sensitivity')
            
        irf_file = self.direction_model.get_IRF_data(zenith, azimuth)[3]
        hudl = fits.open(irf_file)

        energy_center = hudl['SENSITIVITY'].data['ENERG_LO'] + 0.5 * (hudl['SENSITIVITY'].data['ENERG_HI'] - hudl['SENSITIVITY'].data['ENERG_LO'])
        plt.plot(flux_sensitivity_table['energy'], flux_sensitivity_table['flux_sensitivity'], label='RF')
        plt.fill_between(flux_sensitivity_table['energy'], flux_sensitivity_table['flux_sensitivity']-flux_sensitivity_table['flux_sensitivity_err_minus'], flux_sensitivity_table['flux_sensitivity']+flux_sensitivity_table['flux_sensitivity_err_plus'], alpha=0.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
        plt.plot(energy_center[0], hudl['SENSITIVITY'].data['ENERGY_FLUX_SENSITIVITY'][0,0,:], label='CTLearn')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Energy [TeV]')
        plt.ylabel('Sensitivity [erg s$^{-1}$ cm$^{-2}$]')
        plt.legend()
        plt.show()

        energy_center = hudl['ANGULAR RESOLUTION '].data['ENERG_LO'] + 0.5 * (hudl['ANGULAR RESOLUTION '].data['ENERG_HI'] - hudl['ANGULAR RESOLUTION '].data['ENERG_LO'])
        energy_center_RF = angular_resolution_table_bins['energy_bins'][1:] - 0.5 * np.diff(angular_resolution_table_bins['energy_bins'])
        plt.plot(energy_center_RF, angular_resolution_table['angular_res'], label='RF 68%')
        plt.fill_between(energy_center_RF, angular_resolution_table['angular_res_err_lo'], angular_resolution_table['angular_res_err_hi'], alpha=0.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
        plt.plot(energy_center[0], hudl['ANGULAR RESOLUTION'].data['ANGULAR_RESOLUTION_68'][0,0,:], label='CTLearn 68%')
        plt.xscale('log')
        plt.xlabel('Energy [TeV]')
        plt.ylabel('Angular resolution [deg]')
        plt.legend()
        plt.show()
        plt.show()
        
        energy_center = hudl['ENERGY BIAS RESOLUTION'].data['ENERG_LO'] + 0.5 * (hudl['ENERGY BIAS RESOLUTION'].data['ENERG_HI'] - hudl['ENERGY BIAS RESOLUTION'].data['ENERG_LO'])
        energy_center_RF = energy_resolution_table_bins['energy_bins'][1:] - 0.5 * np.diff(energy_resolution_table_bins['energy_bins'])
        plt.plot(energy_center_RF, energy_resolution_table['energy_res'], label='RF')
        plt.fill_between(energy_center_RF, energy_resolution_table['energy_res_err_lo'], energy_resolution_table['energy_res_err_hi'], alpha=0.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
        plt.plot(energy_center[0], hudl['ENERGY BIAS RESOLUTION'].data['RESOLUTION'][0,0,:], label='CTLearn')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Energy [TeV]')
        plt.ylabel('Energy resolution')
        plt.legend()
        plt.show()
        
        hudl.close()

    def plot_everything_dl2(self, output_directory: str, dl2_files: list[str], gammaness_cut: float=0.9, edep_cuts: bool=False):
        """
        Plot the angular resolution, energy resolution, and gammaness for DL2 data.
        This function generates plots for the angular resolution, energy resolution,
        and gammaness for the given DL2 files. It uses ctaplot to create the plots
        and saves them in the specified output directory.

        Parameters
        ----------
        output_directory : str
            The directory where the plots will be saved.
        dl2_files : list[str]
            List of DL2 files to be processed.
        dl2_processed_dir : str
            The directory where the processed DL2 files are stored.
        gammaness_cut : float, optional
            The gammaness cut value to be applied. Default is 0.9.

        Returns
        -------
        None
        """
        import os
        import pickle
        tri_model_file = f"{output_directory}/tri_model.pkl"
        self.dl2_data_files = dl2_files

        use_cluster = self.cluster_configuration.use_cluster
        self.cluster_configuration.use_cluster = False # if some DL2 files were not processed, they will be processed in the same job as the plotting job, and not submit multiple new jobs

        with open(tri_model_file, 'wb') as f:
            pickle.dump(self, f)
        self.cluster_configuration.use_cluster = use_cluster

        cmd = f"plot_dl2 --stereo_tri_model {tri_model_file} --output_directory {output_directory} --gammaness_cut {gammaness_cut} --edep_cuts {edep_cuts}"
        
        sbatch_file = self.cluster_configuration.write_sbatch_script("dl2_plots", cmd, output_directory, use_gpu_cscs=False)
        os.system(f"sbatch {sbatch_file}")

    def plot_zenith_azimuth_ranges(self):
        self.direction_model.plot_zenith_azimuth_ranges()

    def apply_energy_dependent_cuts_MC(self, data, cuts_file, theta_cut=True):
        # Apply cuts to the data
        from astropy.io import fits
        from astropy.coordinates import SkyCoord
        with fits.open(cuts_file) as hdul:
            gammaness_cuts = hdul['GH_CUTS'].data['cut']
            energy_low_gamma = hdul['GH_CUTS'].data['low']
            energy_high_gamma = hdul['GH_CUTS'].data['high']
            theta_cuts = hdul['RAD_MAX'].data['cut']
            energy_low_theta = hdul['RAD_MAX'].data['low']
            energy_high_theta = hdul['RAD_MAX'].data['high']
            assert (energy_low_gamma == energy_low_theta).all(), "Energy low values for gammaness and theta cuts do not match"
            assert (energy_high_gamma == energy_high_theta).all(), "Energy high values for gammaness and theta cuts do not match"

            if theta_cut:
                true_coords = SkyCoord(alt=data[self.true_alt_key], az=data[self.true_az_key], frame='altaz', unit='deg')
                reco_coords = SkyCoord(alt=data[self.reco_alt_key], az=data[self.reco_az_key], frame='altaz', unit='deg')
                angular_separation = true_coords.separation(reco_coords).deg
                data['angular_separation'] = angular_separation


            masks = []
            for E_min, E_max, gcut, tcut in zip(energy_low_gamma, energy_high_gamma, gammaness_cuts, theta_cuts):
                energy_mask = (data[self.reco_energy_key] > E_min) & (data[self.reco_energy_key] < E_max)
                gammaness_mask = data[self.gammaness_key] > gcut
                if theta_cut:
                    theta_mask = data['angular_separation'] < tcut
                    mask = energy_mask & gammaness_mask & theta_mask
                else:
                    mask = energy_mask & gammaness_mask
                masks.append(mask)

            full_mask = np.zeros(len(data), dtype=bool)
            for mask in masks:
                full_mask |= mask
            dl2 = data[full_mask]
        return dl2
    
    


        