from astropy.table import QTable
import numpy as np
from pathlib import Path
from .model_manager import CTLearnModelManager
from .utils.utils import set_mpl_style, ClusterConfiguration
from .io.io import load_DL2_data_MC, load_true_shower_parameters

__all__ = [
    "CTLearnTriModelManager",
]



class CTLearnTriModelManager():
    """
    A manager class for handling three CTLearn models: direction, energy, and type.
    Attributes:
        direction_model (CTLearnModelManager): The direction model manager.
        energy_model (CTLearnModelManager): The energy model manager.
        type_model (CTLearnModelManager): The type model manager.
    Methods:
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
        if direction_model.model_parameters_table['reco'][0] == 'direction':
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
            
        # if not (self.direction_model.zd_range == self.energy_model.zd_range == self.type_model.zd_range):
        #     raise ValueError('All models must have the same zenith distance range')
        # if not (self.direction_model.az_range == self.energy_model.az_range == self.type_model.az_range):
        #     raise ValueError('All models must have the same azimuth range')
        
        if not (self.direction_model.stereo == self.energy_model.stereo == self.type_model.stereo):
            raise ValueError('All models must have the same stereo value')
        else:
            self.stereo = self.direction_model.stereo
        if not (self.direction_model.telescope_ids == self.energy_model.telescope_ids == self.type_model.telescope_ids):
            raise ValueError('All models must have the same telescope_ids')
        self.telescope_ids = self.direction_model.telescope_ids
        self.telescope_names = self.direction_model.telescope_names
        self.cluster_configuration = cluster_configuration
        self.reconstruction_method = "CTLearn"
        self.reco_field_suffix = self.reconstruction_method if self.stereo else f"{self.reconstruction_method}_tel"
        self.set_keys()
        print(f"🧠🧠🧠 CTLearnTriModelManager initialized with {self.direction_model.model_nickname}, {self.energy_model.model_nickname}, and {self.type_model.model_nickname}")
        self.get_available_MC_directions()

    def set_keys(self):
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
            
    def set_testing_directories(self, testing_gamma_dirs = [], testing_proton_dirs = [], testing_gamma_zenith_distances = [], testing_gamma_azimuths = [], testing_proton_zenith_distances = [], testing_proton_azimuths = [], testing_gamma_patterns = [], testing_proton_patterns = []):
        if not (len(testing_gamma_dirs) == len(testing_gamma_zenith_distances) == len(testing_gamma_azimuths) == len(testing_gamma_patterns)):
            raise ValueError("All testing gamma lists must be the same length")
        if not (len(testing_proton_dirs) == len(testing_proton_zenith_distances) == len(testing_proton_azimuths) == len(testing_proton_patterns)):
            raise ValueError("All testing proton lists must be the same length")
                
        for model in [self.direction_model, self.energy_model, self.type_model]:
            model.update_model_manager_testing_data(
                testing_gamma_dirs, 
                testing_proton_dirs, 
                testing_gamma_zenith_distances, 
                testing_gamma_azimuths, 
                testing_proton_zenith_distances, 
                testing_proton_azimuths,
                testing_gamma_patterns,
                testing_proton_patterns
            )
    
    def set_DL2_MC_files(self, testing_DL2_gamma_files, testing_DL2_proton_files, testing_DL2_gamma_zenith_distances, testing_DL2_gamma_azimuths, testing_DL2_proton_zenith_distances, testing_DL2_proton_azimuths):
        for model in [self.direction_model, self.energy_model, self.type_model]:
            model.update_model_manager_DL2_MC_files(
                testing_DL2_gamma_files, 
                testing_DL2_proton_files, 
                testing_DL2_gamma_zenith_distances, 
                testing_DL2_gamma_azimuths, 
                testing_DL2_proton_zenith_distances, 
                testing_DL2_proton_azimuths
            )

    def get_available_testing_directions(self):
        from astropy.io.misc.hdf5 import read_table_hdf5
        direction_testing_table =  read_table_hdf5(self.direction_model.model_index_file, path=f'{self.direction_model.model_nickname}/testing/gamma')
        gamma_zeniths = direction_testing_table['testing_gamma_zenith_distances']
        gamma_azimuths = direction_testing_table['testing_gamma_azimuths']
        for zenith, azimuth in zip(gamma_zeniths, gamma_azimuths):
            print(f"(ZD, Az): ({zenith}, {azimuth})")

    def get_available_MC_directions(self):
        from astropy.io.misc.hdf5 import read_table_hdf5
        
        try:
            DL2_gamma_table = read_table_hdf5(self.direction_model.model_index_file, path=f'{self.direction_model.model_nickname}/DL2/MC/gamma')
            gamma_zeniths = DL2_gamma_table['testing_DL2_gamma_zenith_distances']
            gamma_azimuths = DL2_gamma_table['testing_DL2_gamma_azimuths']
        except:
            gamma_zeniths = []
            gamma_azimuths = []

        try:
            DL2_proton_table = read_table_hdf5(self.direction_model.model_index_file, path=f'{self.direction_model.model_nickname}/DL2/MC/proton')
            proton_zeniths = DL2_proton_table['testing_DL2_proton_zenith_distances']
            proton_azimuths = DL2_proton_table['testing_DL2_proton_azimuths']
        except:
            proton_zeniths = []
            proton_azimuths = []

        coords = set(zip(gamma_zeniths, gamma_azimuths)).union(set(zip(proton_zeniths, proton_azimuths)))
        if len(coords) > 0:
            print("Available MC DL2 directions:")
        for zenith, azimuth in coords:
            gamma_available = (zenith, azimuth) in set(zip(gamma_zeniths, gamma_azimuths))
            proton_available = (zenith, azimuth) in set(zip(proton_zeniths, proton_azimuths))
            if gamma_available and proton_available:
                print(f"(ZD, Az): ({zenith}, {azimuth}) \t gamma | proton")
            elif gamma_available:
                print(f"(ZD, Az): ({zenith}, {azimuth}) \t gamma |")
            elif proton_available:
                print(f"(ZD, Az): ({zenith}, {azimuth}) \t       | proton")
            else:
                print(f"(ZD, Az): ({zenith}, {azimuth})")
        
    def launch_testing(self, zenith, azimuth, output_dirs: list, config_dir=None, launch_particle_type='both', ):
        import os
        import glob
        from astropy.io.misc.hdf5 import read_table_hdf5
        # Check that the testing files are the same for each model
        gamma_dir = []
        proton_dir = []
        if launch_particle_type not in ['gamma', 'proton', 'both']:
            raise ValueError("launch_particle_type must be 'gamma', 'proton', or 'both'")
        if launch_particle_type in ['gamma', 'both']:
            direction_testing_table =  read_table_hdf5(self.direction_model.model_index_file, path=f'{self.direction_model.model_nickname}/testing/gamma')
            energy_testing_table =  read_table_hdf5(self.energy_model.model_index_file, path=f'{self.energy_model.model_nickname}/testing/gamma')
            type_testing_table =  read_table_hdf5(self.type_model.model_index_file, path=f'{self.type_model.model_nickname}/testing/gamma')
            if not (direction_testing_table['testing_gamma_dirs'] == energy_testing_table['testing_gamma_dirs'] == type_testing_table['testing_gamma_dirs']):
                raise ValueError("All models must have the same testing gamma directories, use set_testing_files to set them")
            if not direction_testing_table['testing_gamma_dirs'] or not energy_testing_table['testing_gamma_dirs'] or not type_testing_table['testing_gamma_dirs']:
                raise ValueError("Testing gamma directories cannot be empty")
            gamma_dirs = direction_testing_table['testing_gamma_dirs']
            gamma_zeniths = direction_testing_table['testing_gamma_zenith_distances']
            gamma_azimuths = direction_testing_table['testing_gamma_azimuths']
            gamma_patterns = direction_testing_table['testing_gamma_patterns']
            matching_dirs = [gamma_dirs[i] for i in range(len(gamma_dirs)) if gamma_zeniths[i] == zenith and gamma_azimuths[i] == azimuth]
            if not matching_dirs:
                raise ValueError(f"No matching gamma directory found for zenith {zenith} and azimuth {azimuth}")
            gamma_dir = matching_dirs[0]
            gamma_pattern = [gamma_patterns[i] for i in range(len(gamma_patterns)) if gamma_zeniths[i] == zenith and gamma_azimuths[i] == azimuth][0]
        if launch_particle_type in ['proton', 'both']:
            direction_testing_table =  read_table_hdf5(self.direction_model.model_index_file, path=f'{self.direction_model.model_nickname}/testing/proton')
            energy_testing_table =  read_table_hdf5(self.energy_model.model_index_file, path=f'{self.energy_model.model_nickname}/testing/proton')
            type_testing_table =  read_table_hdf5(self.type_model.model_index_file, path=f'{self.type_model.model_nickname}/testing/proton')
            if not (direction_testing_table['testing_proton_dirs'] == energy_testing_table['testing_proton_dirs'] == type_testing_table['testing_proton_dirs']):
                raise ValueError("All models must have the same testing proton directories, use set_testing_files to set them")
            if not direction_testing_table['testing_proton_dirs'] or not energy_testing_table['testing_proton_dirs'] or not type_testing_table['testing_proton_dirs']:
                raise ValueError("Testing proton directories cannot be empty")
            proton_dirs = direction_testing_table['testing_proton_dirs']
            proton_zeniths = direction_testing_table['testing_proton_zenith_distances']
            proton_azimuths = direction_testing_table['testing_proton_azimuths']
            proton_patterns = direction_testing_table['testing_proton_patterns']
            matching_dirs = [proton_dirs[i] for i in range(len(proton_dirs)) if proton_zeniths[i] == zenith and proton_azimuths[i] == azimuth]
            if not matching_dirs:
                raise ValueError(f"No matching proton directory found for zenith {zenith} and azimuth {azimuth}")
            proton_dir = matching_dirs[0] 
            proton_pattern = [proton_patterns[i] for i in range(len(proton_patterns)) if proton_zeniths[i] == zenith and proton_azimuths[i] == azimuth][0]
            
        if len(output_dirs) == 1:
            gamma_output_dir = output_dirs[0]
            proton_output_dir = output_dirs[0]
        elif len(output_dirs) == 2:
            gamma_output_dir = output_dirs[0]
            proton_output_dir = output_dirs[1]
        else:
            raise ValueError("output_dirs must have length 1 or 2, to store all in the same directory, or gammas in the first and protons in the second")
        # gamma_files = np.sort(glob.glob(f"{gamma_dir}/{gamma_pattern}"))
        # proton_files = np.sort(glob.glob(f"{proton_dir}/{proton_pattern}"))
        # testing_files = np.concatenate([gamma_files, proton_files])
        # gamma_output_files = [f"{output_dir}/{Path(file).stem.replace('dl1', 'dl2')}.h5" for file in gamma_files]
        # proton_output_files = [f"{output_dir}/{Path(file).stem.replace('dl1', 'dl2')}.h5" for file in proton_files]
        # output_files = [f"{output_dir}/{Path(file).stem.replace('dl1', 'dl2')}.h5" for file in testing_files]
        # for model in [self.direction_model, self.energy_model, self.type_model]:
        #     model.update_model_manager_DL2_MC_files(
        #         gamma_output_files, 
        #         proton_output_files, 
        #         [zenith] * len(gamma_output_files), 
        #         [azimuth] * len(gamma_output_files), 
        #         [zenith] * len(proton_output_files), 
        #         [azimuth] * len(proton_output_files)
        #     )
        if launch_particle_type in ['gamma', 'both']:
            gamma_files = np.sort(glob.glob(f"{gamma_dir}/{gamma_pattern}"))
            gamma_output_files = [f"{gamma_output_dir}/{Path(file).stem.replace('dl1', 'dl2')}.h5" for file in gamma_files]
        else:
            gamma_files = []
            gamma_output_files = []
        if launch_particle_type in ['proton', 'both']:
            proton_files = np.sort(glob.glob(f"{proton_dir}/{proton_pattern}"))
            proton_output_files = [f"{proton_output_dir}/{Path(file).stem.replace('dl1', 'dl2')}.h5" for file in proton_files]
        else:
            proton_files = []
            proton_output_files = []
        testing_files = np.concatenate([gamma_files, proton_files])
        output_files = np.concatenate([gamma_output_files, proton_output_files])
        for model in [self.direction_model, self.energy_model, self.type_model]:
            model.update_model_manager_DL2_MC_files(
                gamma_output_files, 
                proton_output_files, 
                [zenith] * len(gamma_output_files), 
                [azimuth] * len(gamma_output_files), 
                [zenith] * len(proton_output_files), 
                [azimuth] * len(proton_output_files)
            )
                
        
        channels_string = ""
        for channel in self.channels:
            channels_string += f"--DLImageReader.channels={channel} "
        type_model_dir = np.sort(glob.glob(f"{self.type_model.model_parameters_table['model_dir'][0]}/{self.type_model.model_nickname}_v*"))[-1]
        energy_model_dir = np.sort(glob.glob(f"{self.energy_model.model_parameters_table['model_dir'][0]}/{self.energy_model.model_nickname}_v*"))[-1]
        direction_model_dir = np.sort(glob.glob(f"{self.direction_model.model_parameters_table['model_dir'][0]}/{self.direction_model.model_nickname}_v*"))[-1]
        
            
        for input_file, output_file in zip(testing_files, output_files):
            if self.stereo:
                cmd = f"ctlearn-predict-model --input_url {input_file} \
--type_model={type_model_dir}/ctlearn_model.cpk \
--energy_model={energy_model_dir}/ctlearn_model.cpk \
--direction_model={direction_model_dir}/ctlearn_model.cpk \
--no-dl1-images --no-true-images --output {output_file} \
--DLImageReader.mode=stereo --PredictCTLearnModel.stack_telescope_images=True --DLImageReader.min_telescopes=2 \
--PredictCTLearnModel.overwrite_tables=True -v {channels_string}"
            else:
                # cmd = f"ctlearn-predict-mono --input_url {input_file} --type_model={type_model_dir}/ctlearn_model.cpk --energy_model={energy_model_dir}/ctlearn_model.cpk --direction_model={direction_model_dir}/ctlearn_model.cpk --no-dl1-images --no-true-images --output {output_file} --overwrite -v {channels_string}"
                cmd = f"ctlearn-predict-model --input_url {input_file} \
--type_model={type_model_dir}/ctlearn_model.cpk \
--energy_model={energy_model_dir}/ctlearn_model.cpk \
--direction_model={direction_model_dir}/ctlearn_model.cpk \
--no-dl1-images --no-true-images --output {output_file} \
--PredictCTLearnModel.overwrite_tables=True -v {channels_string}"
            
            if self.cluster_configuration.use_cluster:
                # sbatch_file = write_sbatch_script(cluster_configuration.cluster, Path(input_file).stem, cmd, config_dir, env_name=cluster_configuration.python_env, account=cluster_configuration.account)
                sbatch_file = self.cluster_configuration.write_sbatch_script(Path(input_file).stem, cmd, config_dir)
                os.system(f"sbatch {sbatch_file}")  
            else:
                print(cmd)
                os.system(cmd)
        
    
    def predict_lstchain_data(self, input_file, output_file, run=None, subrun=None, config_dir=None, overwrite=False, pointing_table='/dl1/event/telescope/parameters/LST_LSTCam'):
        import os
        import glob
        import ast
        import json
        import yaml
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
        print(f"🪛 Configuration saved to {config_file}")

        avg_data_ze, avg_data_az = get_avg_pointing(input_file, pointing_table=pointing_table)
        for model in [self.direction_model, self.energy_model, self.type_model]:
            model.update_model_manager_DL2_data_files(
                [output_file], 
                [avg_data_ze],
                [avg_data_az],
            )
        
        cmd = f"ctlearn-predict-LST1 --input_url {input_file} \
--type_model {type_model_dir}/ctlearn_model.cpk \
--energy_model {energy_model_dir}/ctlearn_model.cpk \
--direction_model {direction_model_dir}/ctlearn_model.cpk \
--config '{config_file}' \
-v"
            
        if self.cluster_configuration.use_cluster:
            # sbatch_file = write_sbatch_script(cluster_configuration.cluster, Path(input_file).stem, cmd, config_dir, cluster_configuration.python_env, cluster_configuration.account)
            sbatch_file = self.cluster_configuration.write_sbatch_script(Path(input_file).stem, cmd, config_dir)
            import os
            os.system(f"sbatch {sbatch_file}")
    
        else:
            print(cmd)
            os.system(cmd)
     

        print("")
        
    
    def predict_data(self, input_file, output_file, config_dir=None, overwrite=False, pointing_table='dl0/monitoring/subarray/pointing'):
        import os
        import glob
        import ast
        import json
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
        print(f"🪛 Configuration saved to {config_file}")

        avg_data_ze, avg_data_az = get_avg_pointing(input_file, pointing_table=pointing_table)
        for model in [self.direction_model, self.energy_model, self.type_model]:
            model.update_model_manager_DL2_data_files(
                [output_file], 
                [avg_data_ze],
                [avg_data_az],
            )
        
        cmd = f"ctlearn-predict-model --input_url {input_file} \
--type_model {type_model_dir}/ctlearn_model.cpk \
--energy_model {energy_model_dir}/ctlearn_model.cpk \
--direction_model {direction_model_dir}/ctlearn_model.cpk \
--config '{config_file}' \
--no-dl1-images --no-true-images \
--dl1-features \
--PredictCTLearnModel.overwrite_tables True -v"
            
        if self.cluster_configuration.use_cluster:
            # sbatch_file = write_sbatch_script(cluster_configuration.cluster, Path(input_file).stem, cmd, config_dir, cluster_configuration.python_env, cluster_configuration.account)
            sbatch_file = self.cluster_configuration.write_sbatch_script(Path(input_file).stem, cmd, config_dir)
            os.system(f"sbatch {sbatch_file}")
        else:
            print(cmd)
            os.system(cmd)
            # os.system(cmd)

        print("")
    
    
    def merge_DL2_files(self, zenith, azimuth, output_file_gammas=None, output_file_protons=None, overwrite=False):
        import glob
        import os
        gamma_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[0]
        proton_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[1]
        if len(gamma_files) > 1 and output_file_gammas is not None:
            print(f"🔀 Merging DL2 gamma files for zenith {zenith} and azimuth {azimuth}")
            result = os.system(f"ctapipe-merge {' '.join(gamma_files)} --output={output_file_gammas} --progress --MergeTool.skip_broken_files=True {'--overwrite' if overwrite else ''}")
            if result == 0:
                self.direction_model.update_merged_DL2_MC_files(zenith, azimuth, output_file_gammas, None)
                self.energy_model.update_merged_DL2_MC_files(zenith, azimuth, output_file_gammas, None)
                self.type_model.update_merged_DL2_MC_files(zenith, azimuth, output_file_gammas, None)
            else:
                print(f"Error: Failed to merge gamma files for zenith {zenith} and azimuth {azimuth}")
        else:
            print(f"✅ There already is a single gamma file for zenith {zenith} and azimuth {azimuth}")
        if len(proton_files) > 1 and output_file_protons is not None:
            print(f"🔀 Merging DL2 proton files for zenith {zenith} and azimuth {azimuth}")
            result = os.system(f"ctapipe-merge {' '.join(proton_files)} --output={output_file_protons} --progress --MergeTool.skip_broken_files=True {'--overwrite' if overwrite else ''}")
            if result == 0:
                self.direction_model.update_merged_DL2_MC_files(zenith, azimuth, None, output_file_protons)
                self.energy_model.update_merged_DL2_MC_files(zenith, azimuth, None, output_file_protons)
                self.type_model.update_merged_DL2_MC_files(zenith, azimuth, None, output_file_protons)
            else:
                print(f"Error: Failed to merge proton files for zenith {zenith} and azimuth {azimuth}")
        else:
            print(f"✅ There already is a single proton file for zenith {zenith} and azimuth {azimuth}")
    
    def plot_DL2_classification(self, zenith, azimuth):
        import matplotlib.pyplot as plt
        from astropy.table import vstack
        set_mpl_style()
        testing_DL2_gamma_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[0]
        testing_DL2_proton_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[1]
        dl2_gamma = []
        for file in testing_DL2_gamma_files:
            dl2_gamma.append(load_DL2_data_MC(file))
        dl2_gamma = vstack(dl2_gamma)
        
        dl2_protons = []
        for file in testing_DL2_proton_files:
            dl2_protons.append(load_DL2_data_MC(file))
        dl2_proton = vstack(dl2_protons)
        plt.hist(dl2_gamma["CTLearn_prediction"], bins=100, range=(0, 1), histtype="step", density=True, lw=2, label="Gammas")
        plt.hist(dl2_proton["CTLearn_prediction"], bins=100, range=(0, 1), histtype="step", density=True, lw=2, label="Protons")
        plt.xlabel("Gammaness")
        plt.ylabel("Density")
        plt.legend()
        plt.show()
        
    def plot_DL2_energy(self, zenith, azimuth):
        import matplotlib.pyplot as plt
        from astropy.table import vstack
        set_mpl_style()
        testing_DL2_gamma_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[0]
        testing_DL2_proton_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[1]
        dl2_gamma = []
        for file in testing_DL2_gamma_files:
            dl2_gamma.append(load_DL2_data_MC(file))
        dl2_gamma = vstack(dl2_gamma)
        
        dl2_protons = []
        for file in testing_DL2_proton_files:
            dl2_protons.append(load_DL2_data_MC(file))
        dl2_proton = vstack(dl2_protons)
        log_bins = np.logspace(np.log10(0.1), np.log10(500), 100)
        plt.hist(dl2_gamma["CTLearn_energy"], bins=log_bins, range=(0, 1), histtype="step", density=True, lw=2, label="Gammas")
        plt.hist(dl2_proton["CTLearn_energy"], bins=log_bins, range=(0, 1), histtype="step", density=True, lw=2, label="Protons")
        plt.xlabel("Energy [TeV]")
        plt.ylabel("Density")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.show()
        
    def plot_DL2_AltAz(self, zenith, azimuth):
        import matplotlib.pyplot as plt
        from astropy.table import vstack
        set_mpl_style()
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        testing_DL2_gamma_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[0]
        testing_DL2_proton_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[1]

        tel_id = None if self.stereo else self.telescope_ids[0]

        if len(testing_DL2_gamma_files) > 0:
            dl2_gamma = []
            for file in testing_DL2_gamma_files:
                dl2_gamma.append(load_DL2_data_MC(file, tel_id=tel_id))
            dl2_gamma = vstack(dl2_gamma)

            axs[0].scatter(dl2_gamma[self.pointing_alt_key][0]/np.pi*180, dl2_gamma[self.pointing_az_key][0]/np.pi*180, color="red", label="Array pointing", marker="x", s=100)
            axs[0].hist2d(dl2_gamma[self.reco_alt_key], dl2_gamma[self.reco_az_key], bins=100, zorder=0, cmap="viridis", norm=plt.cm.colors.LogNorm())
            axs[0].set_xlabel("Altitude [deg]")
            axs[0].set_ylabel("Azimuth [deg]")
            axs[0].legend()
            axs[0].set_title("Gammas")
            cbar = plt.colorbar(axs[0].collections[0], ax=axs[0])
            cbar.set_label("Counts")
        
        if len(testing_DL2_proton_files) > 0:
            dl2_protons = []
            for file in testing_DL2_proton_files:
                dl2_protons.append(load_DL2_data_MC(file, tel_id=tel_id))
            dl2_proton = vstack(dl2_protons)
            
            
            
            axs[1].scatter(dl2_proton[self.pointing_alt_key][0]/np.pi*180, dl2_proton[self.pointing_az_key][0]/np.pi*180, color="red", label="Array pointing", marker="x", s=100)
            axs[1].hist2d(dl2_proton[self.reco_alt_key], dl2_proton[self.reco_az_key], bins=100, zorder=0, cmap="viridis", norm=plt.cm.colors.LogNorm())
            axs[1].set_xlabel("Altitude [deg]")
            axs[1].set_ylabel("Azimuth [deg]")
            axs[1].legend()
            axs[1].set_title("Protons")
            cbar = plt.colorbar(axs[1].collections[0], ax=axs[1])
            cbar.set_label("Counts")
        
        plt.tight_layout()
        plt.show()
        
    def plot_migration_matrix(self, zenith, azimuth):      
        import matplotlib.pyplot as plt
        from astropy.table import vstack, join
        set_mpl_style()
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        testing_DL2_gamma_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[0]
        testing_DL2_proton_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[1]
        dl2_gamma = []
        shower_parameters_gamma = []
        for file in testing_DL2_gamma_files:
            dl2_gamma.append(load_DL2_data_MC(file))
            shower_parameters_gamma.append(load_true_shower_parameters(file))
        dl2_gamma = vstack(dl2_gamma)
        shower_parameters_gamma = vstack(shower_parameters_gamma)
        dl2_gamma = join(dl2_gamma, shower_parameters_gamma, keys=["obs_id", "event_id"])
        
        dl2_protons = []
        shower_parameters_protons = []
        for file in testing_DL2_proton_files:
            dl2_protons.append(load_DL2_data_MC(file))
            shower_parameters_protons.append(load_true_shower_parameters(file))
        dl2_proton = vstack(dl2_protons)
        shower_parameters_protons = vstack(shower_parameters_protons)
        dl2_proton = join(dl2_proton, shower_parameters_protons, keys=["obs_id", "event_id"])
        
        log_bins = np.logspace(np.log10(0.1), np.log10(500), 100)
        
        axs[0].plot([0.1, 500], [0.1, 500], color="red", ls="--")
        axs[0].hist2d(dl2_gamma["CTLearn_energy"], dl2_gamma["true_energy"], bins=log_bins, cmap="viridis", norm=plt.cm.colors.LogNorm())
        axs[0].set_xlabel("CTLean Energy [TeV]")
        axs[0].set_ylabel("True Energy [TeV]")
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        axs[0].axis('equal')
        axs[0].set_title("Gammas")
        cbar = plt.colorbar(axs[0].collections[0], ax=axs[0])
        cbar.set_label("Counts")
        
        axs[1].plot([0.1, 500], [0.1, 500], color="red", ls="--")
        axs[1].hist2d(dl2_proton["CTLearn_energy"], dl2_proton["true_energy"], bins=log_bins, cmap="viridis", norm=plt.cm.colors.LogNorm())
        axs[1].set_xlabel("CTLearn Energy [TeV]")
        axs[1].set_ylabel("True Energy [TeV]")
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        axs[1].axis('equal')
        axs[1].set_title("Protons")
        cbar = plt.colorbar(axs[1].collections[0], ax=axs[1])
        cbar.set_label("Counts")
        
        plt.tight_layout()
        plt.show()
        
    def produce_irfs(self, zenith, azimuth, config=None, output_cuts_file=None, output_irf_file=None, output_benchmark_file=None):
        import os
        if config is None:
            try:
                config = self.direction_model.get_IRF_data(zenith, azimuth)[0]
            except:
                raise ValueError("A configuration file must be provided, at least the first time.")
        if output_cuts_file is None:
            try:
                output_cuts_file = self.direction_model.get_IRF_data(zenith, azimuth)[1]
            except:
                raise ValueError("An output cuts file must be provided, at least the first time.")
        if output_irf_file is None:
            try:
                output_irf_file = self.direction_model.get_IRF_data(zenith, azimuth)[2]
            except:
                raise ValueError("An output IRF file must be provided, at least the first time.")
        if output_benchmark_file is None:
            try:
                output_benchmark_file = self.direction_model.get_IRF_data(zenith, azimuth)[3]
            except:
                raise ValueError("An output benchmark file must be provided, at least the first time.")
        
        self.direction_model.update_model_manager_IRF_data(config, output_cuts_file, output_irf_file, output_benchmark_file, zenith, azimuth)
        self.energy_model.update_model_manager_IRF_data(config, output_cuts_file, output_irf_file, output_benchmark_file, zenith, azimuth)
        self.type_model.update_model_manager_IRF_data(config, output_cuts_file, output_irf_file, output_benchmark_file, zenith, azimuth)
            
        gamma_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[0]
        proton_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[1]
        if len(gamma_files) > 1 or len(proton_files) > 1:
            raise ValueError(f"Multiple files found for zenith {zenith} and azimuth {azimuth}, please merge them first with CTLearnTriModelManager.merge_DL2_files()")
        gamma_file = gamma_files[0]
        proton_file = proton_files[0]
        cmd = f"ctapipe-optimize-event-selection \
-c {config} \
--gamma-file {gamma_file} \
--proton-file {proton_file} \
--output {output_cuts_file} \
--overwrite True"
            # --EventSelectionOptimizer.optimization_algorithm=PercentileCuts"
        os.system(cmd)
        cmd = f"ctapipe-compute-irf \
-c {config} --IrfTool.cuts_file {output_cuts_file} \
--gamma-file {gamma_file} \
--proton-file {proton_file}  \
--do-background \
--output {output_irf_file} \
--benchmark-output {output_benchmark_file} \
--no-spatial-selection-applied --overwrite"
        os.system(cmd)
    
    def plot_benchmark(self, zenith, azimuth):
        set_mpl_style()
        from astropy.io import fits
        import matplotlib.pyplot as plt
        irf_file = self.direction_model.get_IRF_data(zenith, azimuth)[3]
        hudl = fits.open(irf_file)
        # energy_center = hudl['SENSITIVITY'].data['ENERG_LO'] + 0.5 * (hudl['SENSITIVITY'].data['ENERG_HI'] - hudl['SENSITIVITY'].data['ENERG_LO'])
        # plt.plot(energy_center[0], hudl['SENSITIVITY'].data['FLUX_SENSITIVITY'][0,0,:])
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('Energy [TeV]')
        # plt.ylabel('Sensitivity [erg$^{-1}$ s$^{-1}$ cm$^{-2}$]')
        # plt.show()
        
        energy_center = hudl['SENSITIVITY'].data['ENERG_LO'] + 0.5 * (hudl['SENSITIVITY'].data['ENERG_HI'] - hudl['SENSITIVITY'].data['ENERG_LO'])
        plt.plot(energy_center[0], hudl['SENSITIVITY'].data['ENERGY_FLUX_SENSITIVITY'][0,0,:])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Energy [TeV]')
        plt.ylabel('Sensitivity [erg s$^{-1}$ cm$^{-2}$]')
        plt.show()
        
        energy_center = hudl['ANGULAR RESOLUTION '].data['ENERG_LO'] + 0.5 * (hudl['ANGULAR RESOLUTION '].data['ENERG_HI'] - hudl['ANGULAR RESOLUTION '].data['ENERG_LO'])
        plt.plot(energy_center[0], hudl['ANGULAR RESOLUTION'].data['ANGULAR_RESOLUTION_25'][0,0,:], label='25%')
        plt.plot(energy_center[0], hudl['ANGULAR RESOLUTION'].data['ANGULAR_RESOLUTION_50'][0,0,:], label='50%')
        plt.plot(energy_center[0], hudl['ANGULAR RESOLUTION'].data['ANGULAR_RESOLUTION_68'][0,0,:], label='68%')
        plt.plot(energy_center[0], hudl['ANGULAR RESOLUTION'].data['ANGULAR_RESOLUTION_95'][0,0,:], label='95%')
        plt.xscale('log')
        plt.xlabel('Energy [TeV]')
        plt.ylabel('Angular resolution [deg]')
        plt.legend()
        plt.show()
        
        energy_center = hudl['ENERGY BIAS RESOLUTION'].data['ENERG_LO'] + 0.5 * (hudl['ENERGY BIAS RESOLUTION'].data['ENERG_HI'] - hudl['ENERGY BIAS RESOLUTION'].data['ENERG_LO'])
        plt.plot(energy_center[0], hudl['ENERGY BIAS RESOLUTION'].data['RESOLUTION'][0,0,:])
        plt.xscale('log')
        plt.xlabel('Energy [TeV]')
        plt.ylabel('Energy resolution')
        plt.show()
        
        energy_center = hudl['ENERGY BIAS RESOLUTION'].data['ENERG_LO'] + 0.5 * (hudl['ENERGY BIAS RESOLUTION'].data['ENERG_HI'] - hudl['ENERGY BIAS RESOLUTION'].data['ENERG_LO'])
        plt.plot(energy_center[0], hudl['ENERGY BIAS RESOLUTION'].data['BIAS'][0,0,:])
        plt.xscale('log')
        plt.xlabel('Energy [TeV]')
        plt.ylabel('Energy bias')
        plt.show()
        hudl.close() 
        
    def plot_irfs(self, zenith, azimuth):
        set_mpl_style()
        from astropy.io import fits
        import matplotlib.pyplot as plt
        from gammapy.irf import EnergyDispersion2D, EffectiveAreaTable2D, Background2D, RadMax2D
        irf_file = self.direction_model.get_IRF_data(zenith, azimuth)[2]
        # rad_max = RadMax2D.read(irf_file, hdu="RAD MAX")
        aeff = EffectiveAreaTable2D.read(irf_file, hdu="EFFECTIVE AREA")
        bkg = Background2D.read(irf_file, hdu="BACKGROUND")
        edisp = EnergyDispersion2D.read(irf_file, hdu="ENERGY DISPERSION")
        edisp.peek()
        aeff.peek()
        bkg.peek()
        
    def plot_loss(self):
        set_mpl_style()
        import matplotlib.pyplot as plt
        import pandas as pd
        import glob
        
        
        # direction_training_log = np.sort(glob.glob(f"{self.direction_model.model_parameters_table['model_dir'][0]}/{self.direction_model.model_nickname}_v*/training_log.csv"))[-1]
        # energy_training_log = np.sort(glob.glob(f"{self.energy_model.model_parameters_table['model_dir'][0]}/{self.energy_model.model_nickname}_v*/training_log.csv"))[-1]
        # type_training_log = np.sort(glob.glob(f"{self.type_model.model_parameters_table['model_dir'][0]}/{self.type_model.model_nickname}_v*/training_log.csv"))[-1]
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, model in zip(axs, [self.direction_model, self.energy_model, self.type_model]):
            training_logs = np.sort(glob.glob(f"{model.model_parameters_table['model_dir'][0]}/{model.model_nickname}_v*/training_log.csv"))
            losses_train = []
            losses_val = []
            for training_log in training_logs:
                df = pd.read_csv(training_log)
                losses_train = np.concatenate((losses_train, df['loss'].to_numpy()))
                losses_val = np.concatenate((losses_val, df['val_loss'].to_numpy()))
            epochs = np.arange(1, len(losses_train)+1)
            df = pd.read_csv(training_log)
            ax.plot(epochs, losses_train, label=f"Training", lw=2)
            ax.plot(epochs, losses_val, label=f"Validation", ls='--')
            # ax.plot(df['epoch'] + 1, df['loss'], label=f"Training")
            # ax.plot(df['epoch'] + 1, df['val_loss'], label=f"Validation", ls='--')
            ax.set_title(f"{model.model_parameters_table['reco'][0]} training".title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_xticks(np.arange(1, len(epochs) + 1, 2))
            ax.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_angular_resolution_DL2(self, zenith, azimuth):
        set_mpl_style()
        import ctaplot
        import matplotlib.pyplot as plt
        from astropy.table import vstack, join
        import astropy.units as u
        from astropy.io.misc.hdf5 import read_table_hdf5
        DL2_gamma_table = read_table_hdf5(self.direction_model.model_index_file, path=f'{self.direction_model.model_nickname}/DL2/MC/gamma')
        testing_DL2_gamma_files = DL2_gamma_table['testing_DL2_gamma_files'][DL2_gamma_table['testing_DL2_gamma_zenith_distances'] == zenith][DL2_gamma_table['testing_DL2_gamma_azimuths'] == azimuth]
        dl2_gamma = []
        shower_parameters_gamma = []
        tel_id = None if self.stereo else self.telescope_ids[0]
        for file in testing_DL2_gamma_files:
            dl2_gamma.append(load_DL2_data_MC(file, tel_id=tel_id))
            shower_parameters_gamma.append(load_true_shower_parameters(file))
        dl2_gamma = vstack(dl2_gamma)
        shower_parameters_gamma = vstack(shower_parameters_gamma)
        dl2_gamma = join(dl2_gamma, shower_parameters_gamma, keys=["obs_id", "event_id"])

        reco_alt = dl2_gamma[self.reco_alt_key].to(u.deg)
        reco_az = dl2_gamma[self.reco_az_key].to(u.deg)
        true_alt = dl2_gamma[self.true_alt_key].to(u.deg)
        true_az = dl2_gamma[self.true_az_key].to(u.deg)
        reco_energy = dl2_gamma[self.reco_energy_key]
        true_energy = dl2_gamma[self.true_energy_key]
        
        # Define the range of true energy values
        true_energy_min = np.min(true_energy)
        true_energy_max = np.max(true_energy)

        # Create bins with 5 bins per decade in log scale
        bins_per_decade = 5
        log_bins = np.logspace(np.log10(true_energy_min), np.log10(true_energy_max), 
                               num=int(np.log10(true_energy_max/true_energy_min) * bins_per_decade) + 1) * u.TeV

        ctaplot.plot_angular_resolution_per_energy(true_alt, reco_alt, true_az, reco_az, true_energy, bins=log_bins, label=f"Gammas {zenith} {azimuth}")
        plt.legend()
        plt.grid(False, which='both')
        plt.show()
        
    def plot_energy_resolution_DL2(self, zenith, azimuth):
        set_mpl_style()
        import ctaplot
        import matplotlib.pyplot as plt
        from astropy.table import vstack, join
        import astropy.units as u
        from astropy.io.misc.hdf5 import read_table_hdf5
        DL2_gamma_table = read_table_hdf5(self.direction_model.model_index_file, path=f'{self.direction_model.model_nickname}/DL2/MC/gamma')
        testing_DL2_gamma_files = DL2_gamma_table['testing_DL2_gamma_files'][DL2_gamma_table['testing_DL2_gamma_zenith_distances'] == zenith][DL2_gamma_table['testing_DL2_gamma_azimuths'] == azimuth]
        dl2_gamma = []
        shower_parameters_gamma = []
        tel_id = None if self.stereo else self.telescope_ids[0]
        for file in testing_DL2_gamma_files:
            dl2_gamma.append(load_DL2_data_MC(file, tel_id))
            shower_parameters_gamma.append(load_true_shower_parameters(file))
        dl2_gamma = vstack(dl2_gamma)
        shower_parameters_gamma = vstack(shower_parameters_gamma)
        dl2_gamma = join(dl2_gamma, shower_parameters_gamma, keys=["obs_id", "event_id"])

        reco_energy = dl2_gamma[self.reco_energy_key]
        true_energy = dl2_gamma[self.true_energy_key]
        
        # Define the range of true energy values
        true_energy_min = np.min(true_energy)
        true_energy_max = np.max(true_energy)

        # Create bins with 5 bins per decade in log scale
        bins_per_decade = 5
        log_bins = np.logspace(np.log10(true_energy_min), np.log10(true_energy_max), 
                               num=int(np.log10(true_energy_max/true_energy_min) * bins_per_decade) + 1) * u.TeV
        
        ctaplot.plot_energy_resolution(true_energy, reco_energy, bins=log_bins, label=f"Gammas {zenith} {azimuth}")
        plt.legend()
        plt.grid(False, which='both')
        plt.show()
        

    def plot_ROC_curve_DL2(self, zenith, azimuth):
        set_mpl_style()
        import ctaplot
        import matplotlib.pyplot as plt
        from astropy.table import vstack, join
        import astropy.units as u
        import numpy as np

        testing_DL2_gamma_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[0]
        testing_DL2_proton_files = self.direction_model.get_DL2_MC_files(zenith, azimuth)[1]


        tel_id = None if self.stereo else self.telescope_ids[0]

      
        if len(testing_DL2_gamma_files) > 0:
            dl2_gamma = []
            for file in testing_DL2_gamma_files:
                dl2_gamma.append(load_DL2_data_MC(file, tel_id=tel_id))
            dl2_gamma = vstack(dl2_gamma)
        else:
            dl2_gamma = []
        mc_type_gamma = np.zeros(len(dl2_gamma))
        
        
        if len(testing_DL2_proton_files) > 0:
            dl2_protons = []
            for file in testing_DL2_proton_files:
                dl2_protons.append(load_DL2_data_MC(file, tel_id=tel_id))
            dl2_proton = vstack(dl2_protons)
        else:
            dl2_proton = []
        mc_type_proton = np.ones(len(dl2_proton))
            
        mc_type = np.concatenate((mc_type_gamma, mc_type_proton))
        gammaness = np.concatenate((dl2_gamma[self.gammaness_key], dl2_proton[self.gammaness_key]))
        mc_gamma_energies = np.concatenate((dl2_gamma[self.true_energy_key], dl2_proton[self.true_energy_key]))
        plt.figure(figsize=(14,8))
        ax = ctaplot.plot_roc_curve_gammaness_per_energy(mc_type, gammaness, mc_gamma_energies,
                                                        energy_bins=u.Quantity([0.01,0.1,1,3,10], u.TeV),
                                                        linestyle='--',
                                                        alpha=0.8,
                                                        linewidth=3,
                                                        )
        ax.legend()
        plt.show()
        
    def compare_irfs_to_RF(self, zenith, azimuth=None):
        set_mpl_style()
        from astropy.io import fits
        import matplotlib.pyplot as plt
        from .resources.irfs import SST1M
        from astropy.coordinates import Angle
        from gammapy.irf import EnergyDispersion2D, EffectiveAreaTable2D, Background2D, PSF3D
        import importlib
        from astropy.table import Table
        import importlib.resources as pkg_resources
    
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
        