from astropy.table import QTable
import numpy as np
from pathlib import Path
import ast
from ctlearn_manager.utils.utils import set_mpl_style, ClusterConfiguration, ParticleType, DataSample

__all__ = [
    'CTLearnModelManager',
    'DataSample',
    'ModelRangeOfValidity',
    ]

class CTLearnModelManager():

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
    def __init__(self, model_parameters, MODEL_INDEX_FILE, load=False, cluster_configuration=ClusterConfiguration()):
        """
        Initializes the ModelManager instance.

        :param model_parameters: Dictionary containing model parameters.
        :type model_parameters: dict
        :param MODEL_INDEX_FILE: Path to the model index file.
        :type MODEL_INDEX_FILE: str
        :param load: If True, loads an existing model. If False, initializes a new model and saves it to the index.
        :type load: bool
        :param cluster_configuration: Configuration for the cluster. Default is an instance of ClusterConfiguration.
        :type cluster_configuration: ClusterConfiguration

        :raises ValueError: If 'reco' type is 'type' and training_proton_patterns or training_gamma_patterns are missing.
        :raises ValueError: If gamma related lists (training_gamma_patterns, training_gamma_zenith_distances, training_gamma_azimuths) are not the same length.
        :raises ValueError: If proton related lists (training_proton_patterns, training_proton_zenith_distances, training_proton_azimuths) are not the same length.
        """



        from astropy.io.misc.hdf5 import read_table_hdf5
        self.model_index_file = MODEL_INDEX_FILE
        self.model_nickname = model_parameters.get('model_nickname', 'new_model')
        if not load:
            self.save_to_index(model_parameters)
            print(f"🧠 Model name: {self.model_nickname}")
        self.model_parameters_table = read_table_hdf5(f"{self.model_index_file}", path=f"{self.model_nickname}/parameters")
        training_table_gamma = read_table_hdf5(f"{self.model_index_file}", path=f"{self.model_nickname}/training/gamma_diffuse")
        training_table_proton = read_table_hdf5(f"{self.model_index_file}", path=f"{self.model_nickname}/training/proton")
        self.validity = ModelRangeOfValidity(self)
        if self.model_parameters_table['reco'][0] == 'type' and (len(training_table_proton['training_proton_patterns']) == 0 or len(training_table_gamma['training_gamma_diffuse_patterns']) == 0):
            raise ValueError("For reco type, training_proton_patterns and training_gamma_diffuse_patterns are required")
        self.telescope_ids = ast.literal_eval(self.model_parameters_table['telescope_ids'][0])
        self.telescope_names = ast.literal_eval(self.model_parameters_table['telescope_names'][0])
        try:
            self.min_telescopes = self.model_parameters_table['min_telescopes'][0] 
        except:
            self.min_telescopes = len(self.telescope_ids)
        self.stereo = True if self.min_telescopes >= 2 else False
        # Check that all gamma related lists are the same length
        # gamma_lengths = [len(training_table_gamma['training_gamma_diffuse_patterns']), len(training_table_gamma['training_gamma_diffuse_zenith_distances']), len(training_table_gamma['training_gamma_diffuse_azimuths'])]
        # if len(set(gamma_lengths)) != 1:
        #     raise ValueError("All gamma related lists must be the same length")

        # Check that all proton related lists are the same length
        # proton_lengths = [len(training_table_proton['training_proton_patterns']), len(training_table_proton['training_proton_zenith_distances']), len(training_table_proton['training_proton_azimuths'])]
        # if len(set(proton_lengths)) != 1:
        #     raise ValueError("All proton related lists must be the same length")

        self.cluster_configuration = cluster_configuration

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
        This method saves the model parameters and training samples (gamma and proton) to an HDF5 file.
        If the model nickname already exists in the index file, it prints a message and does not overwrite the existing entry.
        :param model_parameters: Dictionary containing model parameters and training samples.
            The dictionary should have the following keys:
            - 'notes' (str): Notes about the model.
            - 'model_dir' (str): Directory where the model is stored.
            - 'reco' (str): Reconstruction method used by the model.
            - 'telescope_names' (list): List of telescope names used in the model.
            - 'telescope_ids' (list): List of telescope IDs used in the model.
            - 'channels' (list): List of channels used in the model.
            - 'max_training_epochs' (int): Maximum number of training epochs.
            - 'gamma_training_samples' (list): List of gamma training samples.
            - 'proton_training_samples' (list): List of proton training samples.
        :raises: Exception if there is an error reading or writing to the HDF5 file.
        :return: None
        """
        

        
        from astropy.io.misc.hdf5 import write_table_hdf5, read_table_hdf5
        
        try:
            model_table = QTable.read(self.model_index_file, format='hdf5', path=f'{self.model_nickname}/parameters')
            print(f"❌ Model nickname {self.model_nickname} already in table")
        except:
            model_table = QTable(names=['model_nickname', 'model_dir', 'reco', 'channels', 'telescope_names', 'telescope_ids', 'notes', 'max_training_epochs', 'min_telescopes'],
                        dtype=['S256', 'S256', 'S256', 'S256', 'S256', 'S256', 'S256', int, bool, int])
            notes = model_parameters.get('notes', '')
            model_dir = f"{model_parameters.get('model_dir', '')}/{self.model_nickname}"
            reco = model_parameters.get('reco', 'default_reco')
            telescope_names = model_parameters.get('telescope_names', [])
            telescope_ids = model_parameters.get('telescope_ids', [])
            channels = model_parameters.get('channels', ['cleaned_image', 'cleaned_relative_peak_time'])
            max_training_epochs = model_parameters.get('max_training_epochs', 10)
            min_telescopes = model_parameters.get('min_telescopes', 1)
            model_table.add_row([self.model_nickname, model_dir, reco, str(channels), str(telescope_names), str(telescope_ids), notes, max_training_epochs, min_telescopes])
            write_table_hdf5(model_table, self.model_index_file, path=f'{self.model_nickname}/parameters',append=True, overwrite=True,)

            training_samples = model_parameters.get('training_samples', []).astype(list[DataSample])

            for training_sample in training_samples:
                particle_type = training_sample.particle_type

                try:
                    training_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/training/{particle_type.value}')
                except:
                    training_table = QTable(names=[f'training_{particle_type.value}_dir', f'training_{particle_type.value}_patterns', f'training_{particle_type.value}_zenith_distances', f'training_{particle_type.value}_azimuths', f'training_{particle_type.value}_energy_min', f'training_{particle_type.value}_energy_max', f'training_{particle_type.value}_nsb_min', f'training_{particle_type.value}_nsb_max'],
                                    dtype=['S256', 'S256', float, float, float, float, float, float], units=[None, None, 'deg', 'deg', 'TeV', 'TeV', 'Hz', 'Hz'])
                
                training_table.add_row([training_sample.directory, 
                                        training_sample.pattern, 
                                        training_sample.zenith_distance, 
                                        training_sample.azimuth,
                                        min(training_sample.energy_range),
                                        max(training_sample.energy_range),
                                        min(training_sample.nsb_range),
                                        max(training_sample.nsb_range)])
                write_table_hdf5(training_table, self.model_index_file, path=f'{self.model_nickname}/training/{particle_type.value}', append=True, overwrite=True, serialize_meta=True)
            print(f"✅ Model nickname {self.model_nickname} added to table")
        
        
    def launch_training(self, n_epochs, save_best_validation_only=None, transfer_learning_model_cpk=None, trainable_backbone=True, config_file=None, batch_size=64):
        """
        Launches the training process for the model.
        :param n_epochs: Number of epochs to train the model.
        :type n_epochs: int
        :param transfer_learning_model_cpk: Path to the checkpoint file for transfer learning, defaults to None.
        :type transfer_learning_model_cpk: str, optional
        :param frozen_backbone: Whether to freeze the backbone of the model during training, defaults to False.
        :type frozen_backbone: bool, optional
        :param config_file: Path to the configuration file, defaults to None.
        :type config_file: str, optional
        :return: None
        """

        
        import json
        from astropy.io.misc.hdf5 import read_table_hdf5
        import glob
        import os
        import numpy as np
        self.cluster_configuration.info()
        if n_epochs == 0:
            print("🛑 Number of epochs set to 0. Will not train the model.")
            return
        n_epoch_trained = self.get_n_epoch_trained()
        max_training_epochs = self.model_parameters_table['max_training_epochs'][0]
        base_model_dir = self.model_parameters_table['model_dir'][0]
        if n_epochs > max_training_epochs - n_epoch_trained:
            print(f"⚠️ Number of epochs increased from {max_training_epochs} to {n_epochs}")
            self.update_model_manager_parameters_in_index({'max_training_epochs': n_epochs})
            max_training_epochs = n_epochs
            n_epochs = max_training_epochs - n_epoch_trained
        if n_epoch_trained >= max_training_epochs:
            print(f"🛑 Model already trained for {n_epoch_trained} epochs. Will not train further.")
            self.plot_loss()
            return
            
        trained_string = "―" * (n_epoch_trained-1)
        trained_spaces = " " * (n_epoch_trained-1)
        remaining_string = "·" * (max_training_epochs - n_epoch_trained)
        to_train_string = "―" * (n_epochs- len(str(n_epoch_trained)))
        print(f"{trained_string}o{remaining_string} | {n_epoch_trained}/{max_training_epochs} epochs")
        print(f"{trained_spaces}{n_epoch_trained}{to_train_string}> 🚀 Training for {n_epochs} epochs")
        
        models_dir = np.sort(glob.glob(f"{base_model_dir}/{self.model_nickname}_v*"))
        load_model = False
        if len(models_dir) > 0 :
            last_model_dir = Path(models_dir[-1])
            size = sum(f.stat().st_size for f in last_model_dir.glob('**/*') if f.is_file())
            model_version = int(models_dir[-1].split("_v")[-1])
            if size > 1e6:
                model_version += 1
                model_dir = f"{base_model_dir}/{self.model_nickname}_v{model_version}/"
                print(f"➡️ Model already exists: will continue training and create {model_dir}")
                _save_best_validation_only = True
                model_to_load = f"{base_model_dir}/{self.model_nickname}_v{model_version - 1}/ctlearn_model.cpk"
                load_model = True
                os.system(f"mkdir -p {model_dir}")
            else :
                model_dir = f"{base_model_dir}/{self.model_nickname}_v{model_version}/"
                if model_version > 0:
                    model_to_load = f"{base_model_dir}/{self.model_nickname}_v{model_version - 1}/ctlearn_model.cpk"
                    load_model = True
                    print(f"➡️ Model already exists: will continue training and create {model_dir}")
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
            load_model_string = "" if transfer_learning_model_cpk is None else f"--TrainCTLearnModel.model_type=LoadedModel --LoadedModel.load_model_from={transfer_learning_model_cpk} "
        
        training_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/training/proton')
        training_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/training/gamma_diffuse')
        background_string = f"--background {training_proton_table['training_proton_dir'][0]} " if self.model_parameters_table['reco'][0] == 'type' else ""
        signal_patterns = ""
        for pattern in training_gamma_table['training_gamma_diffuse_patterns']:
            signal_patterns += f'--pattern-signal "{pattern}" '
        background_patterns = ""
        if self.model_parameters_table['reco'][0] == 'type':
            for pattern in training_proton_table['training_proton_patterns']:
                background_patterns += f'--pattern-background "{pattern}" '
        channels = ast.literal_eval(self.model_parameters_table['channels'][0])
        stereo_mode = 'stereo' if self.stereo else "mono"
        stack_telescope_images = True if self.stereo else False
        allowed_tels = ast.literal_eval(self.model_parameters_table['telescope_ids'][0])
        
        if config_file is None:
            config = {}
            config['TrainCTLearnModel'] = {}
            config['TrainCTLearnModel']['save_best_validation_only'] = _save_best_validation_only
            config['TrainCTLearnModel']['n_epochs'] = int(n_epochs)
            config['TrainCTLearnModel']['stack_telescope_images'] = stack_telescope_images
            config['TrainCTLearnModel']['reco_tasks'] = [self.model_parameters_table['reco'][0]]
            config['TrainCTLearnModel']['output_dir'] = model_dir

            config['TrainCTLearnModel']['DLImageReader'] = {}
            config['TrainCTLearnModel']['DLImageReader']['allowed_tels'] = allowed_tels
            config['TrainCTLearnModel']['DLImageReader']['min_telescopes'] = self.min_telescopes
            config['TrainCTLearnModel']['DLImageReader']['mode'] = stereo_mode
            config['TrainCTLearnModel']['DLImageReader']['channels'] = channels
            
            config['LoadedModel'] = {}
            config['LoadedModel']['trainable_backbone'] = trainable_backbone
            
            config_file = f"{base_model_dir}/train_config{self.model_nickname}_v{model_version}.json"
            with open(config_file, 'w') as file:
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
            sbatch_file = self.cluster_configuration.write_sbatch_script(f"train_{self.model_nickname}_v{model_version}", cmd, base_model_dir)
            os.system(f"sbatch {sbatch_file}")
        else:
            print(cmd)
            os.system(cmd)
        
    def get_n_epoch_trained(self):
        """
        Calculate the total number of epochs trained by summing the lengths of all training logs.
        This method searches for all training log files in the model directory that match the model nickname pattern,
        reads each log file as a pandas DataFrame, and sums the number of rows (epochs) in each DataFrame.
        Returns:
            int: The total number of epochs trained.
        """
        
        
        import glob
        import pandas as pd
        training_logs = np.sort(glob.glob(f"{self.model_parameters_table['model_dir'][0]}/{self.model_nickname}*/training_log.csv"))
        n_epochs = 0
        for training_log in training_logs:
            df = pd.read_csv(training_log)
            n_epochs += len(df)
        return n_epochs

    def plot_loss(self):
        """
        Plots the training and validation loss over epochs.
        This method reads training logs from CSV files, concatenates the loss values,
        and plots the training and validation loss over the epochs using Matplotlib.
        The training logs are expected to be located in the directory specified by
        `self.model_parameters_table['model_dir'][0]` and should follow the naming
        convention `{self.model_nickname}*/training_log.csv`.
        The plot will display the training loss with a solid line and the validation
        loss with a dashed line. The x-axis represents the epochs, and the y-axis
        represents the loss values.
        Imports:
            - matplotlib.pyplot as plt
            - pandas as pd
            - glob
            - numpy as np
        Raises:
            FileNotFoundError: If no training log files are found in the specified directory.
        """

       
        import matplotlib.pyplot as plt
        import pandas as pd
        import glob
        set_mpl_style()
        training_logs = np.sort(glob.glob(f"{self.model_parameters_table['model_dir'][0]}/{self.model_nickname}*/training_log.csv"))
        losses_train = []
        losses_val = []
        for training_log in training_logs:
            df = pd.read_csv(training_log)
            losses_train = np.concatenate((losses_train, df['loss'].to_numpy()))
            losses_val = np.concatenate((losses_val, df['val_loss'].to_numpy()))
        epochs = np.arange(1, len(losses_train)+1)
        if len(epochs) == 0:
            print(f"❌ No training logs found for {self.model_nickname}, stert the training to see the loss.")
            return
        if len(epochs) > 1:
            plt.plot(epochs, losses_train, label=f"Training", lw=2)
            plt.plot(epochs, losses_val, label=f"Validation", ls='--')
        else:
            plt.scatter(epochs, losses_train, label=f"Training", lw=2)
            plt.scatter(epochs, losses_val, label=f"Validation", ls='--')
        plt.title(f"{self.model_parameters_table['reco'][0]} training".title())
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(np.arange(1, len(losses_train) + 1, 2))
        plt.legend()
        plt.show()
        
    def update_model_manager_parameters_in_index(self, parameters: dict):
        """
        Update the model manager parameters in the HDF5 index file.
        This method reads the model parameters from the HDF5 file, updates them with the provided parameters,
        and writes the updated parameters back to the HDF5 file. It also updates the instance attributes with
        the new parameter values.
        :param parameters: Dictionary containing the parameters to update. The keys should match the parameter
                           names in the HDF5 file, and the values are the new values to set.
        :type parameters: dict
        """

        
        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
        
        model_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/parameters')
        # model_index = np.where(model_table['model_nickname'] == self.model_nickname)[0][0]
        print(f"💾 Model {self.model_nickname} index update:")
        for key, value in parameters.items():
            if model_table[key].dtype.kind == 'U':  # Check if the dtype is Unicode string
                model_table[key] = model_table[key].astype('S256')  # Convert to 'S256' for long strings
            model_table[key][0] = value
            self.__dict__[key] = value
            print(f"\t➡️ {key} updated to {value}")
        write_table_hdf5(model_table, self.model_index_file, path=f'{self.model_nickname}/parameters', append=True, overwrite=True, serialize_meta=True)
        
    # def update_model_manager_training_data(self, training_gamma_dir, training_proton_dir, training_gamma_patterns, training_proton_patterns, training_gamma_zenith_distances, training_gamma_azimuths, training_proton_zenith_distances, training_proton_azimuths):

        # from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
        
        # training_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/training/gamma')
        # training_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/training/proton')
        # print(f"💾 Model {self.model_nickname} training data update:")
        # if len(training_gamma_table)==0:
        #     training_gamma_table = QTable(names=['training_gamma_dir', 'training_gamma_patterns', 'training_gamma_zenith_distances', 'training_gamma_azimuths'], 
        #                                   dtype=[str, str, float, float])
        # if len(training_proton_table)==0:
        #     training_proton_table = QTable(names=['training_proton_dir', 'training_proton_patterns', 'training_proton_zenith_distances', 'training_proton_azimuths'], 
        #                                    dtype=[str, str, float, float])
        
        # if len(training_gamma_patterns) > 0:
        #     for i in range(len(training_gamma_patterns)):
        #         match = np.where((training_gamma_table['training_gamma_zenith_distances'] == training_gamma_zenith_distances[i]) & 
        #              (training_gamma_table['training_gamma_azimuths'] == training_gamma_azimuths[i]))[0]
        #         if len(match) > 0:
        #             training_gamma_table['training_gamma_dir'][match[0]] = training_gamma_dir
        #             training_gamma_table['training_gamma_patterns'][match[0]] = training_gamma_patterns[i]
        #         else:
        #             training_gamma_table.add_row([training_gamma_dir, training_proton_patterns[i], training_gamma_zenith_distances[i], training_gamma_azimuths[i]])
        #     write_table_hdf5(training_gamma_table, self.model_index_file, path=f'{self.model_nickname}/training/gamma', append=True, overwrite=True)
        #     print(f"\t➡️ Training gamma data updated")
        
        # if len(training_proton_patterns) > 0:
        #     for i in range(len(training_proton_patterns)):
        #         match = np.where((training_proton_table['training_proton_zenith_distances'] == training_proton_zenith_distances[i]) & 
        #              (training_proton_table['training_proton_azimuths'] == training_proton_azimuths[i]))[0]
        #         if len(match) > 0:
        #             training_proton_table['training_proton_dir'][match[0]] = training_proton_dir
        #             training_proton_table['training_proton_patterns'][match[0]] = training_proton_patterns[i]
        #         else:
        #             training_proton_table.add_row([training_proton_dir, training_proton_patterns[i], training_proton_zenith_distances[i], training_proton_azimuths[i]])
        #     write_table_hdf5(training_proton_table, self.model_index_file, path=f'{self.model_nickname}/training/proton', append=True, overwrite=True)
        #     print(f"\t➡️ Training proton data updated")

    # def update_model_manager_testing_data(self, testing_gamma_dirs, testing_proton_dirs, testing_gamma_zenith_distances, testing_gamma_azimuths, testing_proton_zenith_distances, testing_proton_azimuths, testing_gamma_patterns, testing_proton_patterns):
    def update_model_manager_testing_data(self, testing_data_sample: DataSample):
        """
        Update the model manager's testing data for gamma and proton events.
        This method reads the existing testing data from an HDF5 file, updates it with the provided
        testing data, and writes the updated data back to the HDF5 file. If no existing data is found,
        new tables are created.
        :param testing_gamma_dirs: List of directories containing gamma testing data.
        :type testing_gamma_dirs: list of str
        :param testing_proton_dirs: List of directories containing proton testing data.
        :type testing_proton_dirs: list of str
        :param testing_gamma_zenith_distances: List of zenith distances for gamma testing data.
        :type testing_gamma_zenith_distances: list of float
        :param testing_gamma_azimuths: List of azimuths for gamma testing data.
        :type testing_gamma_azimuths: list of float
        :param testing_proton_zenith_distances: List of zenith distances for proton testing data.
        :type testing_proton_zenith_distances: list of float
        :param testing_proton_azimuths: List of azimuths for proton testing data.
        :type testing_proton_azimuths: list of float
        :param testing_gamma_patterns: List of patterns for gamma testing data.
        :type testing_gamma_patterns: list of str
        :param testing_proton_patterns: List of patterns for proton testing data.
        :type testing_proton_patterns: list of str
        :raises IOError: If there is an error reading or writing the HDF5 file.
        :return: None
        """

        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
        testing_dir = testing_data_sample.directory
        testing_zenith_distance = testing_data_sample.zenith_distance
        testing_azimuth = testing_data_sample.azimuth
        testing_pattern = testing_data_sample.pattern
        particle_type = testing_data_sample.particle_type
        try:
            testing_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/testing/{particle_type.value}')
        except:
            testing_table = QTable(names=[f'testing_{particle_type.value}_dirs', f'testing_{particle_type.value}_zenith_distances', f'testing_{particle_type.value}_azimuths', f'testing_{particle_type.value}_patterns'], 
                                        dtype=['S256', float, float, 'S256'])
        # print(f"💾 Model {self.model_nickname} testing data update:")
        if len(testing_table)==0:
            testing_table = QTable(names=['testing_gamma_dirs', 'testing_gamma_zenith_distances', 'testing_gamma_azimuths', 'testing_gamma_patterns'], 
                                        dtype=['S256', float, float, 'S256'])
        
 
        match = np.where((testing_table[f'testing_{particle_type.value}_zenith_distances'] == testing_zenith_distance) & 
                (testing_table[f'testing_{particle_type.value}_azimuths'] == testing_azimuth))[0]
        if len(match) > 0:
            testing_table[f'testing_{particle_type.value}_dirs'][match[0]] = testing_dir
            testing_table[f'testing_{particle_type.value}_patterns'][match[0]] = testing_pattern
        else:
            testing_table.add_row([testing_dir, testing_zenith_distance, testing_azimuth, testing_pattern])
        write_table_hdf5(testing_table, self.model_index_file, path=f'{self.model_nickname}/testing/{particle_type.value}', append=True, overwrite=True, serialize_meta=True)
        print(f"\t➡️ Testing {particle_type.value} data updated")
       
    def update_model_manager_DL2_MC_file(self, testing_MC_DL2_file: str, testing_MC_DL2_data_sample: DataSample):
        """
        Update the DL2 MC files for gamma and proton testing data in the model manager.
        This method reads the existing DL2 MC tables for gamma and proton data from the HDF5 file,
        updates them with the provided testing data, and writes the updated tables back to the HDF5 file.
        :param testing_DL2_gamma_files: List of file paths for testing DL2 gamma data.
        :type testing_DL2_gamma_files: list of str
        :param testing_DL2_proton_files: List of file paths for testing DL2 proton data.
        :type testing_DL2_proton_files: list of str
        :param testing_DL2_gamma_zenith_distances: List of zenith distances for testing DL2 gamma data.
        :type testing_DL2_gamma_zenith_distances: list of float
        :param testing_DL2_gamma_azimuths: List of azimuths for testing DL2 gamma data.
        :type testing_DL2_gamma_azimuths: list of float
        :param testing_DL2_proton_zenith_distances: List of zenith distances for testing DL2 proton data.
        :type testing_DL2_proton_zenith_distances: list of float
        :param testing_DL2_proton_azimuths: List of azimuths for testing DL2 proton data.
        :type testing_DL2_proton_azimuths: list of float
        :raises IOError: If there is an error reading or writing the HDF5 file.
        :return: None
        """
        
        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5

        testing_zenith_distance = testing_MC_DL2_data_sample.zenith_distance
        testing_azimuth = testing_MC_DL2_data_sample.azimuth
        particle_type = testing_MC_DL2_data_sample.particle_type
        
        try:
            DL2_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/{particle_type.value}')
        except:
            DL2_gamma_table = QTable(names=[f'testing_DL2_{particle_type.value}_files', f'testing_DL2_{particle_type.value}_zenith_distances', f'testing_DL2_{particle_type.value}_azimuths'], 
                                     dtype=[ 'S256', float, float])

        print(f"💾 Model {self.model_nickname} DL2 data update:")
        if len(DL2_gamma_table)==0:
            DL2_gamma_table = QTable(names=[f'testing_DL2_{particle_type.value}_files', f'testing_DL2_{particle_type.value}_zenith_distances', f'testing_DL2_{particle_type.value}_azimuths'], 
                                     dtype=['S256', float, float])

        match = np.where((DL2_gamma_table[f'testing_DL2_{particle_type.value}_files'] == testing_MC_DL2_file) & 
                (DL2_gamma_table[f'testing_DL2_{particle_type.value}_zenith_distances'] == testing_zenith_distance) & 
                (DL2_gamma_table[f'testing_DL2_{particle_type.value}_azimuths'] == testing_azimuth))[0]
        if len(match) == 0:
            DL2_gamma_table.add_row([testing_MC_DL2_file, testing_zenith_distance, testing_azimuth])
        write_table_hdf5(DL2_gamma_table, self.model_index_file, path=f'{self.model_nickname}/DL2/MC/{particle_type.value}', append=True, overwrite=True, serialize_meta=True)
        print(f"\t➡️ Testing DL2 {particle_type.value} data updated")
    
    def delete_DL2_MC_file(self, testing_DL2_file: str, particle_type: ParticleType):
        """
        Delete the DL2 MC files for gamma and proton testing data in the model manager.
        This method reads the existing DL2 MC tables for gamma and proton data from the HDF5 file,
        removes the specified files from the tables, and writes the updated tables back to the HDF5 file.
        :param testing_DL2_gamma_file: The file path of the DL2 gamma data to be deleted, defaults to None.
        :type
        testing_DL2_gamma_file: str, optional
        :param testing_DL2_proton_file: The file path of the DL2 proton data to be deleted, defaults to None.
        :type testing_DL2_proton_file: str, optional
        :raises IOError: If there is an error reading or writing the HDF5 file.
        :return: None
        """
        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
        
        DL2_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/{particle_type.value}')
        match = np.where(DL2_table['testing_DL2_{particle_type.value}_files'] == testing_DL2_file)[0]
        if len(match) > 0:
            DL2_table.remove_rows(match)
        write_table_hdf5(DL2_table, self.model_index_file, path=f'{self.model_nickname}/DL2/MC/{particle_type.value}', append=True, overwrite=True, serialize_meta=True)


    def update_model_manager_DL2_data_files(self, DL2_files, DL2_zenith_distances, DL2_azimuths,):
        
        """
        Update the DL2 data files for the model manager.
        This method reads the existing DL2 data from an HDF5 file, updates it with the provided
        DL2 files, zenith distances, and azimuths, and writes the updated data back to the HDF5 file.
        :param DL2_files: List of DL2 file paths to be added or updated.
        :type DL2_files: list of str
        :param DL2_zenith_distances: List of zenith distances corresponding to the DL2 files.
        :type DL2_zenith_distances: list of float
        :param DL2_azimuths: List of azimuths corresponding to the DL2 files.
        :type DL2_azimuths: list of float
        :raises IOError: If there is an issue reading or writing the HDF5 file.
        :returns: None
        """
        

        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5


        
        try:
            DL2_data_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/Data')
        except:
            DL2_data_table = QTable(names=['DL2_files', 'DL2_zenith_distances', 'DL2_azimuths'], 
                                     dtype=[ 'S256', float, float])

        print(f"💾 Model {self.model_nickname} DL2 data update:")
        if len(DL2_data_table)==0:
            DL2_data_table = QTable(names=['DL2_files', 'DL2_zenith_distances', 'DL2_azimuths'], 
                                     dtype=['S256', float, float])
        
        if len(DL2_files) > 0:
            for i in range(len(DL2_files)):
                match = np.where(
                    (DL2_data_table['DL2_files'] == DL2_files[i]) & 
                        (DL2_data_table['DL2_zenith_distances'] == DL2_zenith_distances[i]) & 
                        (DL2_data_table['DL2_azimuths'] == DL2_azimuths[i]))[0]
                if len(match) == 0:
                    DL2_data_table.add_row([DL2_files[i], DL2_zenith_distances[i], DL2_azimuths[i]])
                # else:
                #     DL2_data_table.remove_rows(match)
            write_table_hdf5(DL2_data_table, self.model_index_file, path=f'{self.model_nickname}/DL2/Data', append=True, overwrite=True, serialize_meta=True)
            print(f"\t➡️ Testing DL2 real data updated")
    
    def update_merged_DL2_MC_files(self, testing_DL2_zenith_distance, testing_DL2_azimuth, testing_DL2_gamma_merged_file=None, testing_DL2_proton_merged_file=None):
        """
        Update the merged DL2 MC files for gamma and proton data.
        This method updates the DL2 merged data for gamma and proton events in the model index file.
        It reads the existing data, removes any matching rows based on the provided zenith distance and azimuth,
        and then adds the new data.
        :param testing_DL2_zenith_distance: The zenith distance of the testing DL2 data.
        :type testing_DL2_zenith_distance: float
        :param testing_DL2_azimuth: The azimuth of the testing DL2 data.
        :type testing_DL2_azimuth: float
        :param testing_DL2_gamma_merged_file: The file path of the merged DL2 gamma data, defaults to None.
        :type testing_DL2_gamma_merged_file: str, optional
        :param testing_DL2_proton_merged_file: The file path of the merged DL2 proton data, defaults to None.
        :type testing_DL2_proton_merged_file: str, optional
        :return: None
        """


       
        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
        print(f"💾 Model {self.model_nickname} DL2 merged data update:")
        if testing_DL2_gamma_merged_file is not None:
            DL2_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/gamma')
            match = np.where((DL2_gamma_table['testing_DL2_gamma_zenith_distances'] == testing_DL2_zenith_distance) &
                    (DL2_gamma_table['testing_DL2_gamma_azimuths'] == testing_DL2_azimuth))[0]
            if len(match) > 0:
                DL2_gamma_table.remove_rows(match)
            DL2_gamma_table.add_row([testing_DL2_gamma_merged_file, testing_DL2_zenith_distance, testing_DL2_azimuth])
            write_table_hdf5(DL2_gamma_table, self.model_index_file, path=f'{self.model_nickname}/DL2/MC/gamma', append=True, overwrite=True, serialize_meta=True)
            print(f"\t➡️ Testing DL2 gamma merged data updated")
        
        if testing_DL2_proton_merged_file is not None:
            DL2_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/proton')
            match = np.where((DL2_proton_table['testing_DL2_proton_zenith_distances'] == testing_DL2_zenith_distance) &
                        (DL2_proton_table['testing_DL2_proton_azimuths'] == testing_DL2_azimuth))[0]
            if len(match) > 0:
                DL2_proton_table.remove_rows(match)
            DL2_proton_table.add_row([testing_DL2_proton_merged_file, testing_DL2_zenith_distance, testing_DL2_azimuth])
            write_table_hdf5(DL2_proton_table, self.model_index_file, path=f'{self.model_nickname}/DL2/MC/proton', append=True, overwrite=True, serialize_meta=True)
            print(f"\t➡️ Testing DL2 proton merged data updated")
        
    def update_model_manager_IRF_data(self, config, cuts_file, irf_file, bencmark_file, zenith, azimuth):
        """
        Update the IRF (Instrument Response Function) data for the model manager.
        This method reads the existing IRF data from an HDF5 file, checks if the provided
        IRF data already exists, and updates the table accordingly. If the data does not
        exist, it adds a new row with the provided data. If the data exists, it removes
        the existing row and adds the new data.
        :param config: Configuration identifier for the IRF data.
        :type config: str
        :param cuts_file: Path to the cuts file.
        :type cuts_file: str
        :param irf_file: Path to the IRF file.
        :type irf_file: str
        :param bencmark_file: Path to the benchmark file.
        :type bencmark_file: str
        :param zenith: Zenith angle for the IRF data.
        :type zenith: float
        :param azimuth: Azimuth angle for the IRF data.
        :type azimuth: float
        :raises IOError: If there is an error reading or writing the HDF5 file.
        :return: None
        """


        
        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
        
        try:
            IRF_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/IRF')
        except:
            IRF_table = QTable(names=['config', 'cuts_file', 'irf_file', 'benckmark_file', 'zenith', 'azimuth'], 
                               dtype=['S256', 'S256', 'S256', 'S256', float, float])
        print(f"💾 Model {self.model_nickname} IRF data update:")
        if len(IRF_table)==0:
            IRF_table = QTable(names=['config', 'cuts_file', 'irf_file', 'benckmark_file', 'zenith', 'azimuth'], 
                               dtype=['S256', 'S256', 'S256', 'S256', float, float])
        
        match = np.where((IRF_table['config'] == config) or 
                (IRF_table['cuts_file'] == cuts_file) or 
                (IRF_table['irf_file'] == irf_file) or
                (IRF_table['benckmark_file'] == bencmark_file) or
                ((IRF_table['zenith'] == zenith) and
                (IRF_table['azimuth'] == azimuth))
                )[0]
        if len(match) == 0:
            IRF_table.add_row([config, cuts_file, irf_file, bencmark_file, zenith, azimuth])
            write_table_hdf5(IRF_table, self.model_index_file, path=f'{self.model_nickname}/IRF', append=True, overwrite=True, serialize_meta=True)
        else:
            IRF_table.remove_rows(match)
            IRF_table.add_row([config, cuts_file, irf_file, bencmark_file, zenith, azimuth])
            write_table_hdf5(IRF_table, self.model_index_file, path=f'{self.model_nickname}/IRF', append=True, overwrite=True, serialize_meta=True)
        print(f"\t➡️ IRF data updated")
        
    def get_IRF_data(self, zenith, azimuth):
        """
        Retrieve the Instrument Response Function (IRF) data for a given zenith and azimuth.
        :param zenith: The zenith angle for which to retrieve the IRF data.
        :type zenith: float
        :param azimuth: The azimuth angle for which to retrieve the IRF data.
        :type azimuth: float
        :returns: A tuple containing the configuration, cuts file, IRF file, and benchmark file.
        :rtype: tuple
        :raises IndexError: If no IRF data is found for the specified zenith and azimuth.
        """


        
        from astropy.io.misc.hdf5 import read_table_hdf5
        
        IRF_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/IRF')
        match = np.where((IRF_table['zenith'] == zenith) & (IRF_table['azimuth'] == azimuth))[0]
        if len(match) == 0:
            raise IndexError(f"No IRF data found for altitude {zenith} and azimuth {azimuth}")
        return IRF_table['config'][match][0], IRF_table['cuts_file'][match][0], IRF_table['irf_file'][match][0], IRF_table['benckmark_file'][match][0]

    def get_closest_IRF_data(self, zenith, azimuth):
        """
        Retrieve the closest Instrument Response Function (IRF) data based on the given zenith and azimuth angles.
        This function reads an HDF5 table containing IRF data and finds the entry with the closest matching zenith and azimuth angles to the provided values. It then returns the configuration, cuts file, IRF file, and benchmark file associated with the closest match.
        :param zenith: The zenith angle to match.
        :type zenith: float
        :param azimuth: The azimuth angle to match.
        :type azimuth: float
        :return: A tuple containing the configuration, cuts file, IRF file, and benchmark file of the closest matching IRF data.
        :rtype: tuple
        """

        
        from astropy.io.misc.hdf5 import read_table_hdf5
        
        IRF_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/IRF')
        match = np.argmin(np.abs(IRF_table['zenith'] - zenith) + np.abs(IRF_table['azimuth'] - azimuth))
        return IRF_table['config'][match], IRF_table['cuts_file'][match], IRF_table['irf_file'][match], IRF_table['benckmark_file'][match]

    def get_DL2_MC_files(self, zenith, azimuth):
        """
        Retrieve DL2 Monte Carlo (MC) files for given zenith and azimuth angles.
        This method reads HDF5 tables containing DL2 MC data for gamma and proton particles,
        and filters the files based on the provided zenith and azimuth angles.
        :param zenith: Zenith angle to filter the DL2 MC files.
        :type zenith: float
        :param azimuth: Azimuth angle to filter the DL2 MC files.
        :type azimuth: float
        :return: A tuple containing two lists: DL2 gamma MC files and DL2 proton MC files.
        :rtype: tuple(list, list)
        :raises IndexError: If no matching DL2 gamma or proton MC files are found for the given zenith and azimuth.
        """

    
        from astropy.io.misc.hdf5 import read_table_hdf5
        
        try:
            DL2_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/gamma')
            match_gamma = np.where((DL2_gamma_table['testing_DL2_gamma_zenith_distances'] == zenith) & (DL2_gamma_table['testing_DL2_gamma_azimuths'] == azimuth))[0]
            if len(match_gamma) == 0:
                raise IndexError(f"No DL2 gamma MC files found for zenith {zenith} and azimuth {azimuth}")
            DL2_gamma_files = DL2_gamma_table['testing_DL2_gamma_files'][match_gamma]
        except:
            DL2_gamma_files = []
        try:
            DL2_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/proton')
            match_proton = np.where((DL2_proton_table['testing_DL2_proton_zenith_distances'] == zenith) & (DL2_proton_table['testing_DL2_proton_azimuths'] == azimuth))[0]
            if len(match_proton) == 0:
                raise IndexError(f"No DL2 proton MC files found for zenith {zenith} and azimuth {azimuth}")
            DL2_proton_files = DL2_proton_table['testing_DL2_proton_files'][match_proton]
        except:
            DL2_proton_files = []
        return DL2_gamma_files, DL2_proton_files
      
    def plot_zenith_azimuth_ranges(self, ax=None):
        """
        Plot the zenith and azimuth ranges on a polar plot.
        This method visualizes the zenith and azimuth ranges defined in the 
        `self.validity` attribute. It can plot circles, points, or areas 
        depending on the ranges provided. If no azimuth range is specified, 
        it defaults to a full circle.
        Parameters
        ----------
        ax : matplotlib.axes._subplots.PolarAxesSubplot, optional
            A polar subplot axis to plot on. If None, a new subplot will be created.
        Notes
        -----
        - The method uses `astropy.units` for unit conversions.
        - The method reads training gamma data from an HDF5 file specified by 
          `self.model_index_file` and `self.model_nickname`.
        - The plot style is set using `set_mpl_style()`.
        """

        
        set_mpl_style()
        import matplotlib.pyplot as plt
        import astropy.units as u
        from astropy.io.misc.hdf5 import read_table_hdf5
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        
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
                ax.scatter(azimuth_min, zenith_min, s=100, zorder=0)
            else:
                # Plot a portion of a circle between the azimuth range at the correct zenith
                theta = np.linspace(azimuth_min, azimuth_max, 100)
                r = np.full_like(theta, zenith_min).to(u.deg)
                ax.plot(theta, r, lw=3, zorder=0)
                training_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/training/gamma')
                zeniths = training_gamma_table['training_gamma_zenith_distances']
                azimuths = training_gamma_table['training_gamma_azimuths'].to(u.rad)
                for zenith, azimuth in zip(zeniths, azimuths):
                    ax.scatter(azimuth, zenith, s=50, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
        else:
            if np.isnan(azimuth_min) and np.isnan(azimuth_max):
                # Plot the area between the two circles
                theta = np.linspace(0, 2 * np.pi, 100) * u.rad
                r1 = np.full_like(theta, zenith_min).to(u.deg)
                r2 = np.full_like(theta, zenith_max).to(u.deg)
                ax.fill_between(theta.value, r1.value, r2.value, alpha=0.3, zorder=0)
                ax.plot(theta, r1, lw=3, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], zorder=0)
                ax.plot(theta, r2, lw=3, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], zorder=0)
            else:
                theta = np.linspace(azimuth_min, azimuth_max, 100)
                r1 = np.full_like(theta, zenith_min).to(u.deg).value
                r2 = np.full_like(theta, zenith_max).to(u.deg).value
                theta = theta.value
                ax.fill_between(theta, r1, r2, alpha=0.3, zorder=0)
                ax.plot(theta, r1, lw=3, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], zorder=0)
                ax.plot(theta, r2, lw=3, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], zorder=0)
                ax.plot((theta[0], theta[0]), (r1[0], r2[0]), lw=3, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], zorder=0)
                ax.plot((theta[-1], theta[-1]), (r1[-1], r2[-1]), lw=3, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], zorder=0)
                
                training_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/training/gamma')
                zeniths = training_gamma_table['training_gamma_zenith_distances']
                azimuths = training_gamma_table['training_gamma_azimuths'].to(u.rad)
                for zenith, azimuth in zip(zeniths, azimuths):
                    ax.scatter(azimuth, zenith, s=50, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
        
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(-30)
        ax.set_ylim(0, 60)
        ax.set_yticks(np.arange(10, 61, 10))
        ax.set_yticklabels(["", "", "30°", "", "", "60°"], fontsize=10)
        ax.set_xlabel('Azimuth [deg]', fontsize=10)
        
        
        ax.set_title('Zenith and Azimuth Ranges', pad=30)
        plt.tight_layout()
        if ax is None:
            plt.show()
        
    def plot_training_nodes(self):
        """
        Plot the training nodes for gamma and proton events on a polar plot.
        This method reads the training data for gamma and proton events from HDF5 files,
        extracts the zenith and azimuth angles, and plots them on a polar plot. The gamma
        events are plotted as filled circles, and the proton events are plotted as circles
        with white faces.
        The plot is customized with specific settings for the theta zero location, theta
        direction, radial label position, and y-ticks. A legend is added to distinguish
        between gamma and proton events.
        If no valid zenith or azimuth values are found for either gamma or proton events,
        a message is printed indicating that the training nodes cannot be shown.
        Raises
        ------
        FileNotFoundError
            If the specified HDF5 file does not exist.
        KeyError
            If the specified path within the HDF5 file does not exist.
        """


        from astropy.io.misc.hdf5 import read_table_hdf5
        import matplotlib.pyplot as plt
        import astropy.units as u
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        training_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/training/gamma')
        zeniths = training_gamma_table['training_gamma_zenith_distances']
        azimuths = training_gamma_table['training_gamma_azimuths'].to(u.rad)
        i = 0
        for zenith, azimuth in zip(zeniths, azimuths):
            if (zenith == np.nan) or (azimuth == np.nan):
                continue    
            else:
                ax.scatter(azimuth, zenith, s=50, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], label='Gammas', zorder=2)
                i += 1
        if i == 0:
            print('Training nodes for gammas cannot be shown because the zenith or azimuth are not defined.')
            
        training_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/training/proton')
        zeniths = training_proton_table['training_proton_zenith_distances']
        azimuths = training_proton_table['training_proton_azimuths'].to(u.rad)
        i = 0
        for zenith, azimuth in zip(zeniths, azimuths):
            if (zenith == np.nan) or (azimuth == np.nan):
                continue
            else:
                ax.scatter(azimuth, zenith, label='Protons', edgecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], facecolors='w', zorder=1, s=100, lw=2)
                i += 1
        if i == 0:
            print('Training nodes for protons cannot be shown because the zenith or azimuth are not defined.')  
            
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(-30)
        ax.set_ylim(0, 60)
        ax.set_yticks(np.arange(10, 61, 10))
        ax.set_yticklabels(["", "", "30°", "", "", "60°"], fontsize=10)
        ax.set_xlabel('Azimuth [deg]', fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        
        ax.set_title('Training nodes', pad=30)
        plt.tight_layout()
        plt.show()
      


        

class ModelRangeOfValidity:
    """
    A class to determine the range of validity for a given model based on its training data.
    :param model_manager: An instance of CTLearnModelManager containing model information.
    :type model_manager: CTLearnModelManager
    Attributes
    ----------
    zenith_range : astropy.units.Quantity
        The range of zenith distances covered by the training data.
    azimuth_range : astropy.units.Quantity
        The range of azimuth angles covered by the training data.
    energy_range : astropy.units.Quantity
        The range of energies covered by the training data.
    nsb_range : astropy.units.Quantity
        The range of night sky background (NSB) levels covered by the training data.
    Methods
    -------
    matches(**kwargs)
        Checks if the given parameters fall within the ranges of validity.
    """

    def __init__(self, model_manager: CTLearnModelManager):
        """
        Initialize the ModelManager with the given CTLearnModelManager instance.
        This method reads the training gamma data from an HDF5 file and extracts
        the zenith, azimuth, energy, and NSB ranges for the model.
        :param model_manager: An instance of CTLearnModelManager containing the
                              model index file and model nickname.
        :type model_manager: CTLearnModelManager
        :ivar zenith_range: The range of zenith distances in the training gamma data.
        :vartype zenith_range: astropy.units.Quantity
        :ivar azimuth_range: The range of azimuths in the training gamma data.
        :vartype azimuth_range: astropy.units.Quantity
        :ivar energy_range: The range of energies in the training gamma data.
        :vartype energy_range: astropy.units.Quantity
        :ivar nsb_range: The range of NSB values in the training gamma data.
        :vartype nsb_range: astropy.units.Quantity
        """

        from astropy.io.misc.hdf5 import read_table_hdf5
        training_gamma_table = read_table_hdf5(model_manager.model_index_file, path=f'{model_manager.model_nickname}/training/gamma')
        # training_proton_table = read_table_hdf5(model_manager.model_index_file, path=f'{model_manager.model_nickname}/training/proton')
        
        
        training_gamma_zeniths = training_gamma_table['training_gamma_zenith_distances']
        self.zenith_range = [min(training_gamma_zeniths).value, max(training_gamma_zeniths).value] * training_gamma_zeniths.unit
        training_gamma_azimuths = training_gamma_table['training_gamma_azimuths']
        self.azimuth_range = [min(training_gamma_azimuths.value), max(training_gamma_azimuths).value] * training_gamma_azimuths.unit
        training_gamma_energies_mins = training_gamma_table['training_gamma_energy_min']
        training_gamma_energies_maxs = training_gamma_table['training_gamma_energy_max']
        taining_gamma_energies = np.concatenate((training_gamma_energies_mins, training_gamma_energies_maxs))
        self.energy_range = [min(taining_gamma_energies).value, max(taining_gamma_energies).value] * taining_gamma_energies.unit
        training_gamma_nsbs_mins = training_gamma_table['training_gamma_nsb_min']
        training_gamma_nsbs_maxs = training_gamma_table['training_gamma_nsb_max']
        taining_gamma_nsbs = np.concatenate((training_gamma_nsbs_mins, training_gamma_nsbs_maxs))
        self.nsb_range = [min(taining_gamma_nsbs).value, max(taining_gamma_nsbs).value] * taining_gamma_nsbs.unit
        

    def matches(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'zenith':
                if not (self.zenith_range[0] <= value <= self.zenith_range[1]):
                    return False
            elif key == 'azimuth':
                if self.azimuth_range is not None and not (self.azimuth_range[0] <= value <= self.azimuth_range[1]):
                    return False
            elif key == 'energy':
                if self.energy_range is not None and not (self.energy_range[0] <= value <= self.energy_range[1]):
                    return False
            elif key == 'nsb':
                if self.nsb_range is not None and not (self.nsb_range[0] <= value <= self.nsb_range[1]):
                    return False
        return True
    
