"""CTLearn Model Manager"""
from .version import __version__
from astropy.table import QTable
import numpy as np
from pathlib import Path
# from . import get_predict_data_sbaych_script
__all__ = [
    "__version__",
]


class CTLearnModelManager():
    """
    A class to manage CTLearn models, including initialization, saving to index, launching training, and plotting loss.
    Attributes:
        model_index_file (str): Path to the model index file.
        model_nickname (str): Nickname of the model.
        notes (str): Notes about the model.
        model_dir (str): Directory where the model is stored.
        reco (str): Type of reconstruction.
        telescope_names (list): Names of the telescopes.
        telescopes_indices (list): Indices of the telescopes.
        training_gamma_dir (str): Directory of training gamma data.
        training_proton_dir (str): Directory of training proton data.
        training_gamma_patterns (list): Patterns of training gamma data.
        training_proton_patterns (list): Patterns of training proton data.
        training_gamma_zenith_distances (list): Zenith distances of training gamma data.
        training_gamma_azimuths (list): Azimuths of training gamma data.
        training_proton_zenith_distances (list): Zenith distances of training proton data.
        training_proton_azimuths (list): Azimuths of training proton data.
        testing_gamma_dirs (list): Directories of testing gamma data.
        testing_proton_dirs (list): Directories of testing proton data.
        testing_gamma_zenith_distances (list): Zenith distances of testing gamma data.
        testing_gamma_azimuths (list): Azimuths of testing gamma data.
        testing_proton_zenith_distances (list): Zenith distances of testing proton data.
        testing_proton_azimuths (list): Azimuths of testing proton data.
        testing_DL2_gamma_files (list): DL2 gamma files for testing.
        testing_DL2_proton_files (list): DL2 proton files for testing.
        channels (list): Channels used in the model.
        max_training_epochs (int): Maximum number of training epochs.
        stereo (bool): Whether the model is stereo or not.
        zd_range (list): Zenith distance range.
        az_range (list): Azimuth range.
        model_name (str): Name of the model.
        model_index (int): Index of the model.
    Methods:
        __init__(model_parameters, MODEL_INDEX_FILE, load=False):
            Initializes the CTLearnModelManager with the given parameters.
        save_to_index():
            Saves the model parameters to the index file.
        launch_training(n_epochs):
            Launches the training process for the model.
        get_n_epoch_trained():
            Returns the number of epochs the model has been trained for.
        plot_loss():
            Plots the training and validation loss over epochs.
        info():
            Prints information about the model.
        update_model_manager_parameters_in_index(parameters):
            Updates the model parameters in the index file.
        update_model_manager_training_data(training_gamma_dir, training_proton_dir, training_gamma_patterns, training_proton_patterns, training_gamma_zenith_distances, training_gamma_azimuths, training_proton_zenith_distances, training_proton_azimuths):
            Updates the training data in the model manager.
        update_model_manager_testing_data(testing_gamma_dirs, testing_proton_dirs, testing_gamma_zenith_distances, testing_gamma_azimuths, testing_proton_zenith_distances, testing_proton_azimuths):
            Updates the testing data in the model manager.
        update_model_manager_DL2_MC_files(testing_DL2_gamma_files, testing_DL2_proton_files, testing_DL2_gamma_zenith_distances, testing_DL2_gamma_azimuths, testing_DL2_proton_zenith_distances, testing_DL2_proton_azimuths):
            Updates the DL2 MC files in the model manager.
        update_model_manager_IRF_data(config, cuts_file, irf_file, zenith, azimuth):
            Updates the IRF data in the model manager.
        get_IRF_data(zenith, azimuth):
            Retrieves the IRF data for the given zenith and azimuth.
        get_DL2_MC_files(zenith, azimuth):
            Retrieves the DL2 MC files for the given zenith and azimuth.
    """
    def __init__(self, model_parameters, MODEL_INDEX_FILE, load=False):
        from astropy.io.misc.hdf5 import read_table_hdf5
        self.model_index_file = MODEL_INDEX_FILE
        
        if not load:
            # Model parameters
            self.model_nickname = model_parameters.get('model_nickname', 'new_model')
            self.notes = model_parameters.get('notes', '')
            self.model_dir = model_parameters.get('model_dir', '')
            self.reco = model_parameters.get('reco', 'default_reco')
            self.telescope_names = model_parameters.get('telescope_names', [])
            self.telescopes_indices = model_parameters.get('telescopes_indices', [])
            self.channels = model_parameters.get('channels', ['cleaned_image', 'cleaned_relative_peak_time'])
            self.max_training_epochs = model_parameters.get('max_training_epochs', 10)
            
            # Training tables
            self.training_gamma_dir = model_parameters.get('training_gamma_dir', "")
            self.training_proton_dir = model_parameters.get('training_proton_dir', "")
            self.training_gamma_patterns = model_parameters.get('training_gamma_patterns', [])
            self.training_proton_patterns = model_parameters.get('training_proton_patterns', [])
            self.training_gamma_zenith_distances = model_parameters.get('training_gamma_zenith_distances', [])
            self.training_gamma_azimuths = model_parameters.get('training_gamma_azimuths', [])
            self.training_proton_zenith_distances = model_parameters.get('training_proton_zenith_distances', [])
            self.training_proton_azimuths = model_parameters.get('training_proton_azimuths', [])
            
            # Testing tables
            self.testing_gamma_dirs = model_parameters.get('testing_gamma_dirs', [])
            self.testing_proton_dirs = model_parameters.get('testing_proton_dirs', [])
            self.testing_gamma_zenith_distances = model_parameters.get('testing_gamma_zenith_distances', [])
            self.testing_gamma_azimuths = model_parameters.get('testing_gamma_azimuths', [])
            self.testing_proton_zenith_distances = model_parameters.get('testing_proton_zenith_distances', [])
            self.testing_proton_azimuths = model_parameters.get('testing_proton_azimuths', [])
            
            # DL2 tables
            self.testing_DL2_gamma_files = model_parameters.get('testing_DL2_gamma_files', [])
            self.testing_DL2_proton_files = model_parameters.get('testing_DL2_proton_files', [])
        else:
            import ast
            self.model_nickname = model_parameters.get('model_nickname', 'new_model')
            model_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/parameters')
            for key in model_table.colnames:
                self.__dict__[key] = model_table[key][0]
            self.channels = ast.literal_eval(model_table['channels'][0])
            self.telescopes_indices = ast.literal_eval(model_table['telescopes_indices'][0])
            self.telescope_names = ast.literal_eval(model_table['telescope_names'][0])
            training_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/training/gamma')
            for key in training_gamma_table.colnames:
                self.__dict__[key] = training_gamma_table[key]
            training_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/training/proton')
            for key in training_proton_table.colnames:
                self.__dict__[key] = training_proton_table[key]
            testing_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/testing/gamma')
            for key in testing_gamma_table.colnames:
                self.__dict__[key] = testing_gamma_table[key]
            testing_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/testing/proton')
            for key in testing_proton_table.colnames:
                self.__dict__[key] = testing_proton_table[key]
            try:
                DL2_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/gamma')
                for key in DL2_gamma_table.colnames:
                    self.__dict__[key] = DL2_gamma_table[key]
                DL2_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/proton')
                for key in DL2_proton_table.colnames:
                    self.__dict__[key] = DL2_proton_table[key]
            except:
                print("No DL2 MC files yet")
        self.model_index = 0
        self.stereo = len(self.telescopes_indices) > 1
        if self.reco == 'type' and (len(self.training_proton_patterns) == 0 or len(self.training_gamma_patterns) == 0):
            raise ValueError("For reco type, training_proton_patterns and training_gamma_patterns are required")
        # Check that all gamma related lists are the same length
        gamma_lengths = [len(self.training_gamma_patterns), len(self.training_gamma_zenith_distances), len(self.training_gamma_azimuths)]
        if len(set(gamma_lengths)) != 1:
            raise ValueError("All gamma related lists must be the same length")

        # Check that all proton related lists are the same length
        proton_lengths = [len(self.training_proton_patterns), len(self.training_proton_zenith_distances), len(self.training_proton_azimuths)]
        if len(set(proton_lengths)) != 1:
            raise ValueError("All proton related lists must be the same length")
        
        # Model parameters
        self.zd_range = [min(self.training_gamma_zenith_distances), max(self.training_gamma_zenith_distances)]
        self.az_range = [min(self.training_gamma_azimuths), max(self.training_gamma_azimuths)]
        self.model_name = f"{self.reco}_TEL{'_'.join(map(str, self.telescopes_indices))}_ZD{'_'.join(map(str, self.training_gamma_zenith_distances))}_Az{'_'.join(map(str, self.training_gamma_azimuths))}"
        print(f"🧠 Model name: {self.model_nickname}")
        
        
    def save_to_index(self):
        """
        Save the current model information to the index file.
        This method attempts to read an existing model index file and append the current model's
        information to it. If the index file does not exist, it creates a new one with the necessary
        columns and data types. The model information is only added if the model nickname is not
        already present in the index.
        Raises:
            Exception: If there is an error reading or writing the model index file.
        Notes:
            - The model index file is expected to be in the 'ascii.ecsv' format.
            - The method ensures that the model nickname is unique in the index.
            - The model information includes various attributes such as model nickname, name, directory,
              reconstruction type, channels, telescope names and indices, training directories, zenith
              distances, azimuths, notes, and maximum training epochs.
        """
        from astropy.io.misc.hdf5 import write_table_hdf5
        try:
            model_table = QTable.read(self.model_index_file, format='hdf5', path=f'{self.model_nickname}/parameters')
            print(f"❌ Model nickname {self.model_nickname} already in table")
        except:
            model_table = QTable(names=['model_index', 'model_nickname', 'model_name', 'model_dir', 'reco', 'channels', 'telescope_names', 'telescopes_indices', 'notes', 'zd_range', 'az_range', 'max_training_epochs'], dtype=[int, str, str, str, str, list, list, list, str, list, list, int])
            
            model_table = QTable(names=['model_index', 'model_nickname', 'model_name', 'model_dir', 'reco', 'channels', 'telescope_names', 'telescopes_indices', 'notes', 'zd_range', 'az_range', 'max_training_epochs'],
                        dtype=[int, str, str, str, str, str, str, str, str, str, str, int])
            training_table_gamma = QTable(names=['model_index', 'training_gamma_dir', 'training_gamma_patterns', 'training_gamma_zenith_distances', 'training_gamma_azimuths'],
                            dtype=[int, str, str, float, float])
            training_table_proton = QTable(names=['model_index', 'training_proton_dir', 'training_proton_patterns', 'training_proton_zenith_distances', 'training_proton_azimuths'],
                            dtype=[int, str, str, float, float])
            testing_table_gamma = QTable(names=['model_index', 'testing_gamma_dirs', 'testing_gamma_zenith_distances', 'testing_gamma_azimuths'],
                            dtype=[int, str, float, float])
            testing_table_proton = QTable(names=['model_index', 'testing_proton_dirs', 'testing_proton_zenith_distances', 'testing_proton_azimuths'],
                            dtype=[int, str, float, float])
            
            model_table.add_row([self.model_index, self.model_nickname, self.model_name, self.model_dir, self.reco, str(self.channels), str(self.telescope_names), str(self.telescopes_indices), self.notes, str(self.zd_range), str(self.az_range), self.max_training_epochs])
            for i in range(len(self.training_gamma_patterns)):
                training_table_gamma.add_row([self.model_index, self.training_gamma_dir, self.training_proton_patterns[i], self.training_gamma_zenith_distances[i], self.training_gamma_azimuths[i]])
            for i in range(len(self.training_proton_patterns)):
                training_table_proton.add_row([self.model_index, self.training_proton_dir, self.training_proton_patterns[i], self.training_proton_zenith_distances[i], self.training_proton_azimuths[i]])
            for i in range(len(self.testing_gamma_dirs)):
                testing_table_gamma.add_row([self.model_index, self.testing_gamma_dirs[i], self.testing_gamma_zenith_distances[i], self.testing_gamma_azimuths[i]])
            for i in range(len(self.testing_proton_dirs)):
                testing_table_proton.add_row([self.model_index, self.testing_proton_dirs[i], self.testing_proton_zenith_distances[i], self.testing_proton_azimuths[i]])
            
            write_table_hdf5(training_table_gamma, self.model_index_file, path=f'{self.model_nickname}/training/gamma', append=True, overwrite=True)
            write_table_hdf5(training_table_proton, self.model_index_file, path=f'{self.model_nickname}/training/proton', append=True, overwrite=True)
            write_table_hdf5(testing_table_gamma, self.model_index_file, path=f'{self.model_nickname}/testing/gamma', append=True, overwrite=True)
            write_table_hdf5(testing_table_proton, self.model_index_file, path=f'{self.model_nickname}/testing/proton', append=True, overwrite=True)
            write_table_hdf5(model_table, self.model_index_file, path=f'{self.model_nickname}/parameters',append=True, overwrite=True)
            
            print(f"✅ Model nickname {self.model_nickname} added to table")
            
        #     print(f"❌ Model nickname {self.model_nickname} already in table")
        
        
    def launch_training(self, n_epochs, transfer_learning_model_cpk=None, frozen_backbone=False):
        """
        Launches the training process for the model.
        Parameters:
        n_epochs (int, optional): The number of epochs to train the model. If not specified, the training will continue 
                                  until the maximum number of training epochs is reached.
        Returns:
        None
        This method performs the following steps:
        1. Checks the number of epochs the model has already been trained for.
        2. If the specified number of epochs exceeds the maximum allowed, updates the maximum training epochs.
        3. If the model has already been trained for the maximum number of epochs, it stops further training.
        4. Determines the number of epochs to train in this session.
        5. Checks if a model already exists and determines whether to continue training the existing model or create a new one.
        6. Constructs the command to launch the training process with the appropriate parameters.
        7. Executes the training command.
        Note:
        - The method assumes that the model directory and nickname are already set.
        - The method uses the `ctlearn-train-model` command to launch the training process.
        - The method prints the constructed command and executes it using `os.system`.
        """
        import yaml
        n_epoch_training = self.get_n_epoch_trained()
        print(f"📊 Model trained for {n_epoch_training} epochs")
        if n_epochs > self.max_training_epochs:
            print(f"⚠️ Number of epochs increased from {self.max_training_epochs} to {n_epochs}")
            self.update_model_manager_parameters_in_index({'max_training_epochs': n_epochs})
            self.max_training_epochs = n_epochs
        if n_epoch_training >= self.max_training_epochs:
            print(f"🛑 Model already trained for {n_epoch_training} epochs. Will not train further.")
            self.plot_loss()
            return
        n_epochs = self.max_training_epochs - n_epoch_training
        print(f"🚀 Launching training for {n_epochs} epochs")
        import glob
        import os
        models_dir = np.sort(glob.glob(f"{self.model_dir}/{self.model_nickname}*"))
        load_model = False
        if len(models_dir) > 0 :
            last_model_dir = Path(models_dir[-1])
            size = sum(f.stat().st_size for f in last_model_dir.glob('**/*') if f.is_file())
            model_version = int(models_dir[-1].split("_v")[-1])
            if size > 1e6:
                model_version += 1
                print(f"➡️ Model already exists: will continue training and create {self.model_nickname}_v{model_version}")
                save_best_validation_only = True
                model_dir = f"{self.model_dir}/{self.model_nickname}_v{model_version}/"
                model_to_load = f"{self.model_dir}/{self.model_nickname}_v{model_version - 1}/ctlearn_model.cpk/"
                load_model = True
                os.system(f"mkdir -p {model_dir}")
            else :
                model_dir = f"{self.model_dir}/{self.model_nickname}_v{model_version}/"
                if model_version > 0:
                    model_to_load = f"{self.model_dir}/{self.model_nickname}_v{model_version - 1}/ctlearn_model.cpk/"
                    load_model = True
                    print(f"➡️ Model already exists: will continue training and create {self.model_nickname}_v{model_version}")
                    save_best_validation_only = True
                else:
                    print(f"🆕 Model does not exist: will create {self.model_nickname}_v{model_version}")
                    save_best_validation_only = False
        else:
            model_version = 0
            print(f"🆕 Model does not exist: will create {self.model_nickname}_v{model_version}")
            model_dir = f"{self.model_dir}/{self.model_nickname}_v{model_version}/"
            os.system(f"mkdir -p {model_dir}")
            save_best_validation_only = False

        if load_model:
            load_model_string = f"--TrainCTLearnModel.model_type=LoadedModel --LoadedModel.load_model_from={model_to_load} "
        else:
            load_model_string = "" if transfer_learning_model_cpk is None else f"--TrainCTLearnModel.model_type=LoadedModel --LoadedModel.load_model_from={transfer_learning_model_cpk} "
        background_string = f"--background {self.training_proton_dir} " if self.reco == 'type' else ""
        signal_patterns = ""
        for pattern in self.training_gamma_patterns:
            signal_patterns += f'--pattern-signal "{pattern}" '
        background_patterns = ""
        for pattern in self.training_proton_patterns:
            background_patterns += f'--pattern-background "{pattern}" '
        channels_string = ""
        for channel in self.channels:
            channels_string += f"--DLImageReader.channels={channel} "

        stereo_mode = 'stereo' if self.stereo else "mono"
        stack_telescope_images = 'true' if self.stereo else 'false'
        min_telescopes = 2 if self.stereo else 1
        allowed_tels = '_'.join(map(str, self.telescopes_indices)) if self.stereo else int(self.telescopes_indices[0])
        
        config = {
            'load_model_string': load_model_string,
            'background_string': background_string,
            'signal_patterns': signal_patterns,
            'background_patterns': background_patterns,
            'channels_string': channels_string,
            'stereo_mode': stereo_mode,
            'stack_telescope_images': stack_telescope_images,
            'min_telescopes': min_telescopes,
            'allowed_tels': allowed_tels,
            'n_epochs': n_epochs,
            'model_dir': model_dir,
            'save_best_validation_only': save_best_validation_only
        }

        config_file = f"{model_dir}/config.yml"
        with open(config_file, 'w') as file:
            yaml.dump(config, file)
        print(f"Configuration saved to {config_file}")
        
        #FIXME take in account all agga or proton dirs
        cmd = f"ctlearn-train-model {load_model_string} \
--TrainCTLearnModel.batch_size=64 \
--signal {self.training_gamma_dir} {signal_patterns} \
{background_string}{background_patterns}\
--reco {self.reco} \
--output {model_dir} \
{channels_string}\
--TrainCTLearnModel.n_epochs={n_epochs} \
--verbose \
--TrainCTLearnModel.save_best_validation_only=True\
--overwrite \
--DLImageReader.mode={stereo_mode} \
--TrainCTLearnModel.stack_telescope_images={stack_telescope_images}\
--DLImageReader.min_telescopes={min_telescopes}"# \
#--DLImageReader.allowed_tels={allowed_tels}"
        print(cmd)
        # !{cmd}
        os.system(cmd)
        
        
    def get_n_epoch_trained(self):
        """
        Calculate the total number of epochs trained across all training logs.
        This method searches for all training log files in the model directory,
        reads each log file, and sums the number of epochs recorded in each log.
        Returns:
            int: The total number of epochs trained.
        """
        
        import glob
        import pandas as pd
        training_logs = np.sort(glob.glob(f"{self.model_dir}/{self.model_nickname}_v*/training_log.csv"))
        n_epochs = 0
        for training_log in training_logs:
            df = pd.read_csv(training_log)
            n_epochs += len(df)
        return n_epochs

    
    def plot_loss(self):
        """
        Plots the training and validation loss over epochs.
        This method reads training logs from CSV files located in the model directory,
        concatenates the loss values, and plots them using matplotlib. The plot includes
        both training and validation loss curves.
        Dependencies:
            - matplotlib.pyplot
            - pandas
            - glob
            - numpy
        Raises:
            FileNotFoundError: If no training log files are found in the specified directory.
        Example:
            self.plot_loss()
        """
        
        import matplotlib.pyplot as plt
        import pandas as pd
        import glob
        set_mpl_style()
        training_logs = np.sort(glob.glob(f"{self.model_dir}/{self.model_nickname}_v*/training_log.csv"))
        losses_train = []
        losses_val = []
        for training_log in training_logs:
            df = pd.read_csv(training_log)
            losses_train = np.concatenate((losses_train, df['loss'].to_numpy()))
            losses_val = np.concatenate((losses_val, df['val_loss'].to_numpy()))
        epochs = np.arange(1, len(losses_train)+1)
        plt.plot(epochs, losses_train, label=f"Training", lw=2)
        plt.plot(epochs, losses_val, label=f"Validation", ls='--')
        plt.title(f"{self.reco} training".title())
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(np.arange(1, len(losses_train) + 1, 2))
        plt.legend()
        plt.show()
    
            
    def info(self):
        print(f"Model nickname: {self.model_nickname}")
        print(f"Model name: {self.model_name}")
        print(f"Model directory: {self.model_dir}")
        print(f"Reco: {self.reco}")
        print(f"Channels: {self.channels}")
        print(f"Telescope names: {self.telescope_names}")
        print(f"Telescope indices: {self.telescopes_indices}")
        print(f"Training gamma dir: {self.training_gamma_dir}")
        print(f"Training proton dir: {self.training_proton_dir}")
        print(f"Training gamma zenith distances: {self.training_gamma_zenith_distances}")
        print(f"Training gamma azimuths: {self.training_gamma_azimuths}")
        print(f"Training proton zenith distances: {self.training_proton_zenith_distances}")
        print(f"Training proton azimuths: {self.training_proton_azimuths}")
        print(f"Notes: {self.notes}")
        print(f"ZD range: {self.zd_range}")
        print(f"Az range: {self.az_range}")
        print(f"Stereo: {self.stereo}")
        print(f"Max training epochs: {self.max_training_epochs}")
        print(f"Model index: {self.model_index}")
        print(f"Testing gamma dirs: {self.testing_gamma_dirs}")
        print(f"Testing proton dirs: {self.testing_proton_dirs}")
        
        
    def update_model_manager_parameters_in_index(self, parameters: dict):
        """
        Updates the model manager parameters in the index file.
        This method reads the model index file, finds the entry corresponding to the
        current model nickname, and updates the specified parameters in the index.
        The updated parameters are also reflected in the instance's attributes.
        Args:
            parameters (dict): A dictionary containing the parameters to update and their new values.
        Raises:
            IndexError: If the model nickname is not found in the model index file.
        """
        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
        
        model_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/parameters')
        # model_index = np.where(model_table['model_nickname'] == self.model_nickname)[0][0]
        print(f"💾 Model {self.model_nickname} index update:")
        for key, value in parameters.items():
            model_table[key][0] = value
            self.__dict__[key] = value
            print(f"\t➡️ {key} updated to {value}")
        write_table_hdf5(model_table, self.model_index_file, path=f'{self.model_nickname}/parameters', append=True, overwrite=True)
        
    def update_model_manager_training_data(self, training_gamma_dir, training_proton_dir, training_gamma_patterns, training_proton_patterns, training_gamma_zenith_distances, training_gamma_azimuths, training_proton_zenith_distances, training_proton_azimuths):

        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
        
        training_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/training/gamma')
        training_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/training/proton')
        print(f"💾 Model {self.model_nickname} training data update:")
        if len(training_gamma_table)==0:
            training_gamma_table = QTable(names=['model_index', 'training_gamma_dir', 'training_gamma_patterns', 'training_gamma_zenith_distances', 'training_gamma_azimuths'], 
                                          dtype=[int, str, str, float, float])
        if len(training_proton_table)==0:
            training_proton_table = QTable(names=['model_index', 'training_proton_dir', 'training_proton_patterns', 'training_proton_zenith_distances', 'training_proton_azimuths'], 
                                           dtype=[int, str, str, float, float])
        
        if len(training_gamma_patterns) > 0:
            for i in range(len(training_gamma_patterns)):
                match = np.where((training_gamma_table['training_gamma_zenith_distances'] == training_gamma_zenith_distances[i]) & 
                     (training_gamma_table['training_gamma_azimuths'] == training_gamma_azimuths[i]))[0]
                if len(match) > 0:
                    training_gamma_table['training_gamma_dir'][match[0]] = training_gamma_dir
                    training_gamma_table['training_gamma_patterns'][match[0]] = training_gamma_patterns[i]
                else:
                    training_gamma_table.add_row([self.model_index, training_gamma_dir, training_proton_patterns[i], training_gamma_zenith_distances[i], training_gamma_azimuths[i]])
            write_table_hdf5(training_gamma_table, self.model_index_file, path=f'{self.model_nickname}/training/gamma', append=True, overwrite=True)
            print(f"\t➡️ Training gamma data updated")
        
        if len(training_proton_patterns) > 0:
            for i in range(len(training_proton_patterns)):
                match = np.where((training_proton_table['training_proton_zenith_distances'] == training_proton_zenith_distances[i]) & 
                     (training_proton_table['training_proton_azimuths'] == training_proton_azimuths[i]))[0]
                if len(match) > 0:
                    training_proton_table['training_proton_dir'][match[0]] = training_proton_dir
                    training_proton_table['training_proton_patterns'][match[0]] = training_proton_patterns[i]
                else:
                    training_proton_table.add_row([self.model_index, training_proton_dir, training_proton_patterns[i], training_proton_zenith_distances[i], training_proton_azimuths[i]])
            write_table_hdf5(training_proton_table, self.model_index_file, path=f'{self.model_nickname}/training/proton', append=True, overwrite=True)
            print(f"\t➡️ Training proton data updated")
        
        self.training_gamma_dir = training_gamma_table['training_gamma_dir']
        self.training_proton_dir = training_proton_table['training_proton_dir']
        self.training_gamma_patterns = training_gamma_table['training_gamma_patterns']
        self.training_proton_patterns = training_proton_table['training_proton_patterns']
        self.training_gamma_zenith_distances = training_gamma_table['training_gamma_zenith_distances']
        self.training_gamma_azimuths = training_gamma_table['training_gamma_azimuths']
        self.training_proton_zenith_distances = training_proton_table['training_proton_zenith_distances']
        self.training_proton_azimuths = training_proton_table['training_proton_azimuths']
            

    def update_model_manager_testing_data(self, testing_gamma_dirs, testing_proton_dirs, testing_gamma_zenith_distances, testing_gamma_azimuths, testing_proton_zenith_distances, testing_proton_azimuths):
        """
        Updates the model manager testing data in the index file.
        This method reads the model index file, finds the entry corresponding to the
        current model nickname, and updates the testing data in the index.
        The updated testing data are also reflected in the instance's attributes.
        Args:
            testing_gamma_dirs (list): Directories of testing gamma data.
            testing_proton_dirs (list): Directories of testing proton data.
            testing_gamma_zenith_distances (list): Zenith distances of testing gamma data.
            testing_gamma_azimuths (list): Azimuths of testing gamma data.
            testing_proton_zenith_distances (list): Zenith distances of testing proton data.
            testing_proton_azimuths (list): Azimuths of testing proton data.
        Raises:
            IndexError: If the model nickname is not found in the model index file.
        """
        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
        
        testing_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/testing/gamma')
        testing_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/testing/proton')
        print(f"💾 Model {self.model_nickname} testing data update:")
        if len(testing_gamma_table)==0:
            testing_gamma_table = QTable(names=['model_index', 'testing_gamma_dirs', 'testing_gamma_zenith_distances', 'testing_gamma_azimuths'], dtype=[int, str, float, float])
        if len(testing_proton_table)==0:
            testing_proton_table = QTable(names=['model_index', 'testing_proton_dirs', 'testing_proton_zenith_distances', 'testing_proton_azimuths'], dtype=[int, str, float, float])
        
        if len(testing_gamma_dirs) > 0:
            for i in range(len(testing_gamma_dirs)):
                match = np.where((testing_gamma_table['testing_gamma_zenith_distances'] == testing_gamma_zenith_distances[i]) & 
                        (testing_gamma_table['testing_gamma_azimuths'] == testing_gamma_azimuths[i]))[0]
                if len(match) > 0:
                    testing_gamma_table['testing_gamma_dirs'][match[0]] = testing_gamma_dirs[i]
                else:
                    testing_gamma_table.add_row([self.model_index, testing_gamma_dirs[i], testing_gamma_zenith_distances[i], testing_gamma_azimuths[i]])
            write_table_hdf5(testing_gamma_table, self.model_index_file, path=f'{self.model_nickname}/testing/gamma', append=True, overwrite=True)
            print(f"\t➡️ Testing gamma data updated")
        
        if len(testing_proton_dirs) > 0:
            for i in range(len(testing_proton_dirs)):
                match = np.where((testing_proton_table['testing_proton_zenith_distances'] == testing_proton_zenith_distances[i]) & 
                        (testing_proton_table['testing_proton_azimuths'] == testing_proton_azimuths[i]))[0]
                if len(match) > 0:
                    testing_proton_table['testing_proton_dirs'][match[0]] = testing_proton_dirs[i]
                else:
                    testing_proton_table.add_row([self.model_index, testing_proton_dirs[i], testing_proton_zenith_distances[i], testing_proton_azimuths[i]])
            write_table_hdf5(testing_proton_table, self.model_index_file, path=f'{self.model_nickname}/testing/proton', append=True, overwrite=True)
            print(f"\t➡️ Testing proton data updated")
            
        self.testing_gamma_dirs = testing_gamma_table['testing_gamma_dirs']
        self.testing_proton_dirs = testing_proton_table['testing_proton_dirs']
        self.testing_gamma_zenith_distances = testing_gamma_table['testing_gamma_zenith_distances']
        self.testing_gamma_azimuths = testing_gamma_table['testing_gamma_azimuths']
        self.testing_proton_zenith_distances = testing_proton_table['testing_proton_zenith_distances']
        self.testing_proton_azimuths = testing_proton_table['testing_proton_azimuths']
            
    def update_model_manager_DL2_MC_files(self, testing_DL2_gamma_files, testing_DL2_proton_files, testing_DL2_gamma_zenith_distances, testing_DL2_gamma_azimuths, testing_DL2_proton_zenith_distances, testing_DL2_proton_azimuths):
        """
        Updates the model manager testing data in the index file.
        This method reads the model index file, finds the entry corresponding to the
        current model nickname, and updates the testing data in the index.
        The updated testing data are also reflected in the instance's attributes.
        Args:
            testing_gamma_dirs (list): Directories of testing gamma data.
            testing_proton_dirs (list): Directories of testing proton data.
            testing_gamma_zenith_distances (list): Zenith distances of testing gamma data.
            testing_gamma_azimuths (list): Azimuths of testing gamma data.
            testing_proton_zenith_distances (list): Zenith distances of testing proton data.
            testing_proton_azimuths (list): Azimuths of testing proton data.
        Raises:
            IndexError: If the model nickname is not found in the model index file.
        """
        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
        
        try:
            DL2_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/gamma')
        except:
            DL2_gamma_table = QTable(names=['model_index', 'testing_DL2_gamma_files', 'testing_DL2_gamma_zenith_distances', 'testing_DL2_gamma_azimuths'], dtype=[int, str, float, float])
            # model_index = 0
            # DL2_gamma_table.add_row([model_index, testing_DL2_gamma_files, testing_gamma_zenith_distances, testing_gamma_azimuths])
            # write_table_hdf5(DL2_gamma_table, self.model_index_file, path=f'{self.model_nickname}/parameters', append=True, overwrite=True)
        try:
            DL2_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/proton')
        except:
            DL2_proton_table = QTable(names=['model_index', 'testing_DL2_proton_files', 'testing_DL2_proton_zenith_distances', 'testing_DL2_proton_azimuths'], dtype=[int, str, float, float])
            # model_index = 0
            # DL2_proton_table.add_row([model_index, testing_DL2_proton_files, testing_proton_zenith_distances, testing_proton_azimuths])
            # write_table_hdf5(DL2_proton_table, self.model_index_file, path=f'{self.model_nickname}/parameters', append=True, overwrite=True)
        print(f"💾 Model {self.model_nickname} DL2 data update:")
        if len(DL2_gamma_table)==0:
            DL2_gamma_table = QTable(names=['model_index', 'testing_DL2_gamma_files', 'testing_DL2_gamma_zenith_distances', 'testing_DL2_gamma_azimuths'], dtype=[int, str, float, float])
        if len(DL2_proton_table)==0:
            DL2_proton_table = QTable(names=['model_index', 'testing_DL2_proton_files', 'testing_DL2_proton_zenith_distances', 'testing_DL2_proton_azimuths'], dtype=[int, str, float, float])
        
        if len(testing_DL2_gamma_files) > 0:
            for i in range(len(testing_DL2_gamma_files)):
                match = np.where((DL2_gamma_table['testing_DL2_gamma_files'] == testing_DL2_gamma_files[i]) & 
                        (DL2_gamma_table['testing_DL2_gamma_zenith_distances'] == testing_DL2_gamma_zenith_distances[i]) & 
                        (DL2_gamma_table['testing_DL2_gamma_azimuths'] == testing_DL2_gamma_azimuths[i]))[0]
                if len(match) == 0:
                    DL2_gamma_table.add_row([self.model_index, testing_DL2_gamma_files[i], testing_DL2_gamma_zenith_distances[i], testing_DL2_gamma_azimuths[i]])
            write_table_hdf5(DL2_gamma_table, self.model_index_file, path=f'{self.model_nickname}/DL2/MC/gamma', append=True, overwrite=True)
            print(f"\t➡️ Testing DL2 gamma data updated")
        
        if len(testing_DL2_proton_files) > 0:
            for i in range(len(testing_DL2_proton_files)):
                match = np.where((DL2_proton_table['testing_DL2_proton_files'] == testing_DL2_proton_files[i]) & 
                        (DL2_proton_table['testing_DL2_proton_zenith_distances'] == testing_DL2_proton_zenith_distances[i]) & 
                        (DL2_proton_table['testing_DL2_proton_azimuths'] == testing_DL2_proton_azimuths[i]))[0]
                if len(match) == 0:
                    DL2_proton_table.add_row([self.model_index, testing_DL2_proton_files[i], testing_DL2_proton_zenith_distances[i], testing_DL2_proton_azimuths[i]])
            write_table_hdf5(DL2_proton_table, self.model_index_file, path=f'{self.model_nickname}/DL2/MC/proton', append=True, overwrite=True)
            print(f"\t➡️ Testing DL2 proton data updated")
        
        self.testing_DL2_gamma_files = DL2_gamma_table['testing_DL2_gamma_files']
        self.testing_DL2_proton_files = DL2_proton_table['testing_DL2_proton_files']
        self.testing_DL2_gamma_zenith_distances = DL2_gamma_table['testing_DL2_gamma_zenith_distances']
        self.testing_DL2_gamma_azimuths = DL2_gamma_table['testing_DL2_gamma_azimuths']
        self.testing_DL2_proton_zenith_distances = DL2_proton_table['testing_DL2_proton_zenith_distances']
        self.testing_DL2_proton_azimuths = DL2_proton_table['testing_DL2_proton_azimuths']
        
    def update_merged_DL2_MC_files(self, testing_DL2_zenith_distance, testing_DL2_azimuth, testing_DL2_gamma_merged_file=None, testing_DL2_proton_merged_file=None):
        """
        Updates the model manager testing data in the index file.
        This method reads the model index file, finds the entry corresponding to the
        current model nickname, and updates the testing data in the index.
        The updated testing data are also reflected in the instance's attributes.
        Args:
            testing_gamma_dirs (list): Directories of testing gamma data.
            testing_proton_dirs (list): Directories of testing proton data.
            testing_gamma_zenith_distances (list): Zenith distances of testing gamma data.
            testing_gamma_azimuths (list): Azimuths of testing gamma data.
            testing_proton_zenith_distances (list): Zenith distances of testing proton data.
            testing_proton_azimuths (list): Azimuths of testing proton data.
        Raises:
            IndexError: If the model nickname is not found in the model index file.
        """
        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
        print(f"💾 Model {self.model_nickname} DL2 merged data update:")
        if testing_DL2_gamma_merged_file is not None:
            DL2_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/gamma')
            match = np.where((DL2_gamma_table['testing_DL2_gamma_zenith_distances'] == testing_DL2_zenith_distance) &
                    (DL2_gamma_table['testing_DL2_gamma_azimuths'] == testing_DL2_azimuth))[0]
            if len(match) > 0:
                DL2_gamma_table.remove_rows(match)
            DL2_gamma_table.add_row([self.model_index, testing_DL2_gamma_merged_file, testing_DL2_zenith_distance, testing_DL2_azimuth])
            write_table_hdf5(DL2_gamma_table, self.model_index_file, path=f'{self.model_nickname}/DL2/MC/gamma', append=True, overwrite=True)
            print(f"\t➡️ Testing DL2 gamma merged data updated")
        
        if testing_DL2_proton_merged_file is not None:
            DL2_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/proton')
            match = np.where((DL2_proton_table['testing_DL2_proton_zenith_distances'] == testing_DL2_zenith_distance) &
                        (DL2_proton_table['testing_DL2_proton_azimuths'] == testing_DL2_azimuth))[0]
            if len(match) > 0:
                DL2_proton_table.remove_rows(match)
            DL2_proton_table.add_row([self.model_index, testing_DL2_proton_merged_file, testing_DL2_zenith_distance, testing_DL2_azimuth])
            write_table_hdf5(DL2_proton_table, self.model_index_file, path=f'{self.model_nickname}/DL2/MC/proton', append=True, overwrite=True)
            print(f"\t➡️ Testing DL2 proton merged data updated")
        
    def update_model_manager_IRF_data(self, config, cuts_file, irf_file, bencmark_file, zenith, azimuth):
        """
        Updates the model manager IRF data in the index file.
        This method reads the model index file, finds the entry corresponding to the
        current model nickname, and updates the IRF data in the index.
        The updated IRF data are also reflected in the instance's attributes.
        Args:
            config (str): Configuration file for IRF data.
            cuts_file (str): Cuts file for IRF data.
            irf_file (str): IRF file for IRF data.
        Raises:
            IndexError: If the model nickname is not found in the model index file.
        """
        from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
        
        try:
            IRF_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/IRF')
        except:
            IRF_table = QTable(names=['model_index', 'config', 'cuts_file', 'irf_file', 'benckmark_file', 'zenith', 'azimuth'], 
                               dtype=[int, str, str, str, str, float, float])
        print(f"💾 Model {self.model_nickname} IRF data update:")
        if len(IRF_table)==0:
            IRF_table = QTable(names=['model_index', 'config', 'cuts_file', 'irf_file', 'benckmark_file', 'zenith', 'azimuth'], 
                               dtype=[int, str, str, str, str, float, float])
        
        match = np.where((IRF_table['config'] == config) or 
                (IRF_table['cuts_file'] == cuts_file) or 
                (IRF_table['irf_file'] == irf_file) or
                (IRF_table['benckmark_file'] == bencmark_file) or
                ((IRF_table['zenith'] == zenith) and
                (IRF_table['azimuth'] == azimuth))
                )[0]
        if len(match) == 0:
            IRF_table.add_row([self.model_index, config, cuts_file, irf_file, bencmark_file, zenith, azimuth])
            write_table_hdf5(IRF_table, self.model_index_file, path=f'{self.model_nickname}/IRF', append=True, overwrite=True)
        else:
            IRF_table.remove_rows(match)
            IRF_table.add_row([self.model_index, config, cuts_file, irf_file, bencmark_file, zenith, azimuth])
            write_table_hdf5(IRF_table, self.model_index_file, path=f'{self.model_nickname}/IRF', append=True, overwrite=True)
        print(f"\t➡️ IRF data updated")
        
        self.config = IRF_table['config']
        self.cuts_file = IRF_table['cuts_file']
        self.irf_file = IRF_table['irf_file']
        
    def get_IRF_data(self, zenith, azimuth):
        
        from astropy.io.misc.hdf5 import read_table_hdf5
        
        IRF_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/IRF')
        match = np.where((IRF_table['zenith'] == zenith) & (IRF_table['azimuth'] == azimuth))[0]
        if len(match) == 0:
            raise IndexError(f"No IRF data found for altitude {zenith} and azimuth {azimuth}")
        return IRF_table['config'][match][0], IRF_table['cuts_file'][match][0], IRF_table['irf_file'][match][0], IRF_table['benckmark_file'][match][0]

    def get_DL2_MC_files(self, zenith, azimuth):
    
        from astropy.io.misc.hdf5 import read_table_hdf5
        
        DL2_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/gamma')
        DL2_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/proton')
        match_gamma = np.where((DL2_gamma_table['testing_DL2_gamma_zenith_distances'] == zenith) & (DL2_gamma_table['testing_DL2_gamma_azimuths'] == azimuth))[0]
        match_proton = np.where((DL2_proton_table['testing_DL2_proton_zenith_distances'] == zenith) & (DL2_proton_table['testing_DL2_proton_azimuths'] == azimuth))[0]
        if len(match_gamma) == 0:
            raise IndexError(f"No DL2 gamma MC files found for zenith {zenith} and azimuth {azimuth}")
        if len(match_proton) == 0:
            raise IndexError(f"No DL2 proton MC files found for zenith {zenith} and azimuth {azimuth}")
        return DL2_gamma_table['testing_DL2_gamma_files'][match_gamma], DL2_proton_table['testing_DL2_proton_files'][match_proton]
            
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
    
    
    
    def __init__(self, direction_model: CTLearnModelManager, energy_model: CTLearnModelManager, type_model: CTLearnModelManager):
        if direction_model.reco == 'direction':
            self.direction_model = direction_model
        else:
            raise ValueError('direction_model must be a direction model')
        if energy_model.reco == 'energy':
            self.energy_model = energy_model
        else:
            raise ValueError('energy_model must be an energy model')
        if type_model.reco == 'type':
            self.type_model = type_model
        else:
            raise ValueError('type_model must be a type model')
        
        if not (self.direction_model.channels == self.energy_model.channels == self.type_model.channels):
            raise ValueError('All models must have the same channels')
        else:
            self.channels = self.direction_model.channels
            
        if not (self.direction_model.zd_range == self.energy_model.zd_range == self.type_model.zd_range):
            raise ValueError('All models must have the same zenith distance range')
        if not (self.direction_model.az_range == self.energy_model.az_range == self.type_model.az_range):
            raise ValueError('All models must have the same azimuth range')
        
        if not (self.direction_model.stereo == self.energy_model.stereo == self.type_model.stereo):
            raise ValueError('All models must have the same stereo value')
        else:
            self.stereo = self.direction_model.stereo
            
    def set_testing_directories(self, testing_gamma_dirs = [], testing_proton_dirs = [], testing_gamma_zenith_distances = [], testing_gamma_azimuths = [], testing_proton_zenith_distances = [], testing_proton_azimuths = []):
        if not (len(testing_gamma_dirs) == len(testing_gamma_zenith_distances) == len(testing_gamma_azimuths)):
            raise ValueError("All testing gamma lists must be the same length")
        if not (len(testing_proton_dirs) == len(testing_proton_zenith_distances) == len(testing_proton_azimuths)):
            raise ValueError("All testing proton lists must be the same length")
                
        for model in [self.direction_model, self.energy_model, self.type_model]:
            model.update_model_manager_testing_data(
                testing_gamma_dirs, 
                testing_proton_dirs, 
                testing_gamma_zenith_distances, 
                testing_gamma_azimuths, 
                testing_proton_zenith_distances, 
                testing_proton_azimuths
            )
              
        
    def launch_testing(self, zenith, azimuth, output_dirs: list, pattern="*.dl1.h5", sbatch_scripts_dir=None, launch_particle_type='both', cluster=None, account=None, python_env='ctlearn-cluster'):
        import os
        import glob
        # Check that the testing files are the same for each model
        gamma_dir = []
        proton_dir = []
        if launch_particle_type not in ['gamma', 'proton', 'both']:
            raise ValueError("launch_particle_type must be 'gamma', 'proton', or 'both'")
        if launch_particle_type in ['gamma', 'both']:
            if not (self.direction_model.testing_gamma_dirs == self.energy_model.testing_gamma_dirs == self.type_model.testing_gamma_dirs):
                raise ValueError("All models must have the same testing gamma directories, use set_testing_files to set them")
            if not self.direction_model.testing_gamma_dirs or not self.energy_model.testing_gamma_dirs or not self.type_model.testing_gamma_dirs:
                raise ValueError("Testing gamma directories cannot be empty")
            gamma_dirs = self.direction_model.testing_gamma_dirs
            gamma_zeniths = self.direction_model.testing_gamma_zenith_distances
            gamma_azimuths = self.direction_model.testing_gamma_azimuths
            matching_dirs = [gamma_dirs[i] for i in range(len(gamma_dirs)) if gamma_zeniths[i] == zenith and gamma_azimuths[i] == azimuth]
            if not matching_dirs:
                raise ValueError(f"No matching gamma directory found for zenith {zenith} and azimuth {azimuth}")
            gamma_dir = matching_dirs[0]
        if launch_particle_type in ['proton', 'both']:
            if not (self.direction_model.testing_proton_dirs == self.energy_model.testing_proton_dirs == self.type_model.testing_proton_dirs):
                raise ValueError("All models must have the same testing proton directories, use set_testing_files to set them")
            if not self.direction_model.testing_proton_dirs or not self.energy_model.testing_proton_dirs or not self.type_model.testing_proton_dirs:
                raise ValueError("Testing proton directories cannot be empty")
            proton_dirs = self.direction_model.testing_proton_dirs
            proton_zeniths = self.direction_model.testing_proton_zenith_distances
            proton_azimuths = self.direction_model.testing_proton_azimuths
            matching_dirs = [proton_dirs[i] for i in range(len(proton_dirs)) if proton_zeniths[i] == zenith and proton_azimuths[i] == azimuth]
            if not matching_dirs:
                raise ValueError(f"No matching proton directory found for zenith {zenith} and azimuth {azimuth}")
            proton_dir = matching_dirs[0] 
            
        if len(output_dirs) == 1:
            output_dir = output_dirs[0]
            gamma_files = np.sort(glob.glob(f"{gamma_dir}/{pattern}"))
            proton_files = np.sort(glob.glob(f"{proton_dir}/{pattern}"))
            testing_files = np.concatenate([gamma_files, proton_files])
            gamma_output_files = [f"{output_dir}/{Path(file).stem.replace('dl1', 'dl2')}.h5" for file in gamma_files]
            proton_output_files = [f"{output_dir}/{Path(file).stem.replace('dl1', 'dl2')}.h5" for file in proton_files]
            output_files = [f"{output_dir}/{Path(file).stem.replace('dl1', 'dl2')}.h5" for file in testing_files]
            for model in [self.direction_model, self.energy_model, self.type_model]:
                model.update_model_manager_DL2_MC_files(
                    gamma_output_files, 
                    proton_output_files, 
                    [zenith] * len(gamma_output_files), 
                    [azimuth] * len(gamma_output_files), 
                    [zenith] * len(proton_output_files), 
                    [azimuth] * len(proton_output_files)
                )
        elif len(output_dirs) == 2:
            gamma_output_dir = output_dirs[0]
            proton_output_dir = output_dirs[1]
            gamma_files = np.sort(glob.glob(f"{gamma_dir}/{pattern}"))
            proton_files = np.sort(glob.glob(f"{proton_dir}/{pattern}"))
            gamma_output_files = [f"{gamma_output_dir}/{Path(file).stem.replace('dl1', 'dl2')}.h5" for file in gamma_files]
            proton_output_files = [f"{proton_output_dir}/{Path(file).stem.replace('dl1', 'dl2')}.h5" for file in proton_files]
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
                
        else:
            raise ValueError("output_dirs must have length 1 or 2, to store all in the same directory, or gammas in the first and protons in the second")
        channels_string = ""
        for channel in self.channels:
            channels_string += f"--DLImageReader.channels={channel} "
        type_model_dir = np.sort(glob.glob(f"{self.type_model.model_dir}/{self.type_model.model_nickname}_v*"))[-1]
        energy_model_dir = np.sort(glob.glob(f"{self.energy_model.model_dir}/{self.energy_model.model_nickname}_v*"))[-1]
        direction_model_dir = np.sort(glob.glob(f"{self.direction_model.model_dir}/{self.direction_model.model_nickname}_v*"))[-1]
        
            
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
            
            if cluster is not None:
                sbatch_file = self.write_sbatch_script(cluster, Path(input_file).stem, cmd, sbatch_scripts_dir, env_name=python_env, account=account)
                os.system(f"sbatch {sbatch_file}")  
            else:
                print(cmd)
                os.system(cmd)
            
    def write_sbatch_script(self, cluster, job_name, cmd, sbatch_scripts_dir, env_name, account):
        sh_script = get_predict_data_sbatch_script(cluster, cmd, job_name, sbatch_scripts_dir, account, env_name)
        sbatch_file = f"{sbatch_scripts_dir}/{job_name}.sh"
        with open(sbatch_file, "w") as f:
            f.write(sh_script)

        print(f"💾 Testing script saved in {sbatch_file}")
        return sbatch_file
    
    def predict_lstchain_data(self, input_file, output_file):
        pass
    
    def predict_data(self, input_file, output_file, sbatch_scripts_dir=None, cluster=None, account=None, python_env=None):
        import os
        import glob
        
        os.system(f"mkdir -p {output_file.rsplit('/', 1)[0]}")
        channels_string = ""
        for channel in self.channels:
            channels_string += f"--DLImageReader.channels {channel} "
        type_model_dir = np.sort(glob.glob(f"{self.type_model.model_dir}/{self.type_model.model_nickname}_v*"))[-1]
        energy_model_dir = np.sort(glob.glob(f"{self.energy_model.model_dir}/{self.energy_model.model_nickname}_v*"))[-1]
        direction_model_dir = np.sort(glob.glob(f"{self.direction_model.model_dir}/{self.direction_model.model_nickname}_v*"))[-1]
        
        if self.stereo:
            cmd = f"ctlearn-predict-model --input_url {input_file} \
--type_model {type_model_dir}/ctlearn_model.cpk \
--energy_model {energy_model_dir}/ctlearn_model.cpk \
--direction_model {direction_model_dir}/ctlearn_model.cpk \
--no-dl1-images --no-true-images \
--output {output_file} \
--PredictCTLearnModel.dl1dh_reader_type DLImageReader \
--DLImageReader.image_mapper_type BilinearMapper \
--DLImageReader.mode stereo --PredictCTLearnModel.stack_telescope_images True --DLImageReader.min_telescopes 2 \
--PredictCTLearnModel.overwrite_tables True -v {channels_string}"
        else:
            # cmd   f"ctlearn-predict-mono --input_url {input_file} --type_model {type_model_dir}/ctlearn_model.cpk --energy_model {energy_model_dir}/ctlearn_model.cpk --direction_model {direction_model_dir}/ctlearn_model.cpk --no-dl1-images --no-true-images --output {output_file} --overwrite -v {channels_string}"
            cmd = f"ctlearn-predict-model --input_url {input_file} \
--type_model {type_model_dir}/ctlearn_model.cpk \
--energy_model {energy_model_dir}/ctlearn_model.cpk \
--direction_model {direction_model_dir}/ctlearn_model.cpk \
--no-dl1-images --no-true-images --output {output_file} \
--PredictCTLearnModel.overwrite_tables True -v {channels_string}"
            
        if cluster is not None:
            sbatch_file = self.write_sbatch_script(cluster, Path(input_file).stem, cmd, sbatch_scripts_dir, python_env, account)
            os.system(f"sbatch {sbatch_file}")
        else:
            print(cmd)
            os.system(cmd)
    
    
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
        dl2_gamma = []
        for file in testing_DL2_gamma_files:
            dl2_gamma.append(load_DL2_data_MC(file))
        dl2_gamma = vstack(dl2_gamma)
        
        dl2_protons = []
        for file in testing_DL2_proton_files:
            dl2_protons.append(load_DL2_data_MC(file))
        dl2_proton = vstack(dl2_protons)
        
        axs[0].scatter(dl2_gamma['array_altitude'][0]/np.pi*180, dl2_gamma['array_azimuth'][0]/np.pi*180, color="red", label="Array pointing", marker="x", s=100)
        axs[0].hist2d(dl2_gamma["CTLearn_alt"], dl2_gamma["CTLearn_az"], bins=100, zorder=0, cmap="viridis", norm=plt.cm.colors.LogNorm())
        axs[0].set_xlabel("Altitude [deg]")
        axs[0].set_ylabel("Azimuth [deg]")
        axs[0].legend()
        axs[0].set_title("Gammas")
        cbar = plt.colorbar(axs[0].collections[0], ax=axs[0])
        cbar.set_label("Counts")
        
        axs[1].scatter(dl2_proton['array_altitude'][0]/np.pi*180, dl2_proton['array_azimuth'][0]/np.pi*180, color="red", label="Array pointing", marker="x", s=100)
        axs[1].hist2d(dl2_proton["CTLearn_alt"], dl2_proton["CTLearn_az"], bins=100, zorder=0, cmap="viridis", norm=plt.cm.colors.LogNorm())
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
                raise ValueError("A cuts file must be provided, at least the first time.")
        if output_irf_file is None:
            try:
                output_irf_file = self.direction_model.get_IRF_data(zenith, azimuth)[2]
            except:
                raise ValueError("An IRF file must be provided, at least the first time.")
        if output_benchmark_file is None:
            try:
                output_benchmark_file = self.direction_model.get_IRF_data(zenith, azimuth)[3]
            except:
                raise ValueError("A benchmark file must be provided, at least the first time.")
        
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
            --point-like \
            --output {output_cuts_file} \
            --overwrite True \
            --EventSelectionOptimizer.optimization_algorithm=PercentileCuts"
        os.system(cmd)
        cmd = f"ctapipe-compute-irf \
            -c {config} --IrfTool.cuts_file {output_cuts_file} \
            --gamma-file {gamma_file} \
            --proton-file {proton_file}  \
            --do-background --point-like \
            --output {output_irf_file} \
            --benchmark-output {output_benchmark_file}"
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
        plt.plot(energy_center[0], hudl['ANGULAR RESOLUTION '].data['ANGULAR_RESOLUTION'][0,0,:])
        plt.xscale('log')
        plt.xlabel('Energy [TeV]')
        plt.ylabel('Angular resolution [deg]')
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
        rad_max = RadMax2D.read(irf_file, hdu="RAD_MAX")
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
        
        
        direction_training_log = np.sort(glob.glob(f"{self.direction_model.model_dir}/{self.direction_model.model_nickname}_v*/training_log.csv"))[-1]
        energy_training_log = np.sort(glob.glob(f"{self.energy_model.model_dir}/{self.energy_model.model_nickname}_v*/training_log.csv"))[-1]
        type_training_log = np.sort(glob.glob(f"{self.type_model.model_dir}/{self.type_model.model_nickname}_v*/training_log.csv"))[-1]
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, model in zip(axs, [self.direction_model, self.energy_model, self.type_model]):
            training_logs = np.sort(glob.glob(f"{model.model_dir}/{model.model_nickname}_v*/training_log.csv"))
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
            ax.set_title(f"{model.reco} training".title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_xticks(np.arange(1, len(df) + 1, 2))
            ax.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_angular_resolution(self, zenith, azimuth):
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
        for file in testing_DL2_gamma_files:
            dl2_gamma.append(self.load_DL2_data(file))
            shower_parameters_gamma.append(load_true_shower_parameters(file))
        dl2_gamma = vstack(dl2_gamma)
        shower_parameters_gamma = vstack(shower_parameters_gamma)
        dl2_gamma = join(dl2_gamma, shower_parameters_gamma, keys=["obs_id", "event_id"])

        reco_alt = dl2_gamma['CTLearn_alt'].to(u.deg)
        reco_az = dl2_gamma['CTLearn_az'].to(u.deg)
        true_alt = dl2_gamma['true_alt'].to(u.deg)
        true_az = dl2_gamma['true_az'].to(u.deg)
        reco_energy = dl2_gamma['CTLearn_energy']
        true_energy = dl2_gamma['true_energy']
        
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
        
    def plot_energy_resolution(self, zenith, azimuth):
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
        for file in testing_DL2_gamma_files:
            dl2_gamma.append(self.load_DL2_data(file))
            shower_parameters_gamma.append(load_true_shower_parameters(file))
        dl2_gamma = vstack(dl2_gamma)
        shower_parameters_gamma = vstack(shower_parameters_gamma)
        dl2_gamma = join(dl2_gamma, shower_parameters_gamma, keys=["obs_id", "event_id"])

        reco_energy = dl2_gamma['CTLearn_energy']
        true_energy = dl2_gamma['true_energy']
        
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
        
        


class TriModelCollection():
    
    def __init__(self, tri_models: list):
        self.tri_models = tri_models
        
    def predict_lstchain_data(self, input_file, output_file, pointing_table='/dl1/event/telescope/parameters/LST_LSTCam'):
        closest_tri_model = self.find_closest_model_to(input_file, pointing_table)
        closest_tri_model.predict_lstchain_data(input_file, output_file)
        
    def predict_data(self, input_file, output_file, pointing_table, pattern="*.dl1.h5", sbatch_scripts_dir=None, cluster=None, account=None, python_env='ctlearn-cluster'):
        closest_tri_model = self.find_closest_model_to(input_file, pointing_table)
        closest_tri_model.predict_data(input_file, output_file, pattern=pattern, sbatch_scripts_dir=sbatch_scripts_dir, cluster=cluster, account=account, python_env=python_env)
        
    def find_closest_model_to(self, input_file, pointing_table):
        from ctapipe.io import read_table
        print(f"This method will choose the closest model to the average pointing zenith and azimuth of the input file and predict the output file.")
        pointing = read_table(input_file, path=pointing_table)
        avg_data_az = np.mean(pointing['az_tel']*180/np.pi)
        avg_data_ze = np.mean(90 - pointing['alt_tel']*180/np.pi)
        print(f"📡 Average pointing of run {input_file} : ({avg_data_ze:3f}, {avg_data_az:3f})")
        avg_model_azs = []
        avg_model_zes = []
        for tri_model in self.tri_models:
            avg_model_azs.append(np.mean((tri_model.direction_model.az_range)))
            avg_model_zes.append(np.mean((tri_model.direction_model.zd_range)))
        print(f"🔍 Closest model avg node : ({avg_model_zes[np.argmin(np.abs(avg_model_zes - avg_data_ze))]:3f}, {avg_model_azs[np.argmin(np.abs(avg_model_azs - avg_data_az))]:3f})")
        closest_model_index = np.argmin(angular_distance(avg_data_ze, avg_data_az, avg_model_zes, avg_model_azs))
        closest_model = self.tri_models[closest_model_index]
        return closest_model
    
    
    
    


def set_mpl_style():
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as font_manager
    from matplotlib import rcParams
    from . import resources


    # font_path = "./resources/Outfit-Medium.ttf"
    import importlib.resources as pkg_resources

    with pkg_resources.path(resources, 'Outfit-Medium.ttf') as font_path:
        font_manager.fontManager.addfont(font_path)
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    rcParams['font.sans-serif'] = prop.get_name()
    rcParams['font.family'] = prop.get_name()
    with pkg_resources.path(resources, 'CTLearnStyle.mplstyle') as style_path:
        plt.style.use(style_path)
    # plt.style.use('./resources/ctlearnStyle.mplstyle')
    
def angular_distance(ze1, az1, ze2, az2):
    ze1, az1, ze2, az2 = map(np.radians, [ze1, az1, ze2, az2])
    delta_az = az2 - az1
    delta_ze = ze2 - ze1
    a = np.sin(delta_ze / 2)**2 + np.cos(ze1) * np.cos(ze2) * np.sin(delta_az / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c

def get_predict_data_sbatch_script(cluster, command, job_name, sbatch_scripts_dir, account, env_name):
    sbatch_predict_data_configs = {
    'camk': 
    f'''#!/bin/sh
#SBATCH --time=03:00:00
#SBATCH -o {sbatch_scripts_dir}/{job_name}%x.%j.out
#SBATCH -e {sbatch_scripts_dir}/{job_name}%x.%j.err 
#SBATCH -J {job_name}
#SBATCH --mem=10000
source ~/.bashrc
###. /home/blacave/mambaforge/etc/profile.d/conda.sh
conda activate {env_name}
echo $CONDA_DEFAULT_ENV
srun {command}''',

    'cscs': f'''#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64000mb
#SBATCH --output={sbatch_scripts_dir}/{job_name}.%x.%j.out
#SBATCH --error={sbatch_scripts_dir}/{job_name}.%x.%j.err
#SBATCH --account={account}

source ~/.bashrc
conda activate {env_name}
echo $CONDA_DEFAULT_ENV
echo $SLURM_ARRAY_TASK_ID

srun {command}
''',
    'lst-cluster':f''' ''',
                    
    }
    return sbatch_predict_data_configs[cluster]


def load_DL2_data_MC(input_file):
    from ctapipe.io import read_table
    from astropy.table import (join, hstack)
    pointing = read_table(input_file, "dl1/monitoring/subarray/pointing/")
    dl2_classification = read_table(input_file, "dl2/event/subarray/classification/CTLearn")
    dl2_classification = hstack([dl2_classification, pointing])
    dl2_classification = dl2_classification[~np.isnan(dl2_classification["CTLearn_prediction"])]
    dl2_energy = read_table(input_file, "dl2/event/subarray/energy/CTLearn")
    dl2_energy = dl2_energy[~np.isnan(dl2_energy["CTLearn_energy"])]
    dl2_geometry = read_table(input_file, "dl2/event/subarray/geometry/CTLearn")
    dl2_geometry = dl2_geometry[~np.isnan(dl2_geometry["CTLearn_alt"])]
    dl2 = join(dl2_classification, dl2_energy, keys=["obs_id", "event_id"])
    dl2 = join(dl2, dl2_geometry, keys=["obs_id", "event_id"])
    return dl2

def load_true_shower_parameters(input_file):
    from ctapipe.io import read_table
    true_shower_parameters = read_table(input_file, "simulation/event/subarray/shower")
    return true_shower_parameters