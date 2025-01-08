from astropy.table import QTable
import numpy as np
from pathlib import Path
import ast
from ctlearn_manager.utils.utils import set_mpl_style

__all__ = ['CTLearnModelManager']

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
        self.model_nickname = model_parameters.get('model_nickname', 'new_model')
        if not load:
            self.save_to_index(model_parameters)
        self.model_parameters_table = read_table_hdf5(f"{self.model_index_file}", path=f"{self.model_nickname}/parameters")
        training_table_gamma = read_table_hdf5(f"{self.model_index_file}", path=f"{self.model_nickname}/training/gamma")
        training_table_proton = read_table_hdf5(f"{self.model_index_file}", path=f"{self.model_nickname}/training/proton")
        # self.model_index = 0
        self.stereo = len(ast.literal_eval(self.model_parameters_table['telescopes_indices'][0])) > 1
        if self.model_parameters_table['reco'][0] == 'type' and (len(training_table_proton['training_proton_patterns']) == 0 or len(training_table_gamma['training_gamma_patterns']) == 0):
            raise ValueError("For reco type, training_proton_patterns and training_gamma_patterns are required")
        # Check that all gamma related lists are the same length
        gamma_lengths = [len(training_table_gamma['training_gamma_patterns']), len(training_table_gamma['training_gamma_zenith_distances']), len(training_table_gamma['training_gamma_azimuths'])]
        if len(set(gamma_lengths)) != 1:
            raise ValueError("All gamma related lists must be the same length")

        # Check that all proton related lists are the same length
        proton_lengths = [len(training_table_proton['training_proton_patterns']), len(training_table_proton['training_proton_zenith_distances']), len(training_table_proton['training_proton_azimuths'])]
        if len(set(proton_lengths)) != 1:
            raise ValueError("All proton related lists must be the same length")
        
        print(f"🧠 Model name: {self.model_nickname}")
        
        
    def save_to_index(self, model_parameters):
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
            model_table = QTable(names=['model_nickname', 'model_dir', 'reco', 'channels', 'telescope_names', 'telescopes_indices', 'notes', 'max_training_epochs'],
                        dtype=[str, str, str, str, str, str, str, int])
            training_table_gamma = QTable(names=['training_gamma_dir', 'training_gamma_patterns', 'training_gamma_zenith_distances', 'training_gamma_azimuths'],
                            dtype=[str, str, float, float])
            training_table_proton = QTable(names=['training_proton_dir', 'training_proton_patterns', 'training_proton_zenith_distances', 'training_proton_azimuths'],
                            dtype=[str, str, float, float])
            
            notes = model_parameters.get('notes', '')
            model_dir = model_parameters.get('model_dir', '')
            reco = model_parameters.get('reco', 'default_reco')
            telescope_names = model_parameters.get('telescope_names', [])
            telescopes_indices = model_parameters.get('telescopes_indices', [])
            channels = model_parameters.get('channels', ['cleaned_image', 'cleaned_relative_peak_time'])
            max_training_epochs = model_parameters.get('max_training_epochs', 10)
            
            #Training tables
            training_gamma_dir = model_parameters.get('training_gamma_dir', "")
            training_proton_dir = model_parameters.get('training_proton_dir', "")
            training_gamma_patterns = model_parameters.get('training_gamma_patterns', [])
            training_proton_patterns = model_parameters.get('training_proton_patterns', [])
            training_gamma_zenith_distances = model_parameters.get('training_gamma_zenith_distances', [])
            training_gamma_azimuths = model_parameters.get('training_gamma_azimuths', [])
            training_proton_zenith_distances = model_parameters.get('training_proton_zenith_distances', [])
            training_proton_azimuths = model_parameters.get('training_proton_azimuths', [])
            
            model_table.add_row([self.model_nickname, model_dir, reco, str(channels), str(telescope_names), str(telescopes_indices), notes, max_training_epochs])
            for i in range(len(training_gamma_patterns)):
                training_table_gamma.add_row([training_gamma_dir, training_gamma_patterns[i], training_gamma_zenith_distances[i], training_gamma_azimuths[i]])
            for i in range(len(training_proton_patterns)):
                training_table_proton.add_row([training_proton_dir, training_proton_patterns[i], training_proton_zenith_distances[i], training_proton_azimuths[i]])
            
            write_table_hdf5(training_table_gamma, self.model_index_file, path=f'{self.model_nickname}/training/gamma', append=True, overwrite=True)
            write_table_hdf5(training_table_proton, self.model_index_file, path=f'{self.model_nickname}/training/proton', append=True, overwrite=True)
            write_table_hdf5(model_table, self.model_index_file, path=f'{self.model_nickname}/parameters',append=True, overwrite=True)
            
            print(f"✅ Model nickname {self.model_nickname} added to table")
        
        
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
        from astropy.io.misc.hdf5 import read_table_hdf5
        n_epoch_training = self.get_n_epoch_trained()
        print(f"📊 Model trained for {n_epoch_training} epochs")
        max_training_epochs = self.model_parameters_table['max_training_epochs'][0]
        model_dir = self.model_parameters_table['model_dir'][0]
        if n_epochs > max_training_epochs:
            print(f"⚠️ Number of epochs increased from {max_training_epochs} to {n_epochs}")
            self.update_model_manager_parameters_in_index({'max_training_epochs': n_epochs})
            max_training_epochs = n_epochs
        if n_epoch_training >= max_training_epochs:
            print(f"🛑 Model already trained for {n_epoch_training} epochs. Will not train further.")
            self.plot_loss()
            return
        n_epochs = max_training_epochs - n_epoch_training
        print(f"🚀 Launching training for {n_epochs} epochs")
        import glob
        import os
        models_dir = np.sort(glob.glob(f"{model_dir}/{self.model_nickname}*"))
        load_model = False
        if len(models_dir) > 0 :
            last_model_dir = Path(models_dir[-1])
            size = sum(f.stat().st_size for f in last_model_dir.glob('**/*') if f.is_file())
            model_version = int(models_dir[-1].split("_v")[-1])
            if size > 1e6:
                model_version += 1
                print(f"➡️ Model already exists: will continue training and create {self.model_nickname}_v{model_version}")
                save_best_validation_only = True
                model_dir = f"{model_dir}/{self.model_nickname}_v{model_version}/"
                model_to_load = f"{model_dir}/{self.model_nickname}_v{model_version - 1}/ctlearn_model.cpk/"
                load_model = True
                os.system(f"mkdir -p {model_dir}")
            else :
                model_dir = f"{model_dir}/{self.model_nickname}_v{model_version}/"
                if model_version > 0:
                    model_to_load = f"{model_dir}/{self.model_nickname}_v{model_version - 1}/ctlearn_model.cpk/"
                    load_model = True
                    print(f"➡️ Model already exists: will continue training and create {self.model_nickname}_v{model_version}")
                    save_best_validation_only = True
                else:
                    print(f"🆕 Model does not exist: will create {self.model_nickname}_v{model_version}")
                    save_best_validation_only = False
        else:
            model_version = 0
            print(f"🆕 Model does not exist: will create {self.model_nickname}_v{model_version}")
            model_dir = f"{model_dir}/{self.model_nickname}_v{model_version}/"
            os.system(f"mkdir -p {model_dir}")
            save_best_validation_only = False

        if load_model:
            load_model_string = f"--TrainCTLearnModel.model_type=LoadedModel --LoadedModel.load_model_from={model_to_load} "
        else:
            load_model_string = "" if transfer_learning_model_cpk is None else f"--TrainCTLearnModel.model_type=LoadedModel --LoadedModel.load_model_from={transfer_learning_model_cpk} "
        
        training_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/training/proton')
        training_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/training/gamma')
        background_string = f"--background {training_proton_table['training_proton_dir'][0]} " if self.model_parameters_table['reco'][0] == 'type' else ""
        signal_patterns = ""
        for pattern in training_gamma_table['training_gamma_patterns']:
            signal_patterns += f'--pattern-signal "{pattern}" '
        background_patterns = ""
        for pattern in training_proton_table['training_proton_patterns']:
            background_patterns += f'--pattern-background "{pattern}" '
        channels_string = ""
        for channel in ast.literal_eval(self.model_parameters_table['channels'][0]):
            channels_string += f"--DLImageReader.channels={channel} "

        stereo_mode = 'stereo' if self.stereo else "mono"
        stack_telescope_images = 'true' if self.stereo else 'false'
        min_telescopes = 2 if self.stereo else 1
        allowed_tels = '_'.join(map(str, ast.literal_eval(self.model_parameters_table['telescope_indices'][0]))) if self.stereo else int(ast.literal_eval(self.model_parameters_table['telescope_indices'][0])[0])
        
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
--reco {self.model_parameters_table['reco'][0]} \
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
        training_logs = np.sort(glob.glob(f"{self.model_parameters_table['model_dir'][0]}/{self.model_nickname}_v*/training_log.csv"))
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
        training_logs = np.sort(glob.glob(f"{self.model_parameters_table['model_dir'][0]}/{self.model_nickname}_v*/training_log.csv"))
        losses_train = []
        losses_val = []
        for training_log in training_logs:
            df = pd.read_csv(training_log)
            losses_train = np.concatenate((losses_train, df['loss'].to_numpy()))
            losses_val = np.concatenate((losses_val, df['val_loss'].to_numpy()))
        epochs = np.arange(1, len(losses_train)+1)
        plt.plot(epochs, losses_train, label=f"Training", lw=2)
        plt.plot(epochs, losses_val, label=f"Validation", ls='--')
        plt.title(f"{self.model_parameters_table['reco'][0]} training".title())
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(np.arange(1, len(losses_train) + 1, 2))
        plt.legend()
        plt.show()
        
        
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
            training_gamma_table = QTable(names=['training_gamma_dir', 'training_gamma_patterns', 'training_gamma_zenith_distances', 'training_gamma_azimuths'], 
                                          dtype=[str, str, float, float])
        if len(training_proton_table)==0:
            training_proton_table = QTable(names=['training_proton_dir', 'training_proton_patterns', 'training_proton_zenith_distances', 'training_proton_azimuths'], 
                                           dtype=[str, str, float, float])
        
        if len(training_gamma_patterns) > 0:
            for i in range(len(training_gamma_patterns)):
                match = np.where((training_gamma_table['training_gamma_zenith_distances'] == training_gamma_zenith_distances[i]) & 
                     (training_gamma_table['training_gamma_azimuths'] == training_gamma_azimuths[i]))[0]
                if len(match) > 0:
                    training_gamma_table['training_gamma_dir'][match[0]] = training_gamma_dir
                    training_gamma_table['training_gamma_patterns'][match[0]] = training_gamma_patterns[i]
                else:
                    training_gamma_table.add_row([training_gamma_dir, training_proton_patterns[i], training_gamma_zenith_distances[i], training_gamma_azimuths[i]])
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
                    training_proton_table.add_row([training_proton_dir, training_proton_patterns[i], training_proton_zenith_distances[i], training_proton_azimuths[i]])
            write_table_hdf5(training_proton_table, self.model_index_file, path=f'{self.model_nickname}/training/proton', append=True, overwrite=True)
            print(f"\t➡️ Training proton data updated")
        
            

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
        
        try:
            testing_gamma_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/testing/gamma')
        except:
            testing_gamma_table = QTable(names=['testing_gamma_dirs', 'testing_gamma_zenith_distances', 'testing_gamma_azimuths'], 
                                        dtype=[str, float, float])
            
        try:
            testing_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/testing/proton')
        except:
            testing_proton_table = QTable(names=['testing_proton_dirs', 'testing_proton_zenith_distances', 'testing_proton_azimuths'], 
                                        dtype=[str, float, float])
        print(f"💾 Model {self.model_nickname} testing data update:")
        if len(testing_gamma_table)==0:
            testing_gamma_table = QTable(names=['testing_gamma_dirs', 'testing_gamma_zenith_distances', 'testing_gamma_azimuths'], 
                                        dtype=[str, float, float])
        if len(testing_proton_table)==0:
            testing_proton_table = QTable(names=['testing_proton_dirs', 'testing_proton_zenith_distances', 'testing_proton_azimuths'], 
                                        dtype=[str, float, float])
        
        if len(testing_gamma_dirs) > 0:
            for i in range(len(testing_gamma_dirs)):
                match = np.where((testing_gamma_table['testing_gamma_zenith_distances'] == testing_gamma_zenith_distances[i]) & 
                        (testing_gamma_table['testing_gamma_azimuths'] == testing_gamma_azimuths[i]))[0]
                if len(match) > 0:
                    testing_gamma_table['testing_gamma_dirs'][match[0]] = testing_gamma_dirs[i]
                else:
                    testing_gamma_table.add_row([testing_gamma_dirs[i], testing_gamma_zenith_distances[i], testing_gamma_azimuths[i]])
            write_table_hdf5(testing_gamma_table, self.model_index_file, path=f'{self.model_nickname}/testing/gamma', append=True, overwrite=True)
            print(f"\t➡️ Testing gamma data updated")
        
        if len(testing_proton_dirs) > 0:
            for i in range(len(testing_proton_dirs)):
                match = np.where((testing_proton_table['testing_proton_zenith_distances'] == testing_proton_zenith_distances[i]) & 
                        (testing_proton_table['testing_proton_azimuths'] == testing_proton_azimuths[i]))[0]
                if len(match) > 0:
                    testing_proton_table['testing_proton_dirs'][match[0]] = testing_proton_dirs[i]
                else:
                    testing_proton_table.add_row([testing_proton_dirs[i], testing_proton_zenith_distances[i], testing_proton_azimuths[i]])
            write_table_hdf5(testing_proton_table, self.model_index_file, path=f'{self.model_nickname}/testing/proton', append=True, overwrite=True)
            print(f"\t➡️ Testing proton data updated")

            
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
            DL2_gamma_table = QTable(names=['testing_DL2_gamma_files', 'testing_DL2_gamma_zenith_distances', 'testing_DL2_gamma_azimuths'], 
                                     dtype=[ str, float, float])
        try:
            DL2_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/proton')
        except:
            DL2_proton_table = QTable(names=['testing_DL2_proton_files', 'testing_DL2_proton_zenith_distances', 'testing_DL2_proton_azimuths'], 
                                      dtype=[ str, float, float])

        print(f"💾 Model {self.model_nickname} DL2 data update:")
        if len(DL2_gamma_table)==0:
            DL2_gamma_table = QTable(names=['testing_DL2_gamma_files', 'testing_DL2_gamma_zenith_distances', 'testing_DL2_gamma_azimuths'], 
                                     dtype=[str, float, float])
        if len(DL2_proton_table)==0:
            DL2_proton_table = QTable(names=['testing_DL2_proton_files', 'testing_DL2_proton_zenith_distances', 'testing_DL2_proton_azimuths'], 
                                      dtype=[str, float, float])
        
        if len(testing_DL2_gamma_files) > 0:
            for i in range(len(testing_DL2_gamma_files)):
                match = np.where((DL2_gamma_table['testing_DL2_gamma_files'] == testing_DL2_gamma_files[i]) & 
                        (DL2_gamma_table['testing_DL2_gamma_zenith_distances'] == testing_DL2_gamma_zenith_distances[i]) & 
                        (DL2_gamma_table['testing_DL2_gamma_azimuths'] == testing_DL2_gamma_azimuths[i]))[0]
                if len(match) == 0:
                    DL2_gamma_table.add_row([testing_DL2_gamma_files[i], testing_DL2_gamma_zenith_distances[i], testing_DL2_gamma_azimuths[i]])
            write_table_hdf5(DL2_gamma_table, self.model_index_file, path=f'{self.model_nickname}/DL2/MC/gamma', append=True, overwrite=True)
            print(f"\t➡️ Testing DL2 gamma data updated")
        
        if len(testing_DL2_proton_files) > 0:
            for i in range(len(testing_DL2_proton_files)):
                match = np.where((DL2_proton_table['testing_DL2_proton_files'] == testing_DL2_proton_files[i]) & 
                        (DL2_proton_table['testing_DL2_proton_zenith_distances'] == testing_DL2_proton_zenith_distances[i]) & 
                        (DL2_proton_table['testing_DL2_proton_azimuths'] == testing_DL2_proton_azimuths[i]))[0]
                if len(match) == 0:
                    DL2_proton_table.add_row([testing_DL2_proton_files[i], testing_DL2_proton_zenith_distances[i], testing_DL2_proton_azimuths[i]])
            write_table_hdf5(DL2_proton_table, self.model_index_file, path=f'{self.model_nickname}/DL2/MC/proton', append=True, overwrite=True)
            print(f"\t➡️ Testing DL2 proton data updated")
        
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
            DL2_gamma_table.add_row([testing_DL2_gamma_merged_file, testing_DL2_zenith_distance, testing_DL2_azimuth])
            write_table_hdf5(DL2_gamma_table, self.model_index_file, path=f'{self.model_nickname}/DL2/MC/gamma', append=True, overwrite=True)
            print(f"\t➡️ Testing DL2 gamma merged data updated")
        
        if testing_DL2_proton_merged_file is not None:
            DL2_proton_table = read_table_hdf5(self.model_index_file, path=f'{self.model_nickname}/DL2/MC/proton')
            match = np.where((DL2_proton_table['testing_DL2_proton_zenith_distances'] == testing_DL2_zenith_distance) &
                        (DL2_proton_table['testing_DL2_proton_azimuths'] == testing_DL2_azimuth))[0]
            if len(match) > 0:
                DL2_proton_table.remove_rows(match)
            DL2_proton_table.add_row([testing_DL2_proton_merged_file, testing_DL2_zenith_distance, testing_DL2_azimuth])
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
            IRF_table = QTable(names=['config', 'cuts_file', 'irf_file', 'benckmark_file', 'zenith', 'azimuth'], 
                               dtype=[str, str, str, str, float, float])
        print(f"💾 Model {self.model_nickname} IRF data update:")
        if len(IRF_table)==0:
            IRF_table = QTable(names=['config', 'cuts_file', 'irf_file', 'benckmark_file', 'zenith', 'azimuth'], 
                               dtype=[str, str, str, str, float, float])
        
        match = np.where((IRF_table['config'] == config) or 
                (IRF_table['cuts_file'] == cuts_file) or 
                (IRF_table['irf_file'] == irf_file) or
                (IRF_table['benckmark_file'] == bencmark_file) or
                ((IRF_table['zenith'] == zenith) and
                (IRF_table['azimuth'] == azimuth))
                )[0]
        if len(match) == 0:
            IRF_table.add_row([config, cuts_file, irf_file, bencmark_file, zenith, azimuth])
            write_table_hdf5(IRF_table, self.model_index_file, path=f'{self.model_nickname}/IRF', append=True, overwrite=True)
        else:
            IRF_table.remove_rows(match)
            IRF_table.add_row([config, cuts_file, irf_file, bencmark_file, zenith, azimuth])
            write_table_hdf5(IRF_table, self.model_index_file, path=f'{self.model_nickname}/IRF', append=True, overwrite=True)
        print(f"\t➡️ IRF data updated")
        
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
      
    def plot_zenith_azimuth_ranges(zenith_range, azimuth_range=None):
        """
        Plots the area or line or point covered by the zenith and azimuth ranges in a polar projection.
        
        Args:
            zenith_range (tuple): A tuple containing the minimum and maximum zenith angles.
            azimuth_range (tuple, optional): A tuple containing the minimum and maximum azimuth angles. If None, the plot will cover all azimuth angles.
        """
        set_mpl_style()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        
        zenith_min, zenith_max = np.radians(zenith_range)
        
        if azimuth_range is None:
            azimuth_min, azimuth_max = 0, 2 * np.pi
        else:
            azimuth_min, azimuth_max = np.radians(azimuth_range)
        
        # Create a grid of azimuth and zenith values
        azimuths = np.linspace(azimuth_min, azimuth_max, 100)
        zeniths = np.linspace(zenith_min, zenith_max, 100)
        
        # Create a meshgrid for plotting
        azimuth_grid, zenith_grid = np.meshgrid(azimuths, zeniths)
        
        # Plot the area covered by the zenith and azimuth ranges
        ax.pcolormesh(azimuth_grid, zenith_grid, np.ones_like(zenith_grid), shading='auto', alpha=0.3)
        
        # Plot the boundary lines
        ax.plot([azimuth_min, azimuth_max], [zenith_min, zenith_min], color='blue')
        ax.plot([azimuth_min, azimuth_max], [zenith_max, zenith_max], color='blue')
        ax.plot([azimuth_min, azimuth_min], [zenith_min, zenith_max], color='blue')
        ax.plot([azimuth_max, azimuth_max], [zenith_min, zenith_max], color='blue')
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(90)
        
        ax.set_title('Zenith and Azimuth Ranges')
        plt.show()
      
      
# class ModelRangeOfValidity:
#     def __init__(self, model_manager: CTLearnModelManager):
        
        
        
#         self.zenith_range = zenith_range
#         self.azimuth_range = azimuth_range
#         self.energy_range = energy_range
#         self.nsb_range = nsb_range
        

#     def matches(self, **kwargs):
#         for key, value in kwargs.items():
#             if key == 'zenith':
#                 if not (self.zenith_range[0] <= value <= self.zenith_range[1]):
#                     return False
#             elif key == 'azimuth':
#                 if self.azimuth_range is not None and not (self.azimuth_range[0] <= value <= self.azimuth_range[1]):
#                     return False
#             elif key == 'energy':
#                 if self.energy_range is not None and not (self.energy_range[0] <= value <= self.energy_range[1]):
#                     return False
#             elif key == 'nsb':
#                 if self.nsb_range is not None and not (self.nsb_range[0] <= value <= self.nsb_range[1]):
#                     return False
#         return True
    
