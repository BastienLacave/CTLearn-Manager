"""CTLearn Model Manager"""
from .version import __version__
from astropy.table import QTable
import numpy as np
from pathlib import Path

__all__ = [
    "__version__",
    "fibonacci",
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
        training_gamma_dirs (list): Directories of training gamma data.
        training_proton_dirs (list): Directories of training proton data.
        training_gamma_zenith_distances (list): Zenith distances of training gamma data.
        training_gamma_azimuths (list): Azimuths of training gamma data.
        training_proton_zenith_distances (list): Zenith distances of training proton data.
        training_proton_azimuths (list): Azimuths of training proton data.
        channels (list): Channels used in the model.
        max_training_epochs (int): Maximum number of training epochs.
        columns (list): Columns of the model index table.
        stereo (bool): Whether the model is stereo or not.
        zd_range (list): Zenith distance range.
        az_range (list): Azimuth range.
        model_name (str): Name of the model.
    Methods:
        __init__(model_parameters, MODEL_INDEX_FILE):
            Initializes the CTLearnModelManager with the given parameters.
        save_to_index():
            Saves the model parameters to the index file.
        launch_training(n_epochs=None):
            Launches the training process for the model.
        get_n_epoch_trained():
            Returns the number of epochs the model has been trained for.
        plot_loss():
            Plots the training and validation loss over epochs.
        info():
            Prints information about the model.
        update_model_manager_parameters_in_index(parameters):
            Updates the model parameters in the index file.
    """
    

    def __init__(self, model_parameters, MODEL_INDEX_FILE):
        self.model_index_file = MODEL_INDEX_FILE
        self.model_nickname = model_parameters['model_nickname']
        self.notes = model_parameters['notes']
        self.model_dir = model_parameters['model_dir'] #f"{model_parameters['model_dir']}/{model_parameters['model_nickname']}"
        self.reco = model_parameters['reco']
        self.telescope_names = model_parameters['telescope_names']
        self.telescopes_indices = model_parameters['telescopes_indices']
        self.training_gamma_dirs = model_parameters['training_gamma_dirs']
        self.training_proton_dirs = model_parameters['training_proton_dirs']
        self.training_gamma_zenith_distances = model_parameters['training_gamma_zenith_distances']
        self.training_gamma_azimuths = model_parameters['training_gamma_azimuths']
        self.training_proton_zenith_distances = model_parameters['training_proton_zenith_distances']
        self.training_proton_azimuths = model_parameters['training_proton_azimuths']
        self.channels = model_parameters['channels']
        self.max_training_epochs = model_parameters['max_training_epochs']
        self.columns = ['model_index', 
           'model_nickname', 'model_name', 
           'model_dir',
           'reco', 
           'channels',
           'telescope_names', 'telescopes_indices', 
           'training_gamma_dirs', 'training_proton_dirs', 
           'training_gamma_zenith_distances', 'training_gamma_azimuths', 
           'training_proton_zenith_distances', 'training_proton_azimuths', 
           'notes', 
           'zd_range', 'az_range',
           'testing_gamma_dirs', 'testing_proton_dirs',
           'testing_gamma_zenith_distances', 'testing_gamma_azimuths',
           'testing_proton_zenith_distances', 'testing_proton_azimuths',
           'max_training_epochs']
        self.stereo = True if len(self.telescopes_indices) > 1 else False
        self.testing_gamma_dirs = []
        self.testing_proton_dirs = []
        self.testing_gamma_zenith_distances = []
        self.testing_gamma_azimuths = []
        self.testing_proton_zenith_distances = []
        self.testing_proton_azimuths = []
        if self.reco == 'type' and (len(self.training_proton_dirs) == 0 or len(self.training_gamma_dirs) == 0):
            raise ValueError("For reco type, training_proton_dirs and training_gamma_dirs are required")
        # if self.reco == 'type' & (len(self.training_proton_dirs) == 0 or len(self.training_gamma_dirs) == 0):
        #     raise ValueError("For reco type, training_proton_dirs and training_gamma_dirs are required")
        # Check that all gamma related lists are the same length
        gamma_lengths = [len(self.training_gamma_dirs), len(self.training_gamma_zenith_distances), len(self.training_gamma_azimuths)]
        if len(set(gamma_lengths)) != 1:
            raise ValueError("All gamma related lists must be the same length")

        # Check that all proton related lists are the same length
        proton_lengths = [len(self.training_proton_dirs), len(self.training_proton_zenith_distances), len(self.training_proton_azimuths)]
        if len(set(proton_lengths)) != 1:
            raise ValueError("All proton related lists must be the same length")
        
        # Model parameters
        self.zd_range = [min(self.training_gamma_zenith_distances), max(self.training_gamma_zenith_distances)]
        self.az_range = [min(self.training_gamma_azimuths), max(self.training_gamma_azimuths)]
        self.model_name = f"{self.reco}_TEL{'_'.join(map(str, self.telescopes_indices))}_ZD{'_'.join(map(str, self.training_gamma_zenith_distances))}_Az{'_'.join(map(str, self.training_gamma_azimuths))}"
        print(f"🧠 Model name: {self.model_name}")
        
        
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
        
        try:
            models_table = QTable.read(self.model_index_file)
            model_index = models_table['model_index'][-1] + 1
        except:
            models_table = QTable(names=self.columns,  
                                #   units=[None, None, None, None, None, None, None, None, 'deg', 'deg', 'deg', 'deg', None, 'deg', 'deg', None, None, 'deg', 'deg', 'deg', 'deg'],
                                dtype=[int,
                                        str, str, 
                                        str, str,
                                        list,
                                        list, list, 
                                        list, list, 
                                        list, list, 
                                        list, list, 
                                        str, 
                                        list, list,
                                        list, list,
                                        list, list,
                                        list, list,
                                        int
                                        ])
            print(f"Model index did not exist, will create {self.model_index_file}")
            model_index = 0
        # if not Path(self.model_dir).exists():
        #     Path(self.model_dir).mkdir()
        #     print(f"Model directory {self.model_dir} created")
        if (self.model_nickname not in models_table['model_nickname']):
            models_table.add_row([model_index, 
                                self.model_nickname, self.model_name, 
                                self.model_dir,
                                self.reco, 
                                self.channels,
                                self.telescope_names, self.telescopes_indices, 
                                self.training_gamma_dirs, self.training_proton_dirs, 
                                self.training_gamma_zenith_distances, self.training_gamma_azimuths, 
                                self.training_proton_zenith_distances, self.training_proton_azimuths, 
                                self.notes, 
                                self.zd_range, self.az_range, 
                                [], [], 
                                [], [], 
                                [], [],
                                self.max_training_epochs
                                ])
            models_table.write(self.model_index_file, format='ascii.ecsv', serialize_method='data_mask', overwrite=True)
            print(f"✅ Model nickname {self.model_nickname} added to table")
        else:
            print(f"❌ Model nickname {self.model_nickname} already in table")
        
        
    def launch_training(self, n_epochs):
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

        load_model_string = f"--TrainCTLearnModel.model_type=LoadedModel --LoadedModel.load_model_from={model_to_load} " if load_model else ""
        background_string = f"--background {self.training_proton_dirs[0]} " if self.reco == 'type' else "" #FIXM loop over protons/gamma dirs and add patterns
        signal_patterns = ""
        background_patterns = ""
        # for ze, az in zip(zes, azs):
        #     signal_patterns += f'--pattern-signal "gamma_theta_{ze:.3f}_az_{az:.3f}_runs*.dl1.h5" '
        # if reco == 'type':
        #     for ze, az in zip(zes_protons, azs_protons):
        #         background_patterns += f'--pattern-background "proton_theta_{ze:.3f}_az_{az:.3f}_runs*.dl1.h5" '
        channels_string = ""
        for channel in self.channels:
            channels_string += f"--DLImageReader.channels={channel} "

        stereo_mode = 'stereo' if self.stereo else "mono"
        stack_telescope_images = 'true' if self.stereo else 'false'
        min_telescopes = 2 if self.stereo else 1
        allowed_tels = '_'.join(map(str, self.telescopes_indices)) if self.stereo else int(self.telescopes_indices[0])
        cmd = f"ctlearn-train-model {load_model_string}\
            --signal {self.training_gamma_dirs[0]} {signal_patterns}\
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
        plt.plot(epochs + 1, losses_train, label=f"Training")
        plt.plot(epochs + 1, losses_val, label=f"Validation", ls='--')
        plt.title(f"{self.reco} training")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
            
    def info(self):
        print(f"Model nickname: {self.model_nickname}")
        print(f"Model name: {self.model_name}")
        print(f"Model directory: {self.model_dir}")
        print(f"Reco: {self.reco}")
        print(f"Telescope names: {self.telescope_names}")
        print(f"Telescope indices: {self.telescopes_indices}")
        print(f"Training gamma dirs: {self.training_gamma_dirs}")
        print(f"Training proton dirs: {self.training_proton_dirs}")
        print(f"Training gamma zenith distances: {self.training_gamma_zenith_distances}")
        print(f"Training gamma azimuths: {self.training_gamma_azimuths}")
        print(f"Training proton zenith distances: {self.training_proton_zenith_distances}")
        print(f"Training proton azimuths: {self.training_proton_azimuths}")
        print(f"Notes: {self.notes}")
        print(f"ZD range: {self.zd_range}")
        print(f"Az range: {self.az_range}")
        print(f"Stereo: {self.stereo}")
        
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
        
        models_table = QTable.read(self.model_index_file)
        model_index = np.where(models_table['model_nickname'] == self.model_nickname)[0][0]
        print(f"💾 Model index update:")
        for key, value in parameters.items():
            models_table[key][model_index] = value
            self.__dict__[key] = value
            print(f"\t➡️ {key} updated to {value}")
        models_table.write(self.model_index_file, format='ascii.ecsv', serialize_method='data_mask', overwrite=True)
        # print(f"✅ Model parameters updated in index")
            
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
            
    def set_testing_files(self, testing_gamma_dirs = [], testing_proton_dirs = [], testing_gamma_zenith_distances = [], testing_gamma_azimuths = [], testing_proton_zenith_distances = [], testing_proton_azimuths = []):
        if not (len(testing_gamma_dirs) == len(testing_gamma_zenith_distances) == len(testing_gamma_azimuths)):
            raise ValueError("All testing gamma lists must be the same length")
        if not (len(testing_proton_dirs) == len(testing_proton_zenith_distances) == len(testing_proton_azimuths)):
            raise ValueError("All testing proton lists must be the same length")
        for model in [self.direction_model, self.energy_model, self.type_model]:
            parameters_to_update = {}
            if len(testing_gamma_dirs) > 0:
                parameters_to_update['testing_gamma_dirs'] = testing_gamma_dirs
            if len(testing_proton_dirs) > 0:
                parameters_to_update['testing_proton_dirs'] = testing_proton_dirs
            if len(testing_gamma_zenith_distances) > 0:
                parameters_to_update['testing_gamma_zenith_distances'] = testing_gamma_zenith_distances
            if len(testing_gamma_azimuths) > 0:
                parameters_to_update['testing_gamma_azimuths'] = testing_gamma_azimuths
            if len(testing_proton_zenith_distances) > 0:
                parameters_to_update['testing_proton_zenith_distances'] = testing_proton_zenith_distances
            if len(testing_proton_azimuths) > 0:
                parameters_to_update['testing_proton_azimuths'] = testing_proton_azimuths
            model.update_model_manager_parameters_in_index(parameters_to_update)
              
        
    def launch_testing(self, zenith, azimuth, output_dirs: list, sbatch_scripts_dir, launch_particle_type='both', cluster=None, account=None):
        import os
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
            testing_files = np.concatenate([np.sort(glob.glob(f"{gamma_dir}/*.dl1.h5")), np.sort(glob.glob(f"{proton_dir}/*.dl1.h5"))])
            output_files = [f"{output_dir}/{Path(file).stem.replace('dl1', 'dl2')}.h5" for file in testing_files]
        elif len(output_dirs) == 2:
            gamma_output_dir = output_dirs[0]
            proton_output_dir = output_dirs[1]
            gamma_files = np.sort(glob.glob(f"{gamma_dir}/*.dl1.h5"))
            proton_files = np.sort(glob.glob(f"{proton_dir}/*.dl1.h5"))
            gamma_output_files = [f"{gamma_output_dir}/{Path(file).stem.replace('dl1', 'dl2')}.h5" for file in gamma_files]
            proton_output_files = [f"{proton_output_dir}/{Path(file).stem.replace('dl1', 'dl2')}.h5" for file in proton_files]
            output_files = np.concatenate([gamma_output_files, proton_output_files])
        else:
            raise ValueError("output_dirs must have length 1 or 2, to store all in the same directory, or gammas in the first and protons in the second")
        import glob
        channels_string = ""
        for channel in self.channels:
            channels_string += f"--DLImageReader.channels={channel} "
        type_model_dir = np.sort(glob.glob(f"{self.type_model.model_dir}/{self.type_model.model_nickname}_v*"))[-1]
        energy_model_dir = np.sort(glob.glob(f"{self.energy_model.model_dir}/{self.energy_model.model_nickname}_v*"))[-1]
        direction_model_dir = np.sort(glob.glob(f"{self.direction_model.model_dir}/{self.direction_model.model_nickname}_v*"))[-1]
        
        for input_file, output_file in zip(testing_files, output_files):
            if self.stereo:
                cmd = "" #TODO implement stereo testing
            else:
                cmd = f"ctlearn-predict-mono --input_url {input_file} --type_model={type_model_dir}/ctlearn_model.cpk --energy_model={energy_model_dir}/ctlearn_model.cpk --direction_model={direction_model_dir}/ctlearn_model.cpk --no-dl1-images --no-true-images --output {output_file} --overwrite -v {channels_string}"
            
            if cluster == 'cscs':
                sbatch_file = self.write_cscs_sbatch_script(Path(input_file).stem, cmd, sbatch_scripts_dir)
                os.system(f"sbatch {sbatch_file}")
            
    def write_cscs_sbatch_script(job_name, cmd, sbatch_scripts_dir, account='cta04', env_name='ctlearn-cluster'):
        sh_script = f'''#!/bin/bash -l
#
#SBATCH --job-name={job_name}
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64000mb
#SBATCH --output=MC_{job_name}.%j.out
#SBATCH --error=MC_{job_name}.%j.err
#SBATCH --account={account}

source ~/.bashrc
conda activate {env_name}
echo $CONDA_DEFAULT_ENV
echo $SLURM_ARRAY_TASK_ID

srun {cmd}
'''
        #--type_model="{type_model_dir}/ctlearn_model.cpk"
        #--energy_model="{energy_model_dir}/ctlearn_model.cpk"
        sbatch_file = f"{sbatch_scripts_dir}/{job_name}.sh"
        with open(sbatch_file, "w") as f:
            f.write(sh_script)

        print(f"💾 Testing script saved in {sbatch_file}")
        return sbatch_file
    
    def predict_lstchain_data(self, input_file, output_file):
        pass
    
    def predict_data(self, input_file, output_file):
        pass
    
    def produce_irfs(self):
        pass
    
    def plot_irfs():
        #Use gammapy to plot the IRFs
        pass
    
    def plot_loss(self):
        set_mpl_style()
        import matplotlib.pyplot as plt
        import pandas as pd
        import glob
        direction_training_log = np.sort(glob.glob(f"{self.direction_model.model_dir}/{self.direction_model.model_nickname}_v*/training_log.csv"))[-1]
        energy_training_log = np.sort(glob.glob(f"{self.energy_model.model_dir}/{self.energy_model.model_nickname}_v*/training_log.csv"))[-1]
        type_training_log = np.sort(glob.glob(f"{self.type_model.model_dir}/{self.type_model.model_nickname}_v*/training_log.csv"))[-1]
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, training_log, model in zip(axs, [direction_training_log, energy_training_log, type_training_log], [self.direction_model, self.energy_model, self.type_model]):
            df = pd.read_csv(training_log)
            ax.plot(df['epoch'] + 1, df['loss'], label=f"Training")
            ax.plot(df['epoch'] + 1, df['val_loss'], label=f"Validation", ls='--')
            ax.set_title(f"{model.reco} training")
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
        plt.tight_layout()
        plt.show()


class TriModelCollection():
    
    def __init__(self, tri_models: list[CTLearnTriModelManager]):
        self.tri_models = tri_models
        
    def predict_lstchain_data(self, input_file, output_file, pointing_table='/dl1/event/telescope/parameters/LST_LSTCam'):
        closest_tri_model = self.find_closest_model_to(input_file, pointing_table)
        closest_tri_model.predict_lstchain_data(input_file, output_file)
        
    def predict_data(self, input_file, output_file, pointing_table):
        closest_tri_model = self.find_closest_model_to(input_file, pointing_table)
        closest_tri_model.predict_data(input_file, output_file)
        
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