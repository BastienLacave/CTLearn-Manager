"""Example module of the python package template."""
from .version import __version__
from astropy.table import QTable
import numpy as np
from pathlib import Path

__all__ = [
    "__version__",
    "fibonacci",
]


class CTLearnModelManager():

    def __init__(self, model_parameters, MODEL_INDEX_FILE):
        self.model_index_file = MODEL_INDEX_FILE
        self.model_nickname = model_parameters['model_nickname']
        self.notes = model_parameters['notes']
        self.model_dir = f"{model_parameters['model_dir']}/{model_parameters['model_nickname']}"
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
        
        
    def launch_training(self, n_epochs=None):
        n_epoch_training = self.get_n_epoch_trained()
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
        models_dir = np.sort(glob.glob(f"{self.model_dir}*"))
        load_model = False
        if len(models_dir) > 0 :
            last_model_dir = Path(models_dir[-1])
            size = sum(f.stat().st_size for f in last_model_dir.glob('**/*') if f.is_file())
            model_version = int(models_dir[-1].split("_v")[-1])
            if size > 1e6:
                model_version += 1
                print(f"➡️ Model already exists: will continue training and create {self.model_nickname}_v{model_version}")
                save_best_validation_only = True
                model_dir = f"{self.model_dir}_v{model_version}/"
                model_to_load = f"{self.model_dir}_v{model_version - 1}/ctlearn_model.cpk/"
                load_model = True
                os.system(f"mkdir -p {model_dir}")
            else :
                model_dir = f"{self.model_dir}_v{model_version}/"
                if model_version > 0:
                    model_to_load = f"{self.model_dir}_v{model_version - 1}/ctlearn_model.cpk/"
                    load_model = True
                    print(f"➡️ Model already exists: will continue training and create {self.model_nickname}_v{model_version}")
                    save_best_validation_only = True
                else:
                    print(f"🆕 Model does not exist: will create {self.model_nickname}_v{model_version}")
                    save_best_validation_only = False
        else:
            model_version = 0
            print(f"🆕 Model does not exist: will create {self.model_nickname}_v{model_version}")
            model_dir = f"{self.model_dir}_v{model_version}/"
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
        import glob
        import pandas as pd
        training_logs = np.sort(glob.glob(f"{self.model_dir}_v*/training_log.csv"))
        n_epochs = 0
        for training_log in training_logs:
            df = pd.read_csv(training_log)
            n_epochs += len(df)
        return n_epochs

    
    def plot_loss(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        import glob
        set_mpl_style()
        training_logs = np.sort(glob.glob(f"{self.model_dir}_v*/training_log.csv"))
        losses_train = []
        losses_val = []
        for training_log in training_logs:
            df = pd.read_csv(training_log)
            losses_train = np.concatenate((losses_train, df['loss'].to_numpy()))
            losses_val = np.concatenate((losses_val, df['val_loss'].to_numpy()))
        epochs = np.arange(1, len(losses_train)+1)
        plt.plot(epochs, losses_train, label=f"Training")
        plt.plot(epochs, losses_val, label=f"Testing", ls='--')
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
        
    def launch_testing(self):
        pass
    
    def produce_irfs(self):
        pass
    
    def plot_irfs():
        #Use gammapy to plot the IRFs
        pass
    
    def plot_loss(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        import glob
        direction_training_log = np.sort(glob.glob(f"{self.direction_model.model_dir}_v*/training_log.csv"))[-1]
        energy_training_log = np.sort(glob.glob(f"{self.energy_model.model_dir}_v*/training_log.csv"))[-1]
        type_training_log = np.sort(glob.glob(f"{self.type_model.model_dir}_v*/training_log.csv"))[-1]
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        for i, training_log in enumerate([direction_training_log, energy_training_log, type_training_log]):
            if Path(training_log).exists():
                df = pd.read_csv(training_log)
                axs[i].plot(df['epoch'], df['loss'], label=f"Training")
                axs[i].plot(df['epoch'], df['val_loss'], label=f"Testing", ls='--')
                axs[i].set_title(f"{self.direction_model.reco} training")
                axs[i].set_xlabel('Epoch')
                axs[i].set_ylabel('Loss')
                axs[i].legend()
            else:
                print(f"Model has not yet been trained.")

      
def load_model_from_index(model_nickname, MODEL_INDEX_FILE):
    models_table = QTable.read(MODEL_INDEX_FILE)
    model_index = np.where(models_table['model_nickname'] == model_nickname)[0][0]
    model_parameters = {key: models_table[key][model_index] for key in models_table.colnames}
    model = CTLearnModelManager(model_parameters, MODEL_INDEX_FILE)
    return model

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
    rcParams['font.sans-serifs'] = prop.get_name()
    rcParams['font.family'] = prop.get_name()
    with pkg_resources.path(resources, 'ctlearnStyle.mplstyle') as style_path:
        plt.style.use(style_path)
    # plt.style.use('./resources/ctlearnStyle.mplstyle')