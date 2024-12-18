from .CTLearnModelManager import CTLearnModelManager
from pathlib import Path
import numpy as np

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