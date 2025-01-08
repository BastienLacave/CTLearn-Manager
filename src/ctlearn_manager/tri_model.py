from astropy.table import QTable
import numpy as np
from pathlib import Path
from .model_manager import CTLearnModelManager
from .utils.utils import get_predict_data_sbatch_script, set_mpl_style
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
    
    
    
    def __init__(self, direction_model: CTLearnModelManager, energy_model: CTLearnModelManager, type_model: CTLearnModelManager):
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
        direction_channels = self.direction_model.model_parameters_table['channels'][0]
        energy_channels = self.energy_model.model_parameters_table['channels'][0]
        type_channels = self.type_model.model_parameters_table['channels'][0]
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
            matching_dirs = [gamma_dirs[i] for i in range(len(gamma_dirs)) if gamma_zeniths[i] == zenith and gamma_azimuths[i] == azimuth]
            if not matching_dirs:
                raise ValueError(f"No matching gamma directory found for zenith {zenith} and azimuth {azimuth}")
            gamma_dir = matching_dirs[0]
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
        type_model_dir = np.sort(glob.glob(f"{self.type_model.model_parameters_table['model_dir'][0]}/{self.type_model.model_nickname}_v*"))[-1]
        energy_model_dir = np.sort(glob.glob(f"{self.energy_model.model_parameters_table['model_dir'][0]}/{self.energy_model.model_nickname}_v*"))[-1]
        direction_model_dir = np.sort(glob.glob(f"{self.direction_model.model_parameters_table['model_dir'][0]}/{self.direction_model.model_nickname}_v*"))[-1]
        
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
            dl2_gamma.append(load_DL2_data_MC(file))
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
            dl2_gamma.append(load_DL2_data_MC(file))
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
        
        
