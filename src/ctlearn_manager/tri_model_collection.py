import numpy as np

from .utils.utils import (
    ClusterConfiguration,
    angular_distance,
    get_files,
)

__all__ = ['TriModelCollection']

class TriModelCollection:
    
    def __init__(self, tri_models: list, cluster_configuration=ClusterConfiguration()):
        self.tri_models = tri_models
        self.cluster_configuration = cluster_configuration
        for tri_model in self.tri_models:
            tri_model.cluster_configuration = cluster_configuration

    def predict_lstchain_run(self, run, output_dir, DL1_data_dir="/fefs/aswg/data/real/DL1/", overwrite=False):
        input_files = get_files(run, DL1_data_dir)
        for input_file in input_files:
            print(f"🔮 Predicting {input_file.split('/')[-1]}")
            subrun = int(input_file.split('.')[-2])
            output_file = f"{output_dir}/LST-1.Run{run:05d}.{subrun:04d}.dl2.h5"
            self.predict_lstchain_data(input_file, output_file, config_dir=output_dir, overwrite=overwrite, run=run, subrun=subrun)
        
    def predict_lstchain_data(self, input_file, output_file, pointing_table='/dl1/event/telescope/parameters/LST_LSTCam', config_dir=None, overwrite=False, run=None, subrun=None, plot=False):
        closest_tri_model = self.find_closest_model_to(input_file, pointing_table, plot=plot)
        if closest_tri_model is not None:
            closest_tri_model.predict_lstchain_data(input_file, output_file, config_dir=config_dir, overwrite=overwrite, run=run, subrun=subrun, pointing_table=pointing_table)
        else:
            return
        
    def predict_data(self, input_file, output_file, pointing_table='dl0/monitoring/subarray/pointing', config_dir=None, overwrite=False, plot=False):
        closest_tri_model = self.find_closest_model_to(input_file, pointing_table, plot=plot)
        if closest_tri_model is not None:
            closest_tri_model.predict_data(input_file, output_file, config_dir=config_dir, overwrite=overwrite, pointing_table=pointing_table)
        else:
            return
        
    def find_closest_model_to(self, input_file, pointing_table, plot=False):
        import astropy.units as u

        from ctlearn_manager.utils.utils import get_avg_pointing
        try:
            avg_data_ze, avg_data_az = get_avg_pointing(input_file, pointing_table=pointing_table)
        except:
            print(f"⚠️ Corrupted file, skipping : {input_file}")
            return
        
        avg_model_azs = []
        avg_model_zes = []
        for tri_model in self.tri_models:
            avg_model_azs.append(np.mean(tri_model.direction_model.validity.azimuth_range).to(u.deg).value)
            avg_model_zes.append(np.mean(tri_model.direction_model.validity.zenith_range).to(u.deg).value)
        avg_model_azs = np.array(avg_model_azs) * u.deg
        avg_model_zes = np.array(avg_model_zes) * u.deg
        # angular_distance_matrix = angular_distance(avg_data_ze, avg_data_az, avg_model_zes, avg_model_azs)
        closest_model_index = np.argmin(angular_distance(avg_data_ze, avg_data_az, avg_model_zes, avg_model_azs))
        closest_model = self.tri_models[closest_model_index]

        print(f"File : {input_file.split('/')[-1]}\tPointing : ({avg_data_ze:3f}, {avg_data_az:3f})\tModel : ({np.mean(closest_model.direction_model.validity.zenith_range).value}, {np.mean(closest_model.direction_model.validity.azimuth_range).value})")
        # print(f"｜📡 Average pointing of {input_file.split('/')[-1]} : ({avg_data_ze:3f}, {avg_data_az:3f})")
        # print(f"｜🔍 Closest model avg node : ({np.mean(closest_model.direction_model.validity.zenith_range).value}, {np.mean(closest_model.direction_model.validity.azimuth_range).value})")
        # print(f"｜🧠 Using models {closest_model.direction_model.model_nickname}, {closest_model.energy_model.model_nickname} and {closest_model.type_model.model_nickname}")
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            closest_model.direction_model.plot_zenith_azimuth_ranges(ax)
            ax.scatter(avg_data_az.to(u.rad), avg_data_ze, label='Average pointing', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3])
            ax.legend()
            plt.show()
        return closest_model


    def plot_zenith_azimuth_ranges(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        for tri_model in self.tri_models:
            tri_model.direction_model.plot_zenith_azimuth_ranges(ax)
        plt.show()