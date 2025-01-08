import numpy as np
from .utils.utils import angular_distance


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
            avg_model_azs.append(np.mean((tri_model.direction_model.az_range))) # FIXME read the ranges from validity
            avg_model_zes.append(np.mean((tri_model.direction_model.zd_range)))
        print(f"🔍 Closest model avg node : ({avg_model_zes[np.argmin(np.abs(avg_model_zes - avg_data_ze))]:3f}, {avg_model_azs[np.argmin(np.abs(avg_model_azs - avg_data_az))]:3f})")
        closest_model_index = np.argmin(angular_distance(avg_data_ze, avg_data_az, avg_model_zes, avg_model_azs))
        closest_model = self.tri_models[closest_model_index]
        return closest_model