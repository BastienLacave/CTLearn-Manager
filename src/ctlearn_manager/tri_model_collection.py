import numpy as np
from .utils.utils import angular_distance, get_avg_pointing, get_files


class TriModelCollection():
    
    def __init__(self, tri_models: list):
        self.tri_models = tri_models

    def predict_lstchain_run(self, run, output_dir, DL1_data_dir="/fefs/aswg/data/real/DL1/", sbatch_scripts_dir=None, cluster=None, account=None, python_env=None, overwrite=False):
        input_files = get_files(run, DL1_data_dir)
        for input_file in input_files:
            print(f"🔮 Predicting {input_file.split('/')[-1]}")
            subrun = int(input_file.split('.')[-2])
            output_file = f"{output_dir}/LST-1.Run{run:05d}.{subrun:04d}.dl2.h5"
            self.predict_lstchain_data(input_file, output_file, sbatch_scripts_dir=sbatch_scripts_dir, cluster=cluster, account=account, python_env=python_env, overwrite=overwrite, run=run, subrun=subrun)
        
    def predict_lstchain_data(self, input_file, output_file, pointing_table='/dl1/event/telescope/parameters/LST_LSTCam', sbatch_scripts_dir=None, cluster=None, account=None, python_env=None, overwrite=False, run=None, subrun=None):
        closest_tri_model = self.find_closest_model_to(input_file, pointing_table)
        closest_tri_model.predict_lstchain_data(input_file, output_file, sbatch_scripts_dir=sbatch_scripts_dir, cluster=cluster, account=account, python_env=python_env, overwrite=overwrite, run=run, subrun=subrun, pointing_table=pointing_table)
        
    def predict_data(self, input_file, output_file, pointing_table='dl0/monitoring/subarray/pointing', sbatch_scripts_dir=None, cluster=None, account=None, python_env='ctlearn-cluster', overwrite=False):
        closest_tri_model = self.find_closest_model_to(input_file, pointing_table)
        closest_tri_model.predict_data(input_file, output_file, sbatch_scripts_dir=sbatch_scripts_dir, cluster=cluster, account=account, python_env=python_env, overwrite=overwrite, pointing_table=pointing_table)
        
    def find_closest_model_to(self, input_file, pointing_table):
        import astropy.units as u
        avg_data_ze, avg_data_az = get_avg_pointing(input_file, pointing_table=pointing_table)
        print(f"｜📡 Average pointing of {input_file.split('/')[-1]} : ({avg_data_ze:3f}, {avg_data_az:3f})")
        avg_model_azs = []
        avg_model_zes = []
        for tri_model in self.tri_models:
            avg_model_azs.append(np.mean((tri_model.direction_model.validity.azimuth_range)).to(u.deg).value)
            avg_model_zes.append(np.mean((tri_model.direction_model.validity.zenith_range)).to(u.deg).value)
        closest_model_index = np.argmin(angular_distance(avg_data_ze, avg_data_az, avg_model_zes, avg_model_azs))
        closest_model = self.tri_models[closest_model_index]
        print(f"｜🔍 Closest model avg node : ({np.mean(closest_model.direction_model.validity.zenith_range).value}, {np.mean(closest_model.direction_model.validity.azimuth_range).value})")
        print(f"｜🧠 Using models {closest_model.direction_model.model_nickname}, {closest_model.energy_model.model_nickname} and {closest_model.type_model.model_nickname}")
        return closest_model