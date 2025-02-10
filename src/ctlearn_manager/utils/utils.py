import numpy as np
import glob
# from numba import njit
# from astropy.coordinates import SkyCoord, AltAz
# import astropy.units as u
# from astropy.time import Time
# from astropy.coordinates import EarthLocation


def set_mpl_style():
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as font_manager
    from matplotlib import rcParams
    from .. import resources


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

def get_dates_from_runs(runs):
    dates_ = np.empty(len(runs))
    for i, run in enumerate(runs):
        pattern = f'/fefs/aswg/data/real/R0V/*/LST-1.1.Run{run:05d}.0000.fits.fz'
        file = glob.glob(pattern)
        date = file[0].split('/')[-2]
        dates_[i] = int(date)
    return runs, dates_.astype(int)

def get_files(run, DL1_data_dir):
    date = get_dates_from_runs([run])[1][0]
    testing_files = np.sort(glob.glob(f"{DL1_data_dir}/{date}/v0.9/tailcut84/dl1_LST-1.Run{run:05d}.*.h5"))
    print(f"{len(testing_files)} files found for run {run:05d}")
    return testing_files

def get_avg_pointing(input_file, pointing_table='/dl1/event/telescope/parameters/LST_LSTCam'):
    from ctapipe.io import read_table
    import astropy.units as u
    pointing = read_table(input_file, path=pointing_table)
    avg_data_az = np.mean(pointing['az_tel']*180/np.pi)
    avg_data_ze = np.mean(90 - pointing['alt_tel']*180/np.pi)
    return avg_data_ze, avg_data_az

def get_predict_data_sbatch_script(cluster, command, job_name, sbatch_scripts_dir, account, env_name, time, partition):
    sbatch_predict_data_configs = {
    'camk': 
    f'''#!/bin/sh
#SBATCH --time={time}
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
#SBATCH --time={time}
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
    'lst-cluster':f'''#!/bin/bash -l
#
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --mem=64000mb
#SBATCH -o {sbatch_scripts_dir}/{job_name}%x.%j.out
#SBATCH -e {sbatch_scripts_dir}/{job_name}%x.%j.err 

source ~/.bashrc
conda activate {env_name}
echo $CONDA_DEFAULT_ENV
echo $SLURM_ARRAY_TASK_ID

srun {command}
''',
                    
    }
    return sbatch_predict_data_configs[cluster]

def remove_model_from_index(model_nickname, MODEL_INDEX_FILE):
    import h5py
    with h5py.File(MODEL_INDEX_FILE, 'a') as f:
        del f[model_nickname]


# def write_sbatch_script(cluster_configuration: ClusterConfiguration, job_name, cmd, sbatch_scripts_dir):
#     sh_script = get_predict_data_sbatch_script(cluster_configuration.cluster, cmd, job_name, sbatch_scripts_dir, cluster_configuration.account, cluster_configuration.env_name)
#     sbatch_file = f"{sbatch_scripts_dir}/{job_name}.sh"
#     with open(sbatch_file, "w") as f:
#         f.write(sh_script)

#     print(f"💾 Testing script saved in {sbatch_file}")
#     return sbatch_file

# @njit
# def transform_coordinates(alt, az, obstime_unix, location_lat, location_lon, location_height, pressure, temperature, relative_humidity, source_position_ra, source_position_dec):
#     n = len(alt)
#     transformed_ra = np.empty(n, dtype=np.float64)
#     transformed_dec = np.empty(n, dtype=np.float64)
#     for i in range(n):
#         frame = AltAz(obstime=Time(obstime_unix[i], format='unix'), location=EarthLocation(lat=location_lat, lon=location_lon, height=location_height), pressure=pressure, temperature=temperature, relative_humidity=relative_humidity)
#         reco_temp = SkyCoord(alt=alt[i]*u.deg, az=az[i]*u.deg, frame=frame)
#         transformed_reco = reco_temp.transform_to(SkyCoord(ra=source_position_ra*u.deg, dec=source_position_dec*u.deg, frame='icrs'))
#         transformed_ra[i] = transformed_reco.ra.deg
#         transformed_dec[i] = transformed_reco.dec.deg
#     return transformed_ra, transformed_dec
def get_current_env():
    import os
    return os.environ.get('CONDA_DEFAULT_ENV') or os.environ.get('VIRTUAL_ENV')

class ClusterConfiguration():
    def __init__(self, account=None, python_env=None, use_cluster=True, partition=None, time=None):
        

        # self.current_env = 
        self.use_cluster = use_cluster
        config = self.get_cluster()
        self.cluster = config['cluster']
        self.account = account if account!=None else config['account']
        self.python_env = python_env if python_env!=None else get_current_env()
        self.partition = partition if partition!=None else config['partition']
        self.time = time if time!=None else config['time']
        if self.use_cluster:
            print(f"🔧 Using cluster {self.cluster} with account {self.account} and python environment {self.python_env}")

    def get_cluster(self):
        import socket
        host_name = socket.gethostname()

        match host_name:
            case "ui.cta.camk.edu.pl":
                cluster = 'camk'
                account = None
                partition = None
                time = '03:00:00'
            case "daint.alps.cscs.ch":
                cluster = 'cscs'
                account = 'cta04'
                partition = 'gpu'
                time = '24:00:00'
            case "cp02":
                cluster = 'lst-cluster'
                account = 'aswg'
                partition = 'long'
                time = '24:00:00'
            case _:
                cluster = None
                account = None
                partition = None
                time = None
        # self.use_cluster = cluster!=None
        return {"cluster": cluster, "account": account, "partition": partition, "time": time}

    

    def write_sbatch_script(self, job_name, cmd, sbatch_scripts_dir):
        sh_script = get_predict_data_sbatch_script(self.cluster, cmd, job_name, sbatch_scripts_dir, self.account, self.python_env, self.time, self.partition)
        sbatch_file = f"{sbatch_scripts_dir}/{job_name}.sh"
        with open(sbatch_file, "w") as f:
            f.write(sh_script)

        print(f"💾 Testing script saved in {sbatch_file}")
        return sbatch_file

def calc_flux_for_N_sigma(N_sigma, cumul_excess, cumul_off, 
                          min_signi, min_exc, min_off_events, alpha,
                          target_obs_time, actual_obs_time, cond=True):
    import astropy.units as u
    from pyirf.statistics import li_ma_significance

    time_factor = target_obs_time.to(u.h) / actual_obs_time.to(u.h)

    start_flux = 1
    flux_factor = start_flux * np.ones_like(cumul_excess)

    good_bin_mask = ((cumul_excess > min_exc*cumul_off) &
                    (cumul_off > min_off_events) &
                    (cumul_excess > 10))
    # print(good_bin_mask)

    if cond:
        flux_factor = np.where(good_bin_mask, flux_factor, np.nan)
    
    # First calculate significance (with 1 off) of the excesses in the provided sample, with no scaling.
    # We will only use the cut combinations where we have at least min_signi sigmas to begin with...
    # NOTE!!! float64 precision is essential for the arguments of li_ma_significance!

    lima_signi = li_ma_significance((flux_factor*cumul_excess + cumul_off).astype('float64'), 
                                    cumul_off.astype('float64'), 
                                    alpha=1)
            
    # Set nan in bins where we do not reach min_signi:
    if cond:
        flux_factor = np.where(lima_signi > min_signi, flux_factor, np.nan)

    
    # Now calculate the significance for the target observation time_
    lima_signi = li_ma_significance((time_factor*(flux_factor*cumul_excess +
                                                cumul_off)).astype('float64'), 
                                    (time_factor*cumul_off/alpha).astype('float64'), 
                                    alpha=alpha)

    
    # iterate to obtain the flux which gives exactly N_sigma:
    for iter in range(10):
        # print(iter)
        tolerance_mask = np.abs(lima_signi-N_sigma)>0.001 # recalculate only what is needed
        flux_factor[tolerance_mask] *= (N_sigma / lima_signi[tolerance_mask])
        # NOTE!!! float64 precision is essential here!!!!
        lima_signi[tolerance_mask] = li_ma_significance((time_factor*(flux_factor[tolerance_mask]*
                                                                    cumul_excess[tolerance_mask]+
                                                                    cumul_off[tolerance_mask])).astype('float64'), 
                                                        (time_factor*cumul_off[tolerance_mask]/alpha).astype('float64'), 
                                                        alpha=alpha)
    # print(lima_signi)
    return flux_factor, lima_signi

def find_68_percent_range(bin_heights, bin_edges, a=0.68):
    # data = np.random.exponential(scale=0.1, size=1000)  # Example positive-only data

    # Create the histogram
    # bin_heights, bin_edges = np.histogram(data, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)
    bin_heights = bin_heights / np.sum(bin_heights)

    bin_heights[bin_heights < 0] = 0  # Remove any negative values

    # Calculate the cumulative distribution function (CDF)
    cdf = []
    for i in range(len(bin_heights)):
        cdf.append(np.sum(bin_heights[:i]))
    # cdf = np.cumsum(bin_heights, axis=0)
    # print(cdf)
    # plt.plot(bin_centers, cdf/np.sum(bin_heights))
    # plt.show()

    # Find the value corresponding to 68% of the CDF
    upper_bound = np.interp(a, cdf/np.sum(bin_heights), bin_centers)
    return upper_bound
