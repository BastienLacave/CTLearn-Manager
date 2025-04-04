import numpy as np
import glob
from enum import Enum
# from numba import njit
# from astropy.coordinates import SkyCoord, AltAz
# import astropy.units as u
# from astropy.time import Time
# from astropy.coordinates import EarthLocation

__all__ = ['set_mpl_style', 'angular_distance', 'get_dates_from_runs', 'get_files', 'get_avg_pointing', 'get_predict_data_sbatch_script', 'remove_model_from_index', 'ClusterConfiguration', 'calc_flux_for_N_sigma', 'find_68_percent_range', 'ClusterConfiguration', 'ParticleType', 'get_current_env', 'DataSample']


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
    testing_files = np.sort(glob.glob(f"{DL1_data_dir}/{date}/v0.10/tailcut84/dl1_LST-1.Run{run:05d}.*.h5"))
    print(f"{len(testing_files)} files found for run {run:05d}")
    return testing_files

def get_avg_pointing(input_file, pointing_table='/dl1/event/telescope/parameters/LST_LSTCam'):
    from ctapipe.io import read_table
    import astropy.units as u
    pointing = read_table(input_file, path=pointing_table)
    avg_data_az = np.mean(pointing['az_tel']*180/np.pi)
    avg_data_ze = np.mean(90 - pointing['alt_tel']*180/np.pi)
    return avg_data_ze, avg_data_az

def get_predict_data_sbatch_script(cluster, command, job_name, sbatch_scripts_dir, account, env_name, time, partition, nodes=1, memory_mb=None):
    if memory_mb==None:
        memory_mb = 64000
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
#SBATCH --partition={partition}
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:{nodes}
#SBATCH --mem={memory_mb}0mb
#SBATCH --output={sbatch_scripts_dir}/{job_name}.%x.%j.out
#SBATCH --error={sbatch_scripts_dir}/{job_name}.%x.%j.err
#SBATCH --account={account}

srun --environment={env_name} {command}
''',
    'lst-cluster':f'''#!/bin/bash -l
#
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --mem={memory_mb}mb
#SBATCH -o {sbatch_scripts_dir}/{job_name}%x.%j.out
#SBATCH -e {sbatch_scripts_dir}/{job_name}%x.%j.err 

source ~/.bashrc
conda activate {env_name}
echo $CONDA_DEFAULT_ENV
echo $SLURM_ARRAY_TASK_ID

srun {command}
''',
                    
    }
    if cluster not in sbatch_predict_data_configs:
        raise ValueError(f"Cluster {cluster} not supported. Supported clusters are: {sbatch_predict_data_configs.keys()}\nIf you wish not to use any slurm job managment system, set use_cluster=False in the ClusterConfiguration object")
    return sbatch_predict_data_configs[cluster]

def remove_model_from_index(model_nickname, MODEL_INDEX_FILE):
    import h5py
    with h5py.File(MODEL_INDEX_FILE, 'a') as f:
        try:
            del f[model_nickname]
            print(f"Model {model_nickname} removed from index")
        except:
            print(f"Model {model_nickname} not found in index")


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
    def __init__(self, account=None, environment=None, use_cluster=True, partition=None, time=None, nodes=1, memory_mb=None):
        

        # self.current_env = 
        self.use_cluster = use_cluster
        config = self.get_cluster()
        self.cluster = config['cluster']
        self.account = account if account!=None else config['account']
        self.environment = environment if environment!=None else get_current_env()
        self.partition = partition if partition!=None else config['partition']
        self.time = time if time!=None else config['time']
        self.nodes = nodes
        self.memory_mb = memory_mb
        # if self.use_cluster:
        #     print(f"🔧 Using cluster {self.cluster} with account {self.account} and python environment {self.python_env}")

    def info(self):
        if self.use_cluster:
            print(f"🔧 Using cluster {self.cluster} \tAccount : {self.account} \tEnvironment : {self.environment} \tPartition : {self.partition} \tTime limi : {self.time}")
        else:
            print("🔧 Not using any cluster")

    def get_cluster(self):
        import socket
        host_name = socket.gethostname()

        #TODO fix ln001 etc for diff login nodes
        if host_name.startswith("daint-ln"):
            host_name = "daint"

        match host_name:
            case "ui.cta.camk.edu.pl":
                cluster = 'camk'
                account = None
                partition = None
                time = '03:00:00'
            case "daint":
                cluster = 'cscs'
                account = 'cta08'
                partition = 'normal'
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
        if self.use_cluster:
            self.use_cluster = cluster!=None
        return {"cluster": cluster, "account": account, "partition": partition, "time": time}

    

    def write_sbatch_script(self, job_name, cmd, sbatch_scripts_dir):
        import os
        if not os.path.exists(sbatch_scripts_dir):
            os.system(f"mkdir {sbatch_scripts_dir}")
        sh_script = get_predict_data_sbatch_script(self.cluster, cmd, job_name, sbatch_scripts_dir, self.account, self.environment, self.time, self.partition, self.nodes, self.memory_mb)
        sbatch_file = f"{sbatch_scripts_dir}/{job_name}.sh"
        with open(sbatch_file, "w") as f:
            f.write(sh_script)

        print(f"SBATCH script saved in {sbatch_file}")
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


class ParticleType(Enum):
    GAMMA_POINT = "gamma_point"
    GAMMA_DIFFUSE = "gamma_diffuse"
    PROTON = "proton"
    ELECTRON = "electron"
    # REAL_DATA = "real_data"
    # ALL = "all"

class DataSample:
    """
    A class to represent a training sample for CTLearn.
    :param directory: The directory where training data is stored.
    :type directory: str
    :param pattern: The pattern to match training files.
    :type pattern: str
    :param zenith_distance: The zenith distance of the training sample.
    :type zenith_distance: astropy.units.Quantity
    :param azimuth: The azimuth of the training sample.
    :type azimuth: astropy.units.Quantity
    :param energy_range: The energy range of the training sample.
    :type energy_range: list of astropy.units.Quantity
    :param nsb_range: The NSB (Night Sky Background) range of the training sample.
    :type nsb_range: list of astropy.units.Quantity
    """

    import astropy.units as u

    @u.quantity_input(zenith_distance=u.deg, azimuth=u.deg, energy_range=u.TeV, nsb_range=u.Hz)
    def __init__(self, directory, pattern, particle_type: ParticleType | None = None, zenith_distance=np.nan * u.deg, azimuth=np.nan * u.deg, energy_range=[np.nan, np.nan] * u.TeV, nsb_range=[np.nan, np.nan] * u.Hz):
        """
        Initialize the ModelManager.
        :param directory: The directory where training data is stored.
        :type directory: str
        :param pattern: The pattern to match training files.
        :type pattern: str
        :param zenith_distance: The zenith distance for training data, defaults to NaN degrees.
        :type zenith_distance: astropy.units.Quantity
        :param azimuth: The azimuth for training data, defaults to NaN degrees.
        :type azimuth: astropy.units.Quantity
        :param energy_range: The energy range for training data, defaults to [NaN, NaN] TeV.
        :type energy_range: list of astropy.units.Quantity
        :param nsb_range: The NSB range for training data, defaults to [NaN, NaN] Hz.
        :type nsb_range: list of astropy.units.Quantity
        """
        import astropy.units as u
        from ctapipe.io import read_table
        from tqdm import tqdm

        self.directory = directory
        self.pattern = pattern
        self.energy_range = energy_range
        self.nsb_range = nsb_range

        files = np.sort(glob.glob(f"{directory}/{pattern}"))
        if len(files) == 0:
            raise ValueError(f"No files found matching {directory}/{pattern}")
        
        i = 0
        for file in tqdm(files, desc="Checking files for particle type and pointing", unit="file"):
            shower_parameters = read_table(file, "simulation/event/subarray/shower")
            pointing = read_table(file, "configuration/telescope/pointing/tel_001") 
            particle_id = np.unique(shower_parameters["true_shower_primary_id"])
            
            zenith_distance = np.unique(90 * u.deg - pointing["telescope_pointing_altitude"].to(u.deg))
            azimuth = np.unique(pointing["telescope_pointing_azimuth"].to(u.deg))

            assert len(zenith_distance) == 1, f"More than one zenith distance found in {file}"
            assert len(azimuth) == 1, f"More than one azimuth found in {file}"
            assert len(particle_id) == 1, f"More than one particle ID found in {file}"

            if i == 0:  
                first_particle_type = particle_id[0]
                first_zenith_distance = zenith_distance[0]
                first_azimuth = azimuth[0]
            else:
                assert first_particle_type == particle_id[0], f"Different particle types found in {file} and {files[0]}"
                assert first_zenith_distance == zenith_distance[0], f"Different zenith distances found in {file} and {files[0]}"
                assert first_azimuth == azimuth[0], f"Different azimuths found in {file} and {files[0]}"
            i += 1

        self.zenith_distance = np.round(first_zenith_distance.to(u.deg).value, 4) * u.deg
        self.azimuth = np.round(first_azimuth.to(u.deg).value, 4) * u.deg

        match particle_id[0]:
            case 0:
                run = read_table(file, "configuration/simulation/run")
                max_viewcone = np.unique(run["max_viewcone_radius"])
                if max_viewcone > 0.5 * u.deg:
                    self.particle_type = ParticleType.GAMMA_DIFFUSE
                else:
                    self.particle_type = ParticleType.GAMMA_POINT
            case 1:
                self.particle_type = ParticleType.ELECTRON
            case 101:
                self.particle_type = ParticleType.PROTON
            case _:
                raise ValueError(f"Unknown particle ID: {particle_id}")
        
        print(f"DataSample : Particle type: {self.particle_type.value} (ZD, Az): ({self.zenith_distance}, {self.azimuth})")

        
