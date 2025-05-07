import fnmatch
import glob
from enum import Enum

# from numba import njit
# from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
import ctadata
import numpy as np
from astropy.table import Table

# from astropy.time import Time
# from astropy.coordinates import EarthLocation

__all__ = ['DefaultCuts', 'Cuts', 'CutType', 'get_irf_type_from_config',  'IRFType', 'CTLearnManagerStyle', 'set_mpl_style', 'angular_distance', 'get_dates_from_runs', 'get_files_LST_cluster', 'get_files_cscs', 'get_avg_pointing', 'get_predict_data_sbatch_script', 'remove_model_from_index', 'ClusterConfiguration', 'calc_flux_for_N_sigma', 'find_68_percent_range', 'ClusterConfiguration', 'ParticleType', 'get_current_env', 'DataSample']

class CTLearnManagerStyle(Enum):
    """
    A class to manage predefined colors and plot styles for consistent visualization.
    """



    # def __init__(self):
    #     # Predefined colors
    #     self.colors = {
    #         "ctlearn_1": "#016279" ,
    #         "ctlearn_2": "#00a693" ,
    #         "ctlearn_3": "#58c68b" ,
    #         "ctlearn_highlight": "#00c6ff" ,
    #         "ctlearn_accent_1": "#cf004b",
    #         "ctlearn_accent_2": "#923e51",
    #         "random_forest": "#000000" ,
    #     }
        # self.ctlearn_1 = "#016279" 
        # self.ctlearn_2 = "#00a693" 
        # self.ctlearn_3 = "#58c68b" 
        # self.ctlearn_highlight = "#00c6ff" 
        # self.ctlearn_accent_1 = "#cf004b"
        # self.ctlearn_accent_2 = "#923e51"
        # self.random_forest = "#000000" 
    ctlearn_1 = "#016279" 
    ctlearn_2 = "#00a693" 
    ctlearn_3 = "#58c68b" 
    ctlearn_highlight = "#00c6ff" 
    ctlearn_accent_1 = "#cf004b"
    ctlearn_accent_2 = "#923e51"
    random_forest = "#000000" 
        



def set_mpl_style():
    # font_path = "./resources/Outfit-Medium.ttf"
    import importlib.resources as pkg_resources

    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    from .. import resources

    with pkg_resources.path(resources, 'Outfit-Medium.ttf') as font_path:
        font_manager.fontManager.addfont(font_path)
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    rcParams['font.sans-serif'] = prop.get_name()
    rcParams['font.family'] = prop.get_name()
    with pkg_resources.path(resources, 'CTLearnStyle.mplstyle') as style_path:
        plt.style.use(style_path)
    # plt.style.use('./resources/ctlearnStyle.mplstyle')

@u.quantity_input(ze1=u.deg, az1=u.deg, ze2=u.deg, az2=u.deg)
def angular_distance(ze1, az1, ze2, az2):
    ze1, az1, ze2, az2 = map(np.radians, [ze1, az1, ze2, az2])
    delta_az = az2 - az1
    delta_ze = ze2 - ze1
    a = np.sin(delta_ze / 2)**2 + np.cos(ze1) * np.cos(ze2) * np.sin(delta_az / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c

def get_dates_from_runs(runs):
    # dates_ = np.empty(len(runs))
    # for i, run in enumerate(runs):
    #     pattern = f'/fefs/aswg/data/real/R0V/*/LST-1.1.Run{run:05d}.0000.fits.fz'
    #     file = glob.glob(pattern)
    #     date = file[0].split('/')[-2]
    #     dates_[i] = int(date)
    # return runs, dates_.astype(int)
    import importlib.resources as pkg_resources

    from .. import resources
    with pkg_resources.path(resources, 'LST_source_catalog.ecsv') as catalog_file:
        catalog_table = Table.read(catalog_file, format="ascii.ecsv")
    dates_ = np.empty(len(runs))
    for i, run in enumerate(runs):
        matching_row = catalog_table[catalog_table["Run ID"] == run]
        if len(matching_row) == 0:
            raise ValueError(f"Run {run} not found in the catalog table")
        dates_[i] = int(matching_row["Date directory"][0].replace("-", ""))
    return runs, dates_.astype(int)

def get_files_LST_cluster(run: int, DL1_data_dir="/fefs/aswg/data/real/DL1/"):
    date = get_dates_from_runs([run])[1][0]
    files = np.sort(glob.glob(f"{DL1_data_dir}/{date}/v0.10/tailcut84/dl1_LST-1.Run{run:05d}.*.h5"))
    print(f"{len(files)} files found for run {run:05d}")
    return files

def get_files_cscs(run: int, DL1_data_dir="/pnfs/cta.cscs.ch/lst/DL1/"):
    date = get_dates_from_runs([run])[1][0]
    try:
        v = 'v0.10'
        directory = f"{DL1_data_dir}/{date}/{v}/tailcut84/"
        all_files = ctadata.list_dir(directory)
    except:
        print("Version v0.10 not found, trying v0.9")
        v = 'v0.9'
        directory = f"{DL1_data_dir}/{date}/{v}/tailcut84/"
        all_files = ctadata.list_dir(directory)
    pattern = f"dl1_LST-1.Run{run:05d}.*.h5"
    files = fnmatch.filter(all_files, pattern)
    files = np.sort(files)
    files = [f"{directory}/{file}" for file in files]
    print(f"{len(files)} files found for run {run:05d}")
    return files, v
    

def get_avg_pointing(input_file, pointing_table='/dl1/event/telescope/parameters/LST_LSTCam', alt_key='alt_tel', az_key='az_tel'):
    import astropy.units as u
    from ctapipe.io import read_table
    pointing = read_table(input_file, path=pointing_table)
    avg_data_az = np.mean(pointing[az_key]*180/np.pi) * u.deg
    avg_data_ze = np.mean(90 - pointing[alt_key]*180/np.pi) * u.deg
    return avg_data_ze, avg_data_az

def get_predict_data_sbatch_script(cluster, command, job_name, sbatch_scripts_dir, account, env_name, time, partition, nodes=1, memory_mb=None, use_gpu_cscs=True):
    if memory_mb==None:
        memory_mb = 64000

    if use_gpu_cscs:
        gpu_string = f'''
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:{nodes}'''
    else:
        gpu_string = ""

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
#SBATCH --nodes=1
#SBATCH --ntasks=1
{gpu_string}
#SBATCH --mem={memory_mb}mb
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

class ClusterConfiguration:
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
            print(f"🔧 Using cluster {self.cluster} ||| Account : {self.account} ||| Environment : {self.environment} ||| Partition : {self.partition} ||| Time limit : {self.time} ||| Nodes : {self.nodes}")
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

    

    def write_sbatch_script(self, job_name, cmd, sbatch_scripts_dir, use_gpu_cscs=True):
        import os
        if not os.path.exists(sbatch_scripts_dir):
            os.system(f"mkdir {sbatch_scripts_dir}")
        sh_script = get_predict_data_sbatch_script(self.cluster, cmd, job_name, sbatch_scripts_dir, self.account, self.environment, self.time, self.partition, self.nodes, self.memory_mb, use_gpu_cscs=use_gpu_cscs)
        sbatch_file = f"{sbatch_scripts_dir}/{job_name}.sh"
        os.makedirs(sbatch_scripts_dir, exist_ok=True)
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

    start_flux = np.float64(1.)
    flux_factor = start_flux * np.ones_like(cumul_excess).astype('float64')

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
        tolerance_mask = np.abs(lima_signi.astype('float64')-N_sigma)> 0.001 # recalculate only what is needed
        flux_factor[tolerance_mask] *= (np.float64(N_sigma) / lima_signi[tolerance_mask].astype('float64'))
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
    def __init__(self, directory: str, pattern: str, particle_type: ParticleType | None = None, zenith_distance=np.nan * u.deg, azimuth=np.nan * u.deg, energy_range=[np.nan, np.nan] * u.TeV, nsb_range=[np.nan, np.nan] * u.Hz):
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

        




class CutType(Enum):
    GLOBAL = "global"
    EFFICIENCY_OPTIMIZED = "energy_dependent_efficiency"
    SENSITIVITY_OPTIMIZED = "sensitivity_optimized"

class IRFType(Enum):
    EFFICIENCY_OPTIMIZED = "energy_dependent_efficiency"
    SENSITIVITY_OPTIMIZED = "sensitivity_optimized"





class Cuts:
    def __init__(self, cut_type: CutType=CutType.GLOBAL, gammaness_cut: float=0., theta_cut: float=None, efficiency_gammaness: float=0.7, efficiency_theta: float=0.7):
        self.cut_type = cut_type

        if gammaness_cut is not None:
            if not (0 <= gammaness_cut <= 1):
                raise ValueError("gammaness_cut must be between 0 and 1.")
            
        if efficiency_gammaness is not None:
            if not (0 <= efficiency_gammaness <= 1):
                raise ValueError("efficiency_gammaness must be between 0 and 1.")
            
        if efficiency_theta is not None:
            if not (0 <= efficiency_theta <= 1):
                raise ValueError("efficiency_theta must be between 0 and 1.")
        
        match cut_type:
            case CutType.GLOBAL:
                if gammaness_cut is None and theta_cut is None:
                    raise ValueError("For GLOBAL cuts, at least one of gammaness_cut or theta_cut must be provided.")
                self.gammaness_cut = gammaness_cut
                self.theta_cut = theta_cut
                self.efficiency_gammaness = None
                self.efficiency_theta = None

            case CutType.EFFICIENCY_OPTIMIZED:
                if efficiency_gammaness is None and efficiency_theta is None:
                    raise ValueError("For ENERGY_DEPENDENT cuts, at least one of efficiency_gammaness or efficiency_theta must be provided.")
                self.gammaness_cut = None
                self.theta_cut = None
                self.efficiency_gammaness = efficiency_gammaness
                self.efficiency_theta = efficiency_theta
                self.irf_type = IRFType.EFFICIENCY_OPTIMIZED

            case CutType.SENSITIVITY_OPTIMIZED:
                if any(param is not None for param in [gammaness_cut, theta_cut, efficiency_gammaness, efficiency_theta]):
                    raise ValueError("For SENSITIVITY_OPTIMIZED cuts, no additional parameters should be provided.")
                self.gammaness_cut = None
                self.theta_cut = None
                self.efficiency_gammaness = None
                self.efficiency_theta = None
                self.irf_type = IRFType.SENSITIVITY_OPTIMIZED

            case _:
                raise ValueError(f"Invalid cut type: {cut_type}")

    def __str__(self):
        return (f"Cut Type: {self.cut_type.value}, "
                f"Gammas cut: {self.gammaness_cut}, "
                f"Theta cut: {self.theta_cut}, "
                f"Efficiency gammaness: {self.efficiency_gammaness}, "
                f"Efficiency theta: {self.efficiency_theta}")
    
    def plot_cuts_info_plt(self, ax):
        final_string = self.get_label()

        if final_string:
            ax.text(
            0.98, 0.02, final_string,
            transform=ax.transAxes,
            fontsize=9,
            color=CTLearnManagerStyle.ctlearn_1.value,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor=CTLearnManagerStyle.ctlearn_highlight.value, alpha=0.2),
            )

    def get_label(self):
        match self.cut_type:
            case CutType.GLOBAL:
                gammaness_cut_type = f"Glob. gcut : {self.gammaness_cut}" if self.gammaness_cut else ""
                theta_cut_type = r"Glob. $\theta$ cut : " + f"{self.theta_cut}" if self.theta_cut else ""
            case CutType.EFFICIENCY_OPTIMIZED:
                gammaness_cut_type = f"E dep. gcut : {self.efficiency_gammaness}eff." if self.efficiency_gammaness else ""
                theta_cut_type = r"E dep. $\theta$ cut : " + f"{self.efficiency_theta}eff." if self.efficiency_theta else ""
            case CutType.SENSITIVITY_OPTIMIZED:
                gammaness_cut_type = "Sensitivity optimized cuts"
                theta_cut_type = ""
            case _:
                gammaness_cut_type = ""
                theta_cut_type = ""

        final_string = f"{gammaness_cut_type} | {theta_cut_type}".strip(" | ")
        return final_string
    
class DefaultCuts(Enum):
    NO_CUTS = Cuts(cut_type=CutType.GLOBAL, gammaness_cut=0.0)



def get_irf_type_from_config(config):
    """
    Get the IRF type from the configuration.
    :param config: The configuration object.
    :type config: object
    :return: The IRF type.
    :rtype: IRFType
    """
    import yaml
    with open(config, 'r') as file:
        config_data = yaml.safe_load(file)
    optimization_algorithm = config_data['EventSelectionOptimizer']['optimization_algorithm']

    match optimization_algorithm:
        case "PercentileCuts":
            irf_type = IRFType.EFFICIENCY_OPTIMIZED
            gammaness_efficiency = config_data['GhPercentileCutCalculator']['target_percentile'] / 100
            theta_efficiency = config_data['ThetaPercentileCutCalculator']['target_percentile'] / 100
            return irf_type, gammaness_efficiency, theta_efficiency
        case "PointSourceSensitivityOptimizer":
            irf_type = IRFType.SENSITIVITY_OPTIMIZED
            gammaness_efficiency = None
            theta_efficiency = None
            return irf_type, gammaness_efficiency, theta_efficiency
        case _:
            raise ValueError(f"Unknown optimization algorithm: {optimization_algorithm}")
        