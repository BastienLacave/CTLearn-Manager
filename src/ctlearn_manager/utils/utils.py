import numpy as np
import glob


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

def get_predict_data_sbatch_script(cluster, command, job_name, sbatch_scripts_dir, account, env_name):
    sbatch_predict_data_configs = {
    'camk': 
    f'''#!/bin/sh
#SBATCH --time=03:00:00
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
#SBATCH --time=24:00:00
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
#SBATCH --account=aswg
#SBATCH --partition=long
#SBATCH --time=24:00:00
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

