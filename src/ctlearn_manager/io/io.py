from ..model_manager import CTLearnModelManager
from astropy.table import QTable
import numpy as np
import astropy.units as u
from astropy.time import TimeDelta
from numba import njit

def load_model_from_index(model_nickname, MODEL_INDEX_FILE):
    # models_table = QTable.read(MODEL_INDEX_FILE)
    # model_index = np.where(models_table['model_nickname'] == model_nickname)[0][0]
    model_parameters = {'model_nickname': model_nickname}
    model = CTLearnModelManager(model_parameters, MODEL_INDEX_FILE, load=True)
    return model


def load_DL2_data_MC(input_file):
    from ctapipe.io import read_table
    from astropy.table import (join, hstack)
    pointing = read_table(input_file, "dl1/monitoring/subarray/pointing/")
    dl2_classification = read_table(input_file, "dl2/event/subarray/classification/CTLearn")
    dl2_classification = hstack([dl2_classification, pointing])
    dl2_classification = dl2_classification[~np.isnan(dl2_classification["CTLearn_prediction"])]
    dl2_energy = read_table(input_file, "dl2/event/subarray/energy/CTLearn")
    dl2_energy = dl2_energy[~np.isnan(dl2_energy["CTLearn_energy"])]
    dl2_geometry = read_table(input_file, "dl2/event/subarray/geometry/CTLearn")
    dl2_geometry = dl2_geometry[~np.isnan(dl2_geometry["CTLearn_alt"])]
    dl2 = join(dl2_classification, dl2_energy, keys=["obs_id", "event_id"])
    dl2 = join(dl2, dl2_geometry, keys=["obs_id", "event_id"])
    return dl2

def load_true_shower_parameters(input_file):
    from ctapipe.io import read_table
    true_shower_parameters = read_table(input_file, "simulation/event/subarray/shower")
    return true_shower_parameters


@njit
def compute_diff(arr):
    n = len(arr)
    diff = np.empty(n, dtype=arr.dtype)
    diff[0] = 0  # Assuming the first difference is 0
    for i in range(1, n):
        diff[i] = arr[i] - arr[i - 1]
    return diff

def load_DL2_data(input_file, DL2DataProcessor):
    tel_id = DL2DataProcessor.telescope_id
    reco_method = DL2DataProcessor.reconstruction_method
    path = "subarray" if DL2DataProcessor.stereo else "telescope"
    tel = f"tel_{tel_id:03d}" if DL2DataProcessor.stereo else f"tel_{tel_id:03d}"
    from ctapipe.io import read_table
    from astropy.table import (join, hstack)
    pointing = read_table(input_file, f"dl1/monitoring/{path}/pointing/{tel}")
    pointing.sort('time')
    dl2_classification = read_table(input_file, f"dl2/event/{path}/classification/{reco_method}/{tel}")
    dl2_energy = read_table(input_file, f"dl2/event/{path}/energy/{reco_method}/{tel}")
    dl2_geometry = read_table(input_file, f"dl2/event/{path}/geometry/{reco_method}/{tel}")
    dl2 = join(dl2_classification, dl2_energy, keys=["obs_id", "event_id"])
    dl2 = join(dl2, dl2_geometry, keys=["obs_id", "event_id"])
    dl2.sort('event_id')
    dl2 = hstack([dl2, pointing])
    dl2.sort('time')
    # times = np.array(dl2['time'])
    print("Computing time differences...")
    # t_diff = np.diff(dl2['time'])#.to_value('unix')
    # t_diff = np.insert(t_diff, 0, TimeDelta(0*u.s, format='jd', scale='tai'))  # Insert 0 at the beginning to align with the original times array
    t_diff = compute_diff(dl2['time'].to_value('unix'))
    dl2['delta_t'] = t_diff
    print(f"Loaded {len(dl2)} events")
    return dl2