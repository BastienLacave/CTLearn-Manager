from ..model_manager import CTLearnModelManager
from astropy.table import QTable
import numpy as np

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

def load_DL2_data(input_file, tel_id):
    from ctapipe.io import read_table
    from astropy.table import (join, hstack)
    pointing = read_table(input_file, f"dl1/monitoring/telescope/pointing/tel_{tel_id:03d}")
    pointing.sort('time')
    dl2_classification = read_table(input_file, f"dl2/event/telescope/classification/CTLearn/tel_{tel_id:03d}")
    dl2_energy = read_table(input_file, f"dl2/event/telescope/energy/CTLearn/tel_{tel_id:03d}")
    dl2_geometry = read_table(input_file, f"dl2/event/telescope/geometry/CTLearn/tel_{tel_id:03d}")
    dl2 = join(dl2_classification, dl2_energy, keys=["obs_id", "event_id"])
    dl2 = join(dl2, dl2_geometry, keys=["obs_id", "event_id"])
    dl2.sort('event_id')
    dl2 = hstack([dl2, pointing])
    dl2.sort('time')
    times = np.array(dl2['time'])
    t_diff = times[1:] - times[:-1] #np.diff(dl2['time'])
    t_diff = np.insert(t_diff, 0, 0)  # Insert 0 at the beginning to align with the original times array
    dl2['delta_t'] = t_diff
    print(f"Loaded {len(dl2)} events")
    return dl2