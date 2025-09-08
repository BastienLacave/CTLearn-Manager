import astropy.units as u
import numpy as np
from numba import njit

from ..model_manager import CTLearnModelManager
from ..utils.utils import ClusterConfiguration, CTLMDirectories

__all__ = [
    # "load_model_from_index",
    "load_DL2_data_MC",
    "load_DL2_data",
    "load_DL2_data_RF",
    "load_true_shower_parameters",
]


# def load_model_from_index(
#     model_nickname, MODEL_INDEX_FILE, cluser_config=ClusterConfiguration()
# ):
#     # models_table = QTable.read(MODEL_INDEX_FILE)
#     # model_index = np.where(models_table['model_nickname'] == model_nickname)[0][0]
#     model_parameters = {"model_nickname": model_nickname}
#     from astropy.io.misc.hdf5 import read_table_hdf5

#     try:
#         read_table_hdf5(f"{MODEL_INDEX_FILE}", path=f"{model_nickname}/parameters")
#     except:
#         raise ValueError(f"Model {model_nickname} not found in {MODEL_INDEX_FILE}")
#     model = CTLearnModelManager(
#         model_parameters,
#         CTLMDirectories(model_nickname)
#         load=True,
#         cluster_configuration=cluser_config,
#     )
#     return model


def load_DL2_data_MC(input_file, tel_id=None):
    from astropy.table import hstack, join
    from ctapipe.io import read_table

    subarray_string = "subarray" if tel_id == None else "telescope"
    tel_id_string = "" if tel_id == None else f"tel_{tel_id:03d}"
    pointing = read_table(
        input_file, f"dl1/monitoring/{subarray_string}/pointing/{tel_id_string}"
    )
    key_tel = "" if tel_id == None else "tel_"

    dl2_tables = []

    try:
        dl2_classification = read_table(
            input_file,
            f"dl2/event/{subarray_string}/classification/CTLearn/{tel_id_string}",
        )
        dl2_classification = hstack([dl2_classification, pointing])
        dl2_classification = dl2_classification[
            ~np.isnan(dl2_classification[f"CTLearn_{key_tel}prediction"])
        ]
        dl2_tables.append(dl2_classification)
    except:
        print(
            f"Classification table not found for dl2/event/{subarray_string}/classification/CTLearn/{tel_id_string}"
        )

    try:
        dl2_energy = read_table(
            input_file, f"dl2/event/{subarray_string}/energy/CTLearn/{tel_id_string}"
        )
        if len(dl2_tables) == 0:
            dl2_energy = hstack([dl2_energy, pointing])
        dl2_energy = dl2_energy[~np.isnan(dl2_energy[f"CTLearn_{key_tel}energy"])]
        dl2_tables.append(dl2_energy)
    except:
        print(
            f"Energy table not found for dl2/event/{subarray_string}/energy/CTLearn/{tel_id_string}"
        )

    try:
        dl2_geometry = read_table(
            input_file, f"dl2/event/{subarray_string}/geometry/CTLearn/{tel_id_string}"
        )
        if len(dl2_tables) == 0:
            dl2_geometry = hstack([dl2_geometry, pointing])
        dl2_geometry = dl2_geometry[~np.isnan(dl2_geometry[f"CTLearn_{key_tel}alt"])]
        dl2_tables.append(dl2_geometry)
    except:
        print(
            f"Geometry table not found for dl2/event/{subarray_string}/geometry/CTLearn/{tel_id_string}"
        )
    # dl2 = join(dl2_classification, dl2_energy, keys=["obs_id", "event_id"])
    # dl2 = join(dl2, dl2_geometry, keys=["obs_id", "event_id"])
    if len(dl2_tables) > 0:
        dl2 = dl2_tables[0]
        for table in dl2_tables[1:]:
            dl2 = join(dl2, table, keys=["obs_id", "event_id"])
    else:
        raise ValueError("No DL2 tables found")

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


def load_DL2_data_chunked(input_file, DL2DataProcessor, chunk_size=10000):
    """Load DL2 data in chunks to avoid memory issues."""
    tel_id = DL2DataProcessor.telescope_id
    reco_method = DL2DataProcessor.reconstruction_method
    path_dl2 = "subarray"
    path_dl1 = "telescope"
    tel = f"tel_{tel_id:03d}"
    
    from astropy.table import hstack, join, vstack
    from ctapipe.io import read_table
    
    # Load pointing data (usually smaller) - we'll join this later more efficiently
    pointing = read_table(input_file, f"dl1/monitoring/{path_dl1}/pointing/tel_{tel_id:03d}")
    pointing.sort("time")
    
    # Get table info to determine total rows
    def get_table_length(table_path):
        try:
            # Read just one row to get structure and check if table exists
            test_table = read_table(input_file, table_path, start=0, stop=1)
            # Get full table info
            import h5py
            with h5py.File(input_file, 'r') as f:
                return len(f[table_path])
        except:
            return 0
    
    # Check which tables exist and get their lengths
    tables_info = {}
    table_paths = {
        'classification': f"dl2/event/{path_dl2}/classification/{reco_method}/",
        'energy': f"dl2/event/{path_dl2}/energy/{reco_method}/", 
        'geometry': f"dl2/event/{path_dl2}/geometry/{reco_method}/",
        'dl1': f"dl1/event/{path_dl1}/parameters/{tel}"
    }
    
    for table_name, path in table_paths.items():
        length = get_table_length(path)
        if length > 0:
            tables_info[table_name] = {'path': path, 'length': length}
            print(f"Found {table_name} table with {length} rows")
    
    if not tables_info:
        raise ValueError("No DL2 tables found")
    
    # Use the maximum length among available tables
    max_length = max(info['length'] for info in tables_info.values())
    num_chunks = (max_length + chunk_size - 1) // chunk_size
    print(f"Processing {max_length} events in {num_chunks} chunks of {chunk_size}")
    
    dl2_chunks = []
    
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        stop = min((chunk_idx + 1) * chunk_size, max_length)
        print(f"Processing chunk {chunk_idx + 1}/{num_chunks} (rows {start}-{stop})")
        
        chunk_tables = []
        
        # Load each available table chunk
        for table_name, info in tables_info.items():
            if stop > info['length']:
                continue
                
            try:
                if table_name == 'dl1':
                    chunk_table = read_table(input_file, info['path'], start=start, stop=stop)[
                        ["obs_id", "event_id", "hillas_intensity"]
                    ]
                else:
                    chunk_table = read_table(input_file, info['path'], start=start, stop=stop)
                
                if len(chunk_tables) == 0:
                    chunk_tables.append(chunk_table)
                else:
                    # Join with previous tables
                    chunk_tables[0] = join(chunk_tables[0], chunk_table, keys=["obs_id", "event_id"])
            except Exception as e:
                print(f"Error loading {table_name} chunk: {e}")
                continue
        
        if chunk_tables:
            chunk_dl2 = chunk_tables[0]
            dl2_chunks.append(chunk_dl2)
    
    if not dl2_chunks:
        raise ValueError("No data chunks could be processed")
    
    # Combine all chunks first
    print("Combining chunks...")
    dl2 = vstack(dl2_chunks)
    
    # Now sort and add pointing data more efficiently
    dl2.sort("event_id")
    
    # Add pointing data by joining on obs_id or using a more memory-efficient method
    # Instead of hstack, we'll add pointing columns manually to avoid broadcasting issues
    print("Adding pointing data...")
    
    # Create arrays for pointing data that match dl2 length
    n_events = len(dl2)
    pointing_cols = {}
    
    # For each pointing column, create an array of the right size
    for col_name in pointing.colnames:
        if col_name not in ['obs_id']:  # Don't duplicate obs_id
            # Initialize with NaN or appropriate default
            if hasattr(pointing[col_name], 'unit'):
                pointing_cols[col_name] = np.full(n_events, np.nan) * pointing[col_name].unit
            else:
                pointing_cols[col_name] = np.full(n_events, np.nan, dtype=pointing[col_name].dtype)
    
    # Fill in pointing data for matching obs_ids
    unique_obs_ids = np.unique(dl2['obs_id'])
    for obs_id in unique_obs_ids:
        # Find matching rows in pointing data
        pointing_mask = pointing['obs_id'] == obs_id
        if np.any(pointing_mask):
            # Get the first matching pointing entry (or implement more sophisticated matching)
            pointing_row = pointing[pointing_mask][0]
            # Find matching rows in dl2
            dl2_mask = dl2['obs_id'] == obs_id
            # Fill in the pointing data
            for col_name in pointing_cols.keys():
                pointing_cols[col_name][dl2_mask] = pointing_row[col_name]
    
    # Add the pointing columns to dl2
    for col_name, col_data in pointing_cols.items():
        dl2[col_name] = col_data
    
    dl2.sort("time")

    print("Computing time differences...")
    t_diff = compute_diff(dl2["time"].to_value("unix"))
    dl2["delta_t"] = t_diff
    
    print(f"Loaded {len(dl2)} events total")
    return dl2


def load_DL2_data(input_file, DL2DataProcessor, use_chunking=True, chunk_size=10000):
    """Load DL2 data with optional chunking for memory efficiency."""
    if use_chunking:
        return load_DL2_data_chunked(input_file, DL2DataProcessor, chunk_size)
    
    # Original implementation for backward compatibility
    tel_id = DL2DataProcessor.telescope_id
    reco_method = DL2DataProcessor.reconstruction_method
    path_dl2 = "subarray"
    path_dl1 = "telescope"
    tel = f"tel_{tel_id:03d}"
    
    from astropy.table import hstack, join
    from ctapipe.io import read_table

    # pointing = read_table(input_file, f"dl1/monitoring/{path}/pointing/{tel}")
    pointing = read_table(input_file, f"dl1/monitoring/{path_dl1}/pointing/tel_{tel_id:03d}")
    # pointing = read_table_hdf5(input_file, path=f"dl1/monitoring/{path}/pointing/{tel}")
    pointing.sort("time")

    dl2 = None

    try:
        print(f"dl2/event/{path_dl2}/classification/{reco_method}/")
        dl2_classification = read_table(
            input_file, f"dl2/event/{path_dl2}/classification/{reco_method}/"
        )
        dl2 = dl2_classification
    except:
        print(f"Classification table not found for {reco_method}/")

    try:
        print(f"dl2/event/{path_dl2}/energy/{reco_method}/")
        dl2_energy = read_table(
            input_file, f"dl2/event/{path_dl2}/energy/{reco_method}/"
        )
        dl2 = (
            join(dl2, dl2_energy, keys=["obs_id", "event_id"])
            if dl2 is not None
            else dl2_energy
        )
    except:
        print(f"Energy table not found for {reco_method}/")

    try:
        print(f"dl2/event/{path_dl2}/energy/{reco_method}/")
        dl2_geometry = read_table(
            input_file, f"dl2/event/{path_dl2}/geometry/{reco_method}/"
        )
        dl2 = (
            join(dl2, dl2_geometry, keys=["obs_id", "event_id"])
            if dl2 is not None
            else dl2_geometry
        )
    except:
        print(f"Geometry table not found for {reco_method}/")

    dl1 = read_table(input_file, f"dl1/event/{path_dl1}/parameters/{tel}")[
        ["obs_id", "event_id", "hillas_intensity"]
    ]
    dl2 = join(dl2, dl1, keys=["obs_id", "event_id"]) if dl2 is not None else dl1

    dl2.sort("event_id")
    dl2 = hstack([dl2, pointing])
    dl2.sort("time")

    print("Computing time differences...")
    t_diff = compute_diff(dl2["time"].to_value("unix"))
    dl2["delta_t"] = t_diff

    print(f"Loaded {len(dl2)} events")
    return dl2


# def load_DL2_data(input_file, DL2DataProcessor):
#     tel_id = DL2DataProcessor.telescope_id
#     reco_method = DL2DataProcessor.reconstruction_method
#     path = "subarray" if DL2DataProcessor.stereo else "telescope"
#     tel = f"tel_{tel_id:03d}" if DL2DataProcessor.stereo else f"tel_{tel_id:03d}"
#     from ctapipe.io import read_table
#     from astropy.table import (join, hstack)
#     pointing = read_table(input_file, f"dl1/monitoring/{path}/pointing/{tel}")
#     pointing.sort('time')
#     dl2_classification = read_table(input_file, f"dl2/event/{path}/classification/{reco_method}/{tel}")
#     dl2_energy = read_table(input_file, f"dl2/event/{path}/energy/{reco_method}/{tel}")
#     dl2_geometry = read_table(input_file, f"dl2/event/{path}/geometry/{reco_method}/{tel}")
#     dl1 = read_table(input_file, f"dl1/event/{path}/parameters/{tel}")[["obs_id", "event_id", "hillas_intensity"]]
#     dl2 = join(dl2_classification, dl2_energy, keys=["obs_id", "event_id"])
#     dl2 = join(dl2, dl2_geometry, keys=["obs_id", "event_id"])
#     dl2 = join(dl2, dl1, keys=["obs_id", "event_id"])
#     dl2.sort('event_id')
#     dl2 = hstack([dl2, pointing])
#     dl2.sort('time')
#     # times = np.array(dl2['time'])
#     print("Computing time differences...")
#     # t_diff = np.diff(dl2['time'])#.to_value('unix')
#     # t_diff = np.insert(t_diff, 0, TimeDelta(0*u.s, format='jd', scale='tai'))  # Insert 0 at the beginning to align with the original times array
#     t_diff = compute_diff(dl2['time'].to_value('unix'))
#     dl2['delta_t'] = t_diff
#     print(f"Loaded {len(dl2)} events")
#     return dl2


def load_DL2_data_RF(input_file, DL2DataProcessor):
    # tel_id = DL2DataProcessor.telescope_id
    # reco_method = DL2DataProcessor.reconstruction_method
    path = "subarray" if DL2DataProcessor.stereo else "telescope"
    # tel = f"tel_{tel_id:03d}" if DL2DataProcessor.stereo else f"tel_{tel_id:03d}"
    from ctapipe.io import read_table

    # from astropy.table import (join, hstack)
    # pointing = read_table(input_file, f"dl1/monitoring/{path}/pointing/{tel}")
    # pointing.sort('time')
    dl2 = read_table(input_file, f"dl2/event/{path}/parameters/LST_LSTCam")
    # dl1 = read_table(input_file, f"dl1/event/{path}/parameters/{tel}")[["obs_id", "event_id", "hillas_intensity"]]
    # dl2 = join(dl2_classification, dl2_energy, keys=["obs_id", "event_id"])
    # dl2 = join(dl2, dl2_geometry, keys=["obs_id", "event_id"])
    # dl2 = join(dl2, dl1, keys=["obs_id", "event_id"])
    # dl2.sort('event_id')
    # dl2 = hstack([dl2, pointing])
    # dl2.sort('time')
    # # times = np.array(dl2['time'])
    # print("Computing time differences...")
    # # t_diff = np.diff(dl2['time'])#.to_value('unix')
    # # t_diff = np.insert(t_diff, 0, TimeDelta(0*u.s, format='jd', scale='tai'))  # Insert 0 at the beginning to align with the original times array
    # t_diff = compute_diff(dl2['time'].to_value('unix'))
    # dl2['delta_t'] = t_diff
    # print(f"Loaded {len(dl2)} events")
    # print(dl2.columns)

    useful_cols = [
        "obs_id",
        "event_id",
        "intensity",
        "alt_tel",
        "az_tel",
        "dragon_time",
        "delta_t",
        "reco_energy",
        "reco_alt",
        "reco_az",
        "gammaness",
    ]

    dl2 = dl2[dl2["event_type"] == 32]
    dl2 = dl2[useful_cols]
    dl2.rename_column("intensity", "hillas_intensity")
    dl2.rename_column("alt_tel", "altitude")
    dl2.rename_column("az_tel", "azimuth")
    dl2.rename_column("dragon_time", "time")
    dl2.rename_column("reco_energy", "RF_tel_energy")
    dl2.rename_column("reco_alt", "RF_tel_alt")
    dl2.rename_column("reco_az", "RF_tel_az")
    dl2.rename_column("gammaness", "RF_tel_prediction")

    dl2["altitude"] = dl2["altitude"] * u.rad
    dl2["azimuth"] = dl2["azimuth"] * u.rad
    dl2["RF_tel_energy"] = dl2["RF_tel_energy"] * u.TeV
    dl2["RF_tel_alt"] = dl2["RF_tel_alt"] * u.rad
    dl2["RF_tel_az"] = dl2["RF_tel_az"] * u.rad
    dl2["delta_t"] = dl2["delta_t"] * u.s
    dl2 = dl2[dl2["hillas_intensity"] > 50]
    return dl2
