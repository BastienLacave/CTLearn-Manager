from .. import CTLearnModelManager
from astropy.table import QTable
import numpy as np
from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5

def load_model_from_index(model_nickname, MODEL_INDEX_FILE):
    # models_table = QTable.read(MODEL_INDEX_FILE)
    # model_index = np.where(models_table['model_nickname'] == model_nickname)[0][0]
    model_parameters = {'model_nickname': model_nickname}
    model = CTLearnModelManager(model_parameters, MODEL_INDEX_FILE, load=True)
    return model