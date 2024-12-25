from .. import CTLearnModelManager
from astropy.table import QTable
import numpy as np

def load_model_from_index(model_nickname, MODEL_INDEX_FILE):
    models_table = QTable.read(MODEL_INDEX_FILE)
    model_index = np.where(models_table['model_nickname'] == model_nickname)[0][0]
    model_parameters = {key: models_table[key][model_index] for key in models_table.colnames}
    model = CTLearnModelManager(model_parameters, MODEL_INDEX_FILE)
    return model