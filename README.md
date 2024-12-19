# CTLearn Manager

A compagnon package to CTLearn that enables to manage models to train, test, monitor and benchmark them.

## Usage

The whole CTLearn Managers revolvesaround the ctlearn_model_index.ecsv, that containsall your models with the relevantinformation for training, predicting, and benchmarking, allowing you to easily jump between models and compare them.

### 🧠 Setup a model manager

To setup `CTLeanrModelManager`, you need to specify the following fields:
- `model_nickname` : "a_name_for_your_model"
- `model_dir` : "where/to/stor/the/models/" ➡️ Note that a subdirectory will be created with the name andversion of your model.
- `notes` : "Stereo model for 20deg zenith distance"
- `reco` : "type" ➡️ Choose among `energy`, `direction` and `type`
- `channels` : ["cleaned_image" "cleaned_relative_peak_time"] # Order matters
- `telescope_names` : ["SST1M_1" "SST1M_2"]
- `telescopes_indices` : [1, 2]
- `training_gamma_dirs` : ["/DL1/SST1M/MC/Gamma_diffuse/training/"]
- `training_proton_dirs` : ["/DL1/SST1M/MC/Proton_diffuse/training/"]
- `training_gamma_zenith_distances` : [20] ➡️ In deg
- `training_gamma_azimuths` : [0]
- `training_proton_zenith_distances` : [20]
- `training_proton_azimuths` : [0]
- `max_training_epochs` : 15 ➡️ Can be changed later, avoids launching training unwantedly.
- `MODEL_INDEX_FILE` = "path/to/index/ctearn_models_index.ecsv"

Call `CTLeanrModelManager.save_to_index()` to save the model to the index file.

### 🚀 Launch training

Simply call `CTLeanrModelManager.launch_training(n_epochs=15)`.
Thenumberof epochs can be increased from the `max_training_epochs`, in which case the model will train for more epochs.
In case of resuming training, a newer version of the model will be created andtrainingwill continue for the remaining amount of epochs.

### 📉 Training monitoring

You can plot the training and validation losses by doing:

```
from ctlearn_manager import load_model_from_index
MODEL_INDEX_FILE = "path/to/index/ctearn_models_index.ecsv"
model = load_model_from_index("model_nickname", MODEL_INDEX_FILE)
model.plot_loss()
```