# CTLearn Manager

A compagnon package to CTLearn that enables to manage models to train, test, monitor and benchmark them.

The whole CTLearn Managers revolves around the `ctlearn_model_index.ecsv`, that contains all your models with the relevant information for training, predicting, and benchmarking, allowing you to easily jump between models and compare them.

[Documentation](https://ctlearn-manager.readthedocs.io/en/latest/)

## 🧠 Setup a model manager

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

```
# Where all the models are stored
MODEL_INDEX_FILE = "/home/blacave/CTLearn/Software/CTLearn-Manager/ctearn_models_index.ecsv"
# General parameters
model_parameters = {
    'model_nickname' : "a_name_for_your_model",
    'model_dir' : "where/to/stor/the/models/",
    'notes' : "Stereo model for 20deg zenith distance",
    'reco' : 'type', #["energy", "direction", "type"]
    'channels' : ['cleaned_image', 'cleaned_relative_peak_time'], # Order matters
    'telescope_names' : ['SST1M_1', 'SST1M_2'],
    'telescopes_indices' : [1, 2],
    'training_gamma_dirs' : ["/DL1/SST1M/MC/Gamma_diffuse/training/"],
    'training_proton_dirs' : ["/DL1/SST1M/MC/Proton_diffuse/training/"],
    'training_gamma_zenith_distances' : [20],
    'training_gamma_azimuths' : [0],
    'training_proton_zenith_distances' : [20],
    'training_proton_azimuths' : [0],
    'max_training_epochs' : 15, 
}
new_model = CTLearnModelManager(model_parameters, MODEL_INDEX_FILE)
new_model.save_to_index()
```

## 🚀 Launch training

Simply call `CTLeanrModelManager.launch_training(n_epochs=15)`.
The number of epochs can be increased from the `max_training_epochs`, in which case the model will train for more epochs.
In case of resuming training, a newer version of the model will be created and training will continue for the remaining amount of epochs.

## 📉 Training monitoring

You can plot the training and validation losses by doing:

```
from ctlearn_manager import load_model_from_index
MODEL_INDEX_FILE = "path/to/index/ctearn_models_index.ecsv"
model = load_model_from_index("model_nickname", MODEL_INDEX_FILE)
model.plot_loss()
```
`CTLeanrModelManager.plot_sky()`

## 🧪 Model testing and MC predictions

Model testing requires 3 models, one for each reconstruction task `energy`, `direction` and `type`.
CTLearn Manager implements the `CTLeanrTriModelManager` that isbuilt by combining the 3 models you want for each task.
Along with these models, you provide the testing files for gammas and protons.

```
from ctlearn_manager import load_model_from_index, CTLearnTriModelManager
MODEL_INDEX_FILE = "path/to/index/ctearn_models_index.ecsv"
energy_model = load_model_from_index("energy_stereo_20deg", MODEL_INDEX_FILE)
direction_model = load_model_from_index("direction_stereo_20deg", MODEL_INDEX_FILE)
type_model = load_model_from_index("type_stereo_20deg", MODEL_INDEX_FILE)
Stereo_Tri_Model = CTLearnTriModelManager(direction_model=direction_model, energy_model=energy_model, type_model=type_model)
```

### Launch testing
`CTLeanrTriModelManager.set_testing_directories()`

Set testing directories for gammas and protons, as well as there respective alt az coordinates, you can set as many directories as you want.

`CTLeanrTriModelManager.launch_testing()`

This launches the testing on all testing directories and stores the ouput file paths to the index for later performance evaluation.

You can then plot stuff:

`Stereo_Tri_Model.plot_migration_matrix()`

`Stereo_Tri_Model.plot_DL2_AltAz()`

`Stereo_Tri_Model.plot_DL2_classification()`

`Stereo_Tri_Model.plot_DL2_energy()`

## 🔭 Produce IRFs

TBD

## 🧾 Predict real data

You can predict any data using `CTLearnTriModelManager.predict_data()` or `CTLearnTriModelManager.predict_lstchain_data()` depending on the type you want to process.

### 📡 TriModelCollection
`TriModelCollection` is a collection of `CTLearnTriModelManager`, that could for example be a set of models along a certain declination line.

`TriModelCollection.predict_data()` and `TriModelCollection.predict_lstchain_data()` will choose the closest model to the pointing of the provided file among the collection you provided.

`TriModelCollection.preview()`

## ↔️ CTLearn model comparison

Compare as many CTLearn models as you want with perf plots, DL2 level analysis and more.

## 🔥 Compare CTLearn and Random Forest