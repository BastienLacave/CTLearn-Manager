# CTLearn Manager

A compagnon package to CTLearn that enables to manage models to train, test, monitor and benchmark them.

## Usage

The whole CTLearn Managers revolvesaround the ctlearn_model_index.ecsv, that containsall your models with the relevantinformation for training, predicting, and benchmarking, allowing you to easily jump between models and compare them.

### 🧠 Setup a model manager

To setup `CTLeanrModelManager`, you need to specify the following fields:
- `model_nickname` : "a_name_for_your_model"
- `model_dir` : "where/to/stor/the/models/" ➡️ Note that a subdirectory will be created with the name andversion of your model.
- `notes` : "Stereo model for 20deg zenith distance"
- `reco` : `type` ➡️ Choose among `energy`, `direction` and `type`
- `channels` : [`cleaned_image` `cleaned_relative_peak_time`] # Order matters
- `telescope_names` : [`SST1M_1` `SST1M_2`]
- `telescopes_indices` : [1, 2]
- `training_gamma_dirs` : [`/DL1/SST1M/MC/Gamma_diffuse/training/`]
- `training_proton_dirs` : [`/DL1/SST1M/MC/Proton_diffuse/training/`]
- `training_gamma_zenith_distances` : [20] ➡️ In deg
- `training_gamma_azimuths` : [0]
- `training_proton_zenith_distances` : [20]
- `training_proton_azimuths` : [0]
- `max_training_epochs` : 15 ➡️ Can be changed later, avoids launching training unwantedly.