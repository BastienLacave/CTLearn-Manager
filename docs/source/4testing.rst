4. Testing
==========

Model testing is a crucial part of the development process. It is important to
ensure that the model is working as expected and that it is generalizing well
to unseen data.

In order to be able to launch the testing of your model, you need 3 trained models, one for each task (energy, direction and type). Creating a TriModelManager will allow the user to manager the 3 tasks at once.

As usual, ceate a ``TriModelManager`` object with the three models you want to test.

.. code-block:: python

    MODEL_INDEX_FILE = "/home/blacave/CTLearn/Software/CTLearn-Manager/ctearn_models_index.h5"
    energy_model = load_model_from_index("energy_stereo_20deg", MODEL_INDEX_FILE)
    direction_model = load_model_from_index("direction_stereo_20deg", MODEL_INDEX_FILE)
    type_model = load_model_from_index("type_stereo_20deg", MODEL_INDEX_FILE)
    Stereo_Tri_Model = CTLearnTriModelManager(direction_model=direction_model, energy_model=energy_model, type_model=type_model)


There are two cases if you want to test your model. Either the user doesn't have the testing DL2 files yet and needs to produce them, or the files exist already.

Testing from scratch
--------------------

The first step is to set the testing directories for gamma and protons, and their respective coordinates. This step is required only once as the DL1 files will be saved in the manager.

.. code-block:: python

    Stereo_Tri_Model.set_testing_directories(
        testing_gamma_dirs = ["/home/user/CTLearn/Data/DL1/SST1M/MC/Gamma_point/20deg/testing/"], 
        testing_proton_dirs = ["/home/user/CTLearn/Data/DL1/SST1M/MC/Proton_diffuse/20deg/merged/testing/"], 
        testing_gamma_zenith_distances = [20], 
        testing_gamma_azimuths = [0], 
        testing_proton_zenith_distances = [20], 
        testing_proton_azimuths = [0],
        testing_gamma_patterns=["*.h5"],
        testing_proton_patterns=["*.h5"],
        )


Then, the user can launch the training for any of the coordinates of the testing files. One can access the available coordinates by calling ``get_available_testing_directories()``. 
A few additional settings enable you to choose the output directories for gammas and proton DL2 files, what type of particle to launch, and the cluster configuration. The DL2 MC files will also be stored in the manager with their coordinates.

.. code-block:: python

    Stereo_Tri_Model.launch_testing(20, 0, 
        ["/home/blacave/CTLearn/Data/DL2/Testing/"], 
        launch_particle_type='both', 
        )


Existing MC DL2 files
---------------------

The user should gather the files and their respective coordinates, using ``set_DL2_MC_files()``, 
inform the manager of the available DL2 files for later plotting and IRF production. 
For this step, the user needs a TriModelManager, hence should create the models with accurate information about the telescope and the training coordinates etc.


.. code-block:: python

    Stereo_Tri_Model.set_DL2_MC_files(
                    ["gamma_file_1.dl2.h5", "gamma_file_2.dl2.h5",], 
                    ["proton_file_1.dl2.h5", "proton_file_2.dl2.h5",], 
                    [30.390, 37.814], # Zenith distances gamma
                    [266.360, 270],  # Azimuths gamma
                    [xxx,xxx],  # Zenith distances proton
                    [xxx,xxx] # Azimuths proton
                    )

From there the user will be able to produce IRFs and DL2 analysis plots.









