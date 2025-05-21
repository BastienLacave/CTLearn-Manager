4. Testing
==========

Model testing is a crucial part of the development process. It is important to
ensure that the model is working as expected and that it is generalizing well
to unseen data.

In order to be able to launch the testing of your model, you need 3 trained models, one for each task (energy, direction and type). Creating a TriModelManager will allow the user to manager the 3 tasks at once.

As usual, ceate a ``TriModelManager`` object with the three models you want to test.

.. code-block:: python

    MODEL_INDEX_FILE = "/path/to/your/ctlearn_models_index.h5"
    energy_model = load_model_from_index("energy_model_nickname", MODEL_INDEX_FILE)
    direction_model = load_model_from_index("direction_model_nickname", MODEL_INDEX_FILE)
    type_model = load_model_from_index("type_model_nickname", MODEL_INDEX_FILE)
    Tri_Model = CTLearnTriModelManager(direction_model=direction_model, energy_model=energy_model, type_model=type_model)


There are two cases if you want to test your model. Either the user doesn't have the testing DL2 files yet and needs to produce them, or the files exist already.

Testing from scratch
--------------------

The first step is to set the testing directories for gamma and protons, and their respective coordinates. This step is required only once as the DL1 files will be saved in the manager.

.. code-block:: python

    # Add electrons, gamma_diffuse and whatever you like to test your model on, each in a separate DataSample
    testing_samples = [
        DataSample(directory='/path/to/your/MC/Gamma_point/20deg/merged/testing/',
                pattern="gamma_point*.h5"),
        DataSample(directory='/path/to/your/MC/Proton_diffuse/20deg/merged/testing/',
                pattern="proton*.h5"),
        ]

    Tri_Model.set_testing_data(testing_samples)


Then, the user can launch the training for any of the coordinates of the testing files. One can access the available coordinates by calling ``get_available_testing_directions()``. 
A few additional settings enable you to choose the output directories, what type of particle to launch, and the cluster configuration. The DL2 MC files will also be stored in the manager with their coordinates.

.. info::
    ``Tri_Model.get_available_testing_directions()`` is a very usefule command from which you can directly copy and paste the direction you want to test on.

    For DL2 data, the ``get_available_MC_directions()`` method is also available. It will return the coordinates of the DL2 files produced by the testing.

.. code-block:: python

    # See available testing directions and check the info of the cluster configuration
    Tri_Model.plot_zenith_azimuth_ranges()
    Tri_Model.get_available_testing_directions()
    Tri_Model.cluster_configuration.info()

Launch the testing:
.. code-block:: python

   # copy the direction in zenith and azimuth you want to test on:
    Tri_Model.launch_testing(32.059 * u.deg, 248.099 * u.deg, 
        output_dirs = ["/path/to/your/Testing_models/gamma_point/"], 
        launch_particle_types=[ParticleType.GAMMA_POINT], # Add as much as you want, must provide a directory for each
        config_dir="/path/to/your/configs/", # Where the .sh scripts, configs and logs will be stored
        batch_size=64,
        # Other options are available
        )


Existing MC DL2 files (not recommended)
---------------------

The user should gather the files and their respective coordinates, using ``set_DL2_MC_file()``, 
inform the manager of the available DL2 files for later plotting and IRF production. 
For this step, the user needs a TriModelManager, hence should create the models with accurate information about the telescope and the training coordinates etc.
The ``DataSample`` is needed to insure the particle type and direction, make sure is corresponds to the testing DL2 file you want to register to the Manager.

.. code-block:: python

    Tri_Model.set_DL2_MC_file(testing_MC_DL2_file, testing_MC_DL2_data_sample)

From there the user will be able to produce IRFs and DL2 analysis plots.









