5. IRFs
=======

In this section, we will show how to produce IRFs from your DL2 data.

Produce IRFs
------------


As always, load your models in a ``TriModelManager`` object.

.. code-block:: python

    MODEL_INDEX_FILE = "/path/to/your/ctlearn_models_index.h5"
    energy_model = load_model_from_index("energy_model_nickname", MODEL_INDEX_FILE)
    direction_model = load_model_from_index("direction_model", MODEL_INDEX_FILE)
    type_model = load_model_from_index("type_model", MODEL_INDEX_FILE)
    Tri_Model = CTLearnTriModelManager(direction_model=direction_model, energy_model=energy_model, type_model=type_model)

.. important::

    CTLearn Manager uses ctapipe to produce IRFs, and anly takes one file per particle type, so if you have multiple, you should merged them with the Manager.

.. code-block:: python

    Tri_Model.merge_DL2_files(10.0 * u.deg, 102.199 * u.deg, 
                                output_file = "/path/to/your/merged.h5", 
                                particle_type = ParticleType.GAMMA_POINT, 
                                overwrite=True)


Then, you need to set a configuration file for the IRFs tool, that includes options for the cuts optimizer and the IRF maker.
The configuration file is a yaml file that can be found in the ``ressources`` folder of the repository
The main parameters are the following:

    optimization_algorithm: "PercentileCuts" or "PointSourceSensitivityOptimizer" \\
    target_percentile: 70 # for gammaness efficiency and/or theta efficiency.

Also make sure that the energy bins are adequate to the telescope.

Then, create the output paths, make sure they are unique for each type of IRF if you want to produce pultiple (e.g. for different efficiencies) but also for the different directions.

.. code-block:: python

    zenith, azimuth = 10 * u.deg, 180 * u.deg

    # ⚠️⚠️⚠️ Make sure output files are unique !!!
    config = "/path/to/your/public-conf.yml" # Example config found in ressources, do not modify or move it after producing IRFs
    output_cuts_file = f"/path/to/your/IRFs/cuts_{zenith.value:.2f}_{azimuth.value:.2f}.fits" 
    output_irf_file = f"/path/to/your/IRFs/IRFs_{zenith.value:.2f}_{azimuth.value:.2f}.fits"
    output_benchmark_file = f"/path/to/your/IRFs/benchmark_{zenith.value:.2f}_{azimuth.value:.2f}.fits"

Then launch the IRF production, that will run locally.

.. code-block:: python

    Tri_Model.produce_irfs(zenith, azimuth, 
                              config=config, 
                              output_cuts_file=output_cuts_file, 
                              output_irf_file=output_irf_file,
                              output_benchmark_file=output_benchmark_file,
                              pointlike=True,
                              electrons=False,
                              protons=True
                              )


Visualization
-------------

One can look into the cuts, benchmark and IRF files produced.

.. code-block:: python

    Tri_Model.plot_cuts(zenith, azimuth)
    Tri_Model.plot_benchmark(zenith, azimuth)
    Tri_Model.plot_irfs(zenith, azimuth)
