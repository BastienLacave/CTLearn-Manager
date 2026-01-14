"""Plotting and predicting with a collection of CTLearnTriModelManager models."""

import os

import astropy.units as u
import ctadata
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from . import CTLearnTriModelManager

from .utils.utils import (
    ClusterConfiguration,
    Cuts,
    CutType,
    DefaultCuts,
    ParticleType,
    angular_distance,
    get_files_cscs,
    get_files_LST_cluster,
    set_mpl_style,
    CTLMDirectories,
    get_color
)

__all__ = ["TriModelCollection"]


class TriModelCollection:
    """
    TriModelCollection is a class that manages a collection of tri-models.
    
    Use for predicting and analyzing data from telescopes. It provides methods for 
    predicting data, finding the closest model to a given pointing, and 
    visualizing various performance metrics.

    Attributes
    ----------
    tri_models : list
        A list of tri-models to be managed by the collection.
    cluster_configuration : ClusterConfiguration
        Configuration for the cluster where the models will run.
    model_labels : list[str]
        Labels for the tri-models in the collection.

    Methods
    -------
    __init__(tri_models, cluster_configuration, model_labels)
        Initialize the TriModelCollection with a list of tri-models, cluster configuration, and optional model labels.
    predict_lstchain_run(run, output_dir, DL1_data_dir=None, overwrite=False, plot=False, batch_size=64)
        Predict data for a specific LST chain run and save the results.
    predict_lstchain_data(input_file, output_file, pointing_table, config_dir=None, overwrite=False, run=None, subrun=None, plot=False, batch_size=64)
        Predict data for a specific input file using the closest tri-model.
    predict_data(input_file, output_file, pointing_table, config_dir=None, overwrite=False, plot=False)
        Predict data for a given input file and save the results.
    find_closest_model_to(input_file, pointing_table, plot=False, alt_key="alt_tel", az_key="az_tel", verbose=True)
        Find the closest tri-model to a given pointing based on angular distance.
    plot_zenith_azimuth_ranges(plot_testing_nodes=True)
        Plot the zenith and azimuth ranges for all tri-models in the collection.
    plot_energy_resolution_DL2(cuts, zenith=None, azimuth=None, ylim=None, particle_type, figsize=None, plot_RF=False, compare_with=None)
        Plot the energy resolution for DL2 data with optional comparison to a reference model.
    plot_angular_resolution_DL2(cuts, zenith=None, azimuth=None, ylim=None, particle_type, figsize=None, plot_RF=False, compare_with=None)
        Plot the angular resolution for DL2 data with optional comparison to a reference model.
    plot_cuts(cuts)
        Plot the cuts applied to the data for all tri-models in the collection.
    plot_everything_dl2(output_directory, dl2_files, gammaness_cut=0.9, edep_cuts=False, pointing_table="/dl1/monitoring/telescope/pointing/tel_001")
        Generate and save plots for angular resolution, energy resolution, and gammaness for DL2 data.
    """

    def __init__(

        self,
        tri_models: list[CTLearnTriModelManager],
        cluster_configuration: ClusterConfiguration = ClusterConfiguration(),
        model_labels: list[str] = None,
        allow_muliple_projects=False,
    ):
        """
        Initialize the TriModelCollection object.

        Parameters
        ----------
        tri_models : list
            A list of tri-model objects to be included in the collection.
        cluster_configuration : ClusterConfiguration, optional
            The cluster configuration to be applied to all tri-models in the collection.
            Defaults to a new instance of `ClusterConfiguration`.
        model_labels : list of str, optional
            A list of labels for the tri-models. If not provided, default labels
            in the format "Model_{j}" will be generated.

        Attributes
        ----------
        tri_models : list
            The list of tri-models in the collection.
        cluster_configuration : ClusterConfiguration
            The cluster configuration applied to all tri-models.
        model_labels : list of str
            The labels for the tri-models in the collection.

        Raises
        ------
        AssertionError
            If `model_labels` is provided and its length does not match the number
            of tri-models.
            If the stereos of all tri-models are not the same.
        """
        self.tri_models: list[CTLearnTriModelManager] = tri_models
        self.cluster_configuration = cluster_configuration

        # Assert that all tri_models have the same project_directory
        self.allow_muliple_projects = allow_muliple_projects
        if not self.allow_muliple_projects:
            project_directories = [tri_model.project_directories.project_directory for tri_model in self.tri_models]
            assert len(set(project_directories)) == 1, "All tri_models must be part of the same project_directory."
            project_directories = [tri_model.project_directories for tri_model in self.tri_models]
            self.project_directories: CTLMDirectories = project_directories[0]
        for tri_model in self.tri_models:
            tri_model.cluster_configuration = cluster_configuration
        telescope_ids = [tri_model.telescope_ids for tri_model in self.tri_models]
        telescope_names = [tri_model.telescope_names for tri_model in self.tri_models]
        stereos = [tri_model.stereo for tri_model in self.tri_models]
        if model_labels is not None:
            assert len(model_labels) == len(self.tri_models), (
                "Model labels must be the same length as the number of tri models."
            )
            self.model_labels = model_labels
        else:
            self.model_labels = [f"Model_{j}" for j in range(len(self.tri_models))]
        assert len(set(stereos)) == 1, "All stereos in the collection must be the same."
        # set_mpl_style()
        self.tri_model_nicknames = [tri_model.project_directories.tri_model_nickname for tri_model in self.tri_models]
        # assert len(set(telescope_ids)) == 1, "All telescope_ids in the collection must be the same."
        # assert len(set(telescope_names)) == 1, "All telescope_names in the collection must be the same."

    def predict_lstchain_run(
        self,
        run: int,
        output_dir: str,
        DL1_data_dir=None,
        overwrite=False,
        plot=False,
        batch_size=64,
        subruns=None,
    ):
        """
        Predict DL2 data for a given LST run using the specified cluster configuration.

        Parameters
        ----------
        run : int
            The run number to process.
        output_dir : str
            Directory where the output DL2 files will be saved.
        DL1_data_dir : str, optional
            Directory containing the DL1 input data. If None, a default path is used
            based on the cluster configuration.
        overwrite : bool, default=False
            Whether to overwrite existing DL2 files.
        plot : bool, default=False
            Whether to generate and save plots during the prediction process.
        batch_size : int, default=64
            Batch size to use during prediction.

        Raises
        ------
        ValueError
            If the cluster configuration is not recognized (neither 'cscs' nor 'lst-cluster').

        Notes
        -----
        - For the 'cscs' cluster, DL1 files are fetched from dCache and copied to a scratch directory
          before prediction.
        - For the 'lst-cluster', DL1 files are directly accessed from the specified directory.
        - The method generates DL2 files for each subrun of the specified run.
        """
        os.makedirs(output_dir, exist_ok=True)
        if self.cluster_configuration.cluster == "cscs":
            if DL1_data_dir is None:
                DL1_data_dir = "/pnfs/cta.cscs.ch/lst/DL1/"
            input_files, v = get_files_cscs(run, DL1_data_dir)
            scratch_dir = os.getenv("SCRATCH")
            scratch_dl1_dir = f"{scratch_dir}/ctlearn_manager_dl1_from_dcache/{run:05d}/{v}/tailcut84/"
            os.system(f"mkdir -p {scratch_dl1_dir}")
            current_directory = os.getcwd()
            print(f"DL1 files will be copied to {scratch_dl1_dir}")
            if subruns is not None:
                input_files = [f for f in input_files if int(f.split('.')[-2]) in subruns]
            for dcache_file in input_files:
                input_file = f"{scratch_dl1_dir}/{dcache_file.split('/')[-1]}"
                subrun = int(input_file.split(".")[-2])
                output_file = f"{output_dir}/LST-1.Run{run:05d}.{subrun:04d}.dl2.h5"
                if os.path.exists(output_file) and not overwrite:
                    print(
                        f"⚠️ Output file already exists and overwrite is set to False : {output_file}"
                    )
                    continue
                if not os.path.exists(input_file) or overwrite:
                    print(f"⌛ Copying {dcache_file} to {scratch_dl1_dir}")
                    ctadata.fetch_and_save_file_or_dir(dcache_file)
                    result = os.system(
                        f"mv {current_directory}/{dcache_file.split('/')[-1]} {scratch_dl1_dir}/{dcache_file.split('/')[-1]}"
                    )
                    if result != 0:
                        print(result)
                        raise RuntimeError()
                # else:
                #     print(f"✅ \nInput file exists: {os.path.exists(input_file)}\n Output file exists: {os.path.exists(output_file)}")


                print(f"Predicting {input_file}")
                self.predict_lstchain_data(
                    input_file,
                    output_file,
                    config_dir=output_dir,
                    overwrite=overwrite,
                    run=run,
                    subrun=subrun,
                    plot=plot,
                    batch_size=batch_size,
                )

        elif self.cluster_configuration.cluster == "lst-cluster":
            if DL1_data_dir is None:
                DL1_data_dir = "/fefs/aswg/data/real/DL1/"
            input_files = get_files_LST_cluster(run, DL1_data_dir)
            for input_file in input_files:
                print(f"Predicting {input_file}")
                subrun = int(input_file.split(".")[-2])
                output_file = f"{output_dir}/LST-1.Run{run:05d}.{subrun:04d}.dl2.h5"
                self.predict_lstchain_data(
                    input_file,
                    output_file,
                    config_dir=output_dir,
                    overwrite=overwrite,
                    run=run,
                    subrun=subrun,
                    plot=plot,
                    batch_size=batch_size,
                )
        else:
            raise ValueError(
                f"To predict LST data run-wise, the cluster must be either 'cscs' or 'lst-cluster'. Current cluster : {self.cluster_configuration.cluster}"
            )

    def predict_lstchain_data(
        self,
        input_file,
        output_file,
        pointing_table="/dl1/event/telescope/parameters/LST_LSTCam",
        config_dir=None,
        overwrite=False,
        run=None,
        subrun=None,
        plot=False,
        batch_size=64,
    ):
        """
        Predict data using the closest trained model for lstchain data.

        This method identifies the closest trained model to the input data
        and uses it to make predictions. The results are saved to the specified
        output file.

        Parameters
        ----------
        input_file : str
            Path to the input file containing lstchain data.
        output_file : str
            Path to the output file where predictions will be saved.
        pointing_table : str, optional
            Path to the pointing table in the input file. Default is 
            "/dl1/event/telescope/parameters/LST_LSTCam".
        config_dir : str, optional
            Directory containing configuration files for the model. Default is None.
        overwrite : bool, optional
            Whether to overwrite the output file if it already exists. Default is False.
        run : int, optional
            Run number to filter the data. Default is None.
        subrun : int, optional
            Subrun number to filter the data. Default is None.
        plot : bool, optional
            Whether to generate and display plots for model selection. Default is False.
        batch_size : int, optional
            Batch size to use during prediction. Default is 64.

        Returns
        -------
        None
            This method does not return any value. The predictions are saved
            to the specified output file.

        Notes
        -----
        If no suitable trained model is found, the method will return without
        performing any predictions. If the output file already exists and 
        `overwrite` is set to False, the method will also return without 
        performing any predictions.
        """
        closest_tri_model = self.find_closest_model_to(
            input_file, pointing_table, plot=plot
        )
        if os.path.exists(output_file) and not overwrite:
            print(
                f"⚠️ Output file already exists and overwrite is set to False : {output_file}"
            )
            return
        if closest_tri_model is not None:
            closest_tri_model.predict_lstchain_data(
                input_file,
                output_file,
                config_dir=config_dir,
                overwrite=overwrite,
                run=run,
                subrun=subrun,
                pointing_table=pointing_table,
                batch_size=batch_size,
            )
        else:
            return

    def predict_data(
        self,
        input_file,
        output_file,
        pointing_table="dl0/monitoring/subarray/pointing",
        config_dir=None,
        overwrite=False,
        plot=False,
    ):
        """
        Predict data using the closest trained model.

        Parameters
        ----------
        input_file : str
            Path to the input file containing the data to be predicted.
        output_file : str
            Path to the output file where the predictions will be saved.
        pointing_table : str, optional
            Path to the pointing table within the input file, by default "dl0/monitoring/subarray/pointing".
        config_dir : str, optional
            Directory containing configuration files for the prediction, by default None.
        overwrite : bool, optional
            Whether to overwrite the output file if it already exists, by default False.
        plot : bool, optional
            Whether to generate and display plots during the process, by default False.

        Returns
        -------
        None
            This method does not return anything. If no suitable model is found, the function exits early.
        """
        closest_tri_model = self.find_closest_model_to(
            input_file, pointing_table, plot=plot
        )
        if closest_tri_model is not None:
            closest_tri_model.predict_data(
                input_file,
                output_file,
                config_dir=config_dir,
                overwrite=overwrite,
                pointing_table=pointing_table,
            )
        else:
            return

    def find_closest_model_to(
        self,
        input_file,
        pointing_table,
        plot=False,
        alt_key="alt_tel",
        az_key="az_tel",
        verbose=True,
    ):
        """
        Find the closest model to the average pointing of a given input file.

        Parameters
        ----------
        input_file : str
            Path to the input file containing pointing data.
        pointing_table : str
            Path to the pointing table file.
        plot : bool, optional
            If True, plot the pointing and model ranges on a polar plot. Default is False.
        alt_key : str, optional
            Key for altitude in the pointing table. Default is "alt_tel".
        az_key : str, optional
            Key for azimuth in the pointing table. Default is "az_tel".
        verbose : bool, optional
            If True, print detailed information about the closest model. Default is True.

        Returns
        -------
        closest_model : object
            The model from `self.tri_models` that is closest to the average pointing.

        Notes
        -----
        This method calculates the average pointing (zenith and azimuth) of the input file
        and compares it to the average pointing ranges of the models in `self.tri_models`.
        The closest model is determined based on the angular distance.

        Raises
        ------
        Exception
            If the pointing data cannot be found in the provided pointing table.
        """
        import astropy.units as u
        from astropy.io.misc.hdf5 import read_table_hdf5

        from ctlearn_manager.utils.utils import get_avg_pointing
        from .model_manager import IndexTables

        try:
            avg_data_ze, avg_data_az = get_avg_pointing(
                input_file,
                pointing_table=pointing_table,
                alt_key=alt_key,
                az_key=az_key,
            )
        except:
            # print(f"⚠️ Error reading pointing data from {pointing_table}: {e}")
            print(f"⚠️ Pointing not found at {pointing_table}, skipping : {input_file}")
            return

        min_distances = []
        
        for tri_model in self.tri_models:
            training_gamma_table = read_table_hdf5(
                tri_model.direction_model.project_directories.model_index_file,
                path=IndexTables(tri_model.direction_model, ParticleType.GAMMA_DIFFUSE).TRAINING.table_path,
            )
            zeniths_nodes = training_gamma_table[
                "training_gamma_diffuse_zenith_distances"
            ]
            azimuths_nodes = training_gamma_table["training_gamma_diffuse_azimuths"].to(
                u.rad
            )
            angular_distances = angular_distance(
                avg_data_ze, avg_data_az, 
                zeniths_nodes, azimuths_nodes
            )
            min_distances.append(np.min(angular_distances).value)

        

        # avg_model_azs = []
        # avg_model_zes = []
        # for tri_model in self.tri_models:
        #     avg_model_azs.append(
        #         np.mean(tri_model.direction_model.validity.azimuth_range)
        #         .to(u.deg)
        #         .value
        #     )
        #     avg_model_zes.append(
        #         np.mean(tri_model.direction_model.validity.zenith_range).to(u.deg).value
        #     )
        # avg_model_azs = np.array(avg_model_azs) * u.deg
        # avg_model_zes = np.array(avg_model_zes) * u.deg
        # # angular_distance_matrix = angular_distance(avg_data_ze, avg_data_az, avg_model_zes, avg_model_azs)
        # closest_model_index = np.argmin(
        #     angular_distance(avg_data_ze, avg_data_az, avg_model_zes, avg_model_azs)
        # )
        closest_model_index = np.argmin(min_distances)
        closest_model = self.tri_models[closest_model_index]

        if verbose:
            print(
                f"📁 File : {input_file.split('/')[-1]}      📡 Pointing : ({avg_data_ze.value:.3f}, {avg_data_az.value:.3f})      🧠 Closest Model {closest_model.direction_model.model_nickname} : ({np.mean(closest_model.direction_model.validity.zenith_range).value:.3f}, {np.mean(closest_model.direction_model.validity.azimuth_range).value:.3f})"
            )
        # print(f"｜📡 Average pointing of {input_file.split('/')[-1]} : ({avg_data_ze:3f}, {avg_data_az:3f})")
        # print(f"｜🔍 Closest model avg node : ({np.mean(closest_model.direction_model.validity.zenith_range).value}, {np.mean(closest_model.direction_model.validity.azimuth_range).value})")
        # print(f"｜🧠 Using models {closest_model.direction_model.model_nickname}, {closest_model.energy_model.model_nickname} and {closest_model.type_model.model_nickname}")
        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            closest_model.direction_model.plot_zenith_azimuth_ranges(ax)
            ax.scatter(
                avg_data_az.to(u.rad),
                avg_data_ze,
                label="Average pointing",
                color=get_color("ctlearn_highlight"),
            )
            ax.legend()
            plt.show()
        return closest_model

    def plot_zenith_azimuth_ranges(self, plot_testing_nodes=True):
        """
        Plot the zenith and azimuth ranges for all tri-models in a polar plot.

        Parameters
        ----------
        plot_testing_nodes : bool, optional
            If True, include testing nodes in the plot. Default is True.

        Notes
        -----
        This method creates a polar plot using Matplotlib and iterates over all
        tri-models to plot their respective zenith and azimuth ranges. The plot
        is displayed using `plt.show()`.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

        for tri_model in self.tri_models:
            tri_model.direction_model.plot_zenith_azimuth_ranges(
                ax, plot_testing_nodes=plot_testing_nodes
            )
        plt.show()

    @u.quantity_input(zenith=u.deg, azimuth=u.deg)
    def plot_energy_resolution_DL2(
        self,
        cuts: Cuts = DefaultCuts.GH_0_9.value,
        zenith: float = None,
        azimuth: float = None,
        ylim=None,
        particle_type: ParticleType = ParticleType.GAMMA_POINT,
        figsize=None,
        plot_RF=False,
        compare_with: str = None,
        output_file=None,
    ):
        """
        Plot the energy resolution for DL2 data.

        Parameters
        ----------
        cuts : Cuts, optional
            The cuts to apply for the analysis. Defaults to `DefaultCuts.GH_0_9.value`.
        zenith : float, optional
            The zenith angle in degrees. Required if `plot_RF` is True or `compare_with` is provided.
        azimuth : float, optional
            The azimuth angle in degrees. Required if `compare_with` is provided.
        ylim : tuple, optional
            The y-axis limits for the plot.
        particle_type : ParticleType, optional
            The type of particle to analyze. Defaults to `ParticleType.GAMMA_POINT`.
        figsize : tuple, optional
            The size of the figure.
        plot_RF : bool, optional
            Whether to include the Random Forest (RF) benchmark in the plot. Defaults to False.
        compare_with : str, optional
            The label of the model to compare with. If provided, zenith and azimuth must also be specified.

        Raises
        ------
        ValueError
            If `compare_with` is provided but `zenith` or `azimuth` is not specified.

        Notes
        -----
        - If `plot_RF` is True, the function will load the corresponding IRF file based on the zenith angle.
        - When `compare_with` is specified, the function will plot the relative improvement of the energy resolution
          compared to the reference model.
        """
        compare_with_index = [
            i for i, label in enumerate(self.model_labels) if label == compare_with
        ]
        if compare_with is not None:
            fig, (ax, ax_rel) = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": [3, 1]},
                sharex=True
            )
            plt.subplots_adjust(hspace=0)
            ax.set_xticks([])
            ax.set_xlabel("")
            ax_rel.set_xlabel("True Energy (TeV)")
            ax_rel.set_ylabel("Rel. Impr. (%)")
            ax_rel.grid(True, linestyle="--", alpha=0.5)
            ax_rel.set_xscale("log")
            ax_rel.set_ymargin(0.05)
            # ax_rel.set_yticks([0, 10, 20, 30, 40, 50])
        else:
            fig, ax = plt.subplots()
        cuts.plot_cuts_info_plt(ax)
        if (
            plot_RF
            and cuts.cut_type == CutType.EFFICIENCY_OPTIMIZED
            and zenith is not None
        ):
            import importlib
            import importlib.resources as pkg_resources

            from astropy.io import fits

            module_name = "ctlearn_manager.resources.irfs.LST1"
            RF_bechmpark = importlib.import_module(module_name)
            available_zeniths = [10.00, 23.63, 32.06, 43.20]
            closest_zenith = min(available_zeniths, key=lambda x: abs(x - zenith.value))

            with pkg_resources.path(
                RF_bechmpark,
                f"irfs_zen_{closest_zenith:.2f}_gh-eff_{cuts.efficiency_gammaness}.fits.gz",
            ) as irf_file:
                # irf_file = "/users/blacave/PhD/Software/CTLearn-Manager/src/ctlearn_manager/resources/irfs/LST1/irfs_zen_10.00_gh-eff_0.7.fits.gz"
                hudl = fits.open(irf_file)
                # plt.plot(hudl['ANGULAR_RESOLUTION'].data['true_energy_center'],hudl['ANGULAR_RESOLUTION'].data['angular_resolution'])
                RF_e = hudl["ENERGY_BIAS_RESOLUTION"].data["true_energy_center"]
                RF_e_res = hudl["ENERGY_BIAS_RESOLUTION"].data["resolution"]
                l = f"RF {closest_zenith:.1f}°"
                if f"{zenith.value:.2f}" == f"{closest_zenith:.2f}":
                    l = "RF"
                ax.plot(RF_e, RF_e_res, label=l, color=get_color('on_background'), zorder=0)
        if zenith is not None and azimuth is not None:
            zeniths = np.array([zenith.value]) * zenith.unit
            azimuths = np.array([azimuth.value]) * azimuth.unit
            text_color = get_color("ctlearn_accent_2")
            background_color = get_color("ctlearn_accent_1")
            ax.text(
                0.02,
                0.02,
                f"Pointing: ({zenith.value:.1f}, {azimuth.value:.1f})°",
                transform=ax.transAxes,
                fontsize=9,
                color=text_color,
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    edgecolor="none",
                    facecolor=background_color,
                    alpha=0.2,
                ),
            )
            if compare_with is not None:
                # ax.set_xscale(ax_rel.get_xscale())
                ax.set_xticks([])
                ax.set_xlabel("")
                fig.subplots_adjust(hspace=0)
                if len(compare_with_index) > 0:
                    ref_e_bins, ref_e_res_err = self.tri_models[
                        compare_with_index[0]
                    ].get_energy_resolution_DL2(
                        zenith=zenith,
                        azimuth=azimuth,
                        cuts=cuts,
                        particle_type=particle_type,
                    )
                    ref_e = (ref_e_bins[:-1] + ref_e_bins[1:]) / 2
                    ref_e_res = [e_r[0] for e_r in ref_e_res_err]
                elif compare_with == "RF" and plot_RF:
                    ref_e = RF_e
                    ref_e_res = RF_e_res
                    ax_rel.plot(
                        ref_e,
                        [0] * len(ref_e),
                        label=f"{compare_with} vs {compare_with}",
                        color=get_color("on_background"),
                        zorder=0,
                    )
                for tri_model, label in tqdm(
                    zip(self.tri_models, self.model_labels),
                    desc="Plotting energy resolution improvment",
                    unit="model",
                    total=len(self.tri_models),
                ):
                    try:
                        e_bins, e_res_err = tri_model.get_energy_resolution_DL2(
                            zenith=zenith,
                            azimuth=azimuth,
                            cuts=cuts,
                            particle_type=particle_type,
                        )
                    except:
                        continue
                    e = (e_bins[:-1] + e_bins[1:]) / 2
                    e_res = [e_r[0] for e_r in e_res_err]
                    # print(e, ref_e, ref_e_res)
                    if not np.array_equal(e, ref_e):
                        ref_e_res_interp = np.interp(e.value, ref_e, ref_e_res)
                    else:
                        ref_e_res_interp = ref_e_res
                    relative_improvement = (
                        100
                        * (np.array(ref_e_res_interp) - np.array(e_res))
                        / np.array(ref_e_res_interp)
                    )
                    ax_rel.plot(
                        e, relative_improvement, label=f"{label} vs {compare_with}"
                    )

        else:
            if compare_with is not None:
                raise ValueError(
                    "If you want to compare with another model, you need to provide zenith and azimuth."
                )
            zeniths = None
            azimuths = None

        for tri_model, label in tqdm(
            zip(self.tri_models, self.model_labels),
            desc="Plotting energy resolution",
            unit="model",
            total=len(self.tri_models),
        ):
            l = tri_model.energy_model.model_nickname if label is None else label
            tri_model.plot_energy_resolution_DL2(
                zeniths=zeniths,
                azimuths=azimuths,
                cuts=[cuts],
                ylim=ylim,
                particle_type=particle_type,
                ax=ax,
                figsize=figsize,
                label=l,
            )

        if compare_with is not None:
            ax_rel.set_xlim(ax.get_xlim())
            # ax_rel.set_ylim(bottom=0)
        ax.legend()
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.0)
        if output_file is not None:
            plt.savefig(output_file, dpi=300)
            # print(f"Saved plot to {output_file}")
        else:
            plt.show()

    def plot_angular_resolution_DL2(
        self,
        cuts: Cuts = DefaultCuts.GH_0_9.value,
        zenith: float = None,
        azimuth: float = None,
        ylim=None,
        particle_type: ParticleType = ParticleType.GAMMA_POINT,
        figsize=None,
        plot_RF=False,
        compare_with: str = None,
        output_file=None,
    ):
        
        compare_with_index = [
            i for i, label in enumerate(self.model_labels) if label == compare_with
        ]
        if compare_with is not None:
            fig, (ax, ax_rel) = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": [3, 1]},
                sharex=True
            )
            plt.subplots_adjust(hspace=0)
            ax.set_xticks([])
            ax.set_xlabel("")
            ax_rel.set_xlabel("True Energy (TeV)")
            ax_rel.set_ylabel("Rel. Impr. (%)")
            ax_rel.grid(True, linestyle="--", alpha=0.5)
            ax_rel.set_xscale("log")
            ax_rel.set_ymargin(0.05)
            # ax_rel.set_yticks([0, 10, 20, 30, 40, 50])
        else:
            fig, ax = plt.subplots()
        stored_efficiency_theta = cuts.efficiency_theta
        cuts.efficiency_theta = None
        cuts.plot_cuts_info_plt(ax)
        cuts.efficiency_theta = stored_efficiency_theta
        if (
            plot_RF
            and cuts.cut_type == CutType.EFFICIENCY_OPTIMIZED
            and zenith is not None
        ):
            import importlib
            import importlib.resources as pkg_resources

            from astropy.io import fits

            module_name = "ctlearn_manager.resources.irfs.LST1"
            RF_bechmpark = importlib.import_module(module_name)
            available_zeniths = [10.00, 23.63, 32.06, 43.20]
            closest_zenith = min(available_zeniths, key=lambda x: abs(x - zenith.value))
            with pkg_resources.path(
                RF_bechmpark,
                f"irfs_zen_{closest_zenith:.2f}_gh-eff_{cuts.efficiency_gammaness}.fits.gz",
            ) as irf_file:
                # irf_file = "/users/blacave/PhD/Software/CTLearn-Manager/src/ctlearn_manager/resources/irfs/LST1/irfs_zen_10.00_gh-eff_0.7.fits.gz"
                hudl = fits.open(irf_file)
                RF_e = hudl["ANGULAR_RESOLUTION"].data["true_energy_center"]
                RF_ang_res = hudl["ANGULAR_RESOLUTION"].data["angular_resolution"]
                l = f"RF {closest_zenith:.1f}°"
                if f"{zenith.value:.2f}" == f"{closest_zenith:.2f}":
                    l = "RF"
                ax.plot(RF_e, RF_ang_res, label=l, color=get_color('on_background'), zorder=0)
            # ax.plot(hudl['ENERGY_BIAS_RESOLUTION'].data['true_energy_center'],hudl['ENERGY_BIAS_RESOLUTION'].data['resolution'], label='RF', color='k', zorder=0)
        if zenith is not None and azimuth is not None:
            zeniths = np.array([zenith.value]) * zenith.unit
            azimuths = np.array([azimuth.value]) * azimuth.unit
            text_color = get_color("ctlearn_accent_2")
            background_color = get_color("ctlearn_accent_1")
            ax.text(
                0.02,
                0.02,
                f"Pointing: ({zenith.value:.1f}, {azimuth.value:.1f})°",
                transform=ax.transAxes,
                fontsize=9,
                color=text_color,
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    edgecolor="none",
                    facecolor=background_color,
                    alpha=0.2,
                ),
            )
            if compare_with is not None:
                # ax.set_xscale(ax_rel.get_xscale())

                # ax.set_xticks([])
                # ax.set_xlabel("")
                fig.subplots_adjust(hspace=0)
                if len(compare_with_index) > 0:
                    ref_e_bins, ref_ang_res_err = self.tri_models[
                        compare_with_index[0]
                    ].get_angular_resolution_DL2(
                        zenith=zenith,
                        azimuth=azimuth,
                        cuts=cuts,
                        particle_type=particle_type,
                    )
                    ref_e = (ref_e_bins[:-1].value + ref_e_bins[1:].value) / 2
                    ref_ang_res = [e_r[0].value for e_r in ref_ang_res_err]
                elif compare_with == "RF" and plot_RF:
                    ref_e = RF_e
                    ref_ang_res = RF_ang_res
                    ax_rel.plot(
                        ref_e,
                        [0] * len(ref_e),
                        label=f"{compare_with} vs {compare_with}",
                        color=get_color("on_background"),
                        zorder=0,
                    )
                for tri_model, label in tqdm(
                    zip(self.tri_models, self.model_labels),
                    desc="Plotting angular resolution improvment",
                    unit="model",
                    total=len(self.tri_models),
                ):
                    try:
                        e_bins, e_res_err = tri_model.get_angular_resolution_DL2(
                            zenith=zenith,
                            azimuth=azimuth,
                            cuts=cuts,
                            particle_type=particle_type,
                        )
                    except:
                        continue
                    e = (e_bins[:-1].value + e_bins[1:].value) / 2
                    e_res = [e_r[0].value for e_r in e_res_err]
                    if not np.array_equal(e, ref_e):
                        ref_e_res_interp = np.interp(e, ref_e, ref_ang_res)
                    else:
                        ref_e_res_interp = ref_ang_res
                    relative_improvement = (
                        100
                        * (np.array(ref_e_res_interp) - np.array(e_res))
                        / np.array(ref_e_res_interp)
                    )
                    ax_rel.plot(
                        e, relative_improvement, label=f"{label} vs {compare_with}"
                    )
                    # ax_rel.text(e[np.where(relative_improvement == np.max(relative_improvement))][0], np.max(relative_improvement), f"{int(np.max(relative_improvement))}", fontsize=8)
        else:
            zeniths = None
            azimuths = None
        for tri_model, label in tqdm(
            zip(self.tri_models, self.model_labels),
            desc="Plotting angular resolution",
            unit="model",
            total=len(self.tri_models),
        ):
            l = tri_model.direction_model.model_nickname if label is None else label
            tri_model.plot_angular_resolution_DL2(
                zeniths=zeniths,
                azimuths=azimuths,
                cuts=[cuts],
                ylim=ylim,
                particle_type=particle_type,
                ax=ax,
                figsize=figsize,
                label=l,
            )

        if compare_with is not None:
            ax_rel.set_xlim(ax.get_xlim())
            # ax_rel.set_ylim(bottom=0)
        ax.legend()
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.0)
        if output_file is not None:
            plt.savefig(output_file)
        else:
            plt.show()

    def plot_cuts(self, cuts: Cuts = DefaultCuts.EFF_70.value):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        cuts.plot_cuts_info_plt(axs[0])
        cuts.plot_cuts_info_plt(axs[1])
        for tri_model, label in tqdm(
            zip(self.tri_models, self.model_labels), desc="Plotting cuts", unit="model"
        ):
            l = tri_model.direction_model.model_nickname if label is None else label
            tri_model.plot_cuts(cuts=[cuts], axs=axs, label=l)
        axs[0].legend()
        axs[1].legend()
        plt.tight_layout()
        plt.show()

    # def plot_benchmark(self, cuts: Cuts=DefaultCuts.GH_0_9.value, ylim=None, particle_type: ParticleType=ParticleType.GAMMA_POINT, figsize=None):
    #     fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    #     cuts.plot_cuts_info_plt(axs[0])
    #     cuts.plot_cuts_info_plt(axs[1])
    #     for tri_model, label in tqdm(zip(self.tri_models, self.model_labels), desc="Plotting benchmarking", unit="model"):
    #         l = tri_model.direction_model.model_nickname if label is None else label
    #         tri_model.plot_benchmark(cuts=[cuts], ylim=ylim, particle_type=particle_type, axs=axs, figsize=figsize, label=l)
    #     axs[0].legend()
    #     axs[1].legend()
    #     plt.show()

    def plot_everything_dl2(
        self,
        # output_directory: str,
        dl2_files: list[str],
        gammaness_cut: float = 0.9,
        edep_cuts: bool = False,
        pointing_table: str = "/dl1/monitoring/telescope/pointing/tel_001",
    ):
        """
        Plot the angular resolution, energy resolution, and gammaness for DL2 data.
        This function generates plots for the angular resolution, energy resolution,
        and gammaness for the given DL2 files. It uses ctaplot to create the plots
        and saves them in the specified output directory.

        Parameters
        ----------
        output_directory : str
            The directory where the plots will be saved.
        dl2_files : list[str]
            List of DL2 files to be processed.
        dl2_processed_dir : str
            The directory where the processed DL2 files are stored.
        gammaness_cut : float, optional
            The gammaness cut value to be applied. Default is 0.9.

        Returns
        -------
        None
        """
        import os
        import pickle
        import concurrent.futures

        grouped_files = {tri_model: [] for tri_model in self.tri_models}

        def assign_model(dl2_file):
            closest_tri_model = self.find_closest_model_to(
                dl2_file,
                pointing_table=pointing_table,
                plot=False,
                alt_key="altitude",
                az_key="azimuth",
                verbose=False,
            )
            return (closest_tri_model, dl2_file)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(assign_model, dl2_files),
                total=len(dl2_files),
                desc="Grouping DL2 files per model",
                unit="file"
            ))

        for closest_tri_model, dl2_file in results:
            if closest_tri_model is not None:
                grouped_files[closest_tri_model].append(dl2_file)

        # for dl2_file in tqdm(
        #     dl2_files, desc="Grouping DL2 files per model", unit="file"
        # ):
        #     closest_tri_model = self.find_closest_model_to(
        #         dl2_file,
        #         pointing_table=pointing_table,
        #         plot=False,
        #         alt_key="altitude",
        #         az_key="azimuth",
        #         verbose=False,
        #     )
        #     if closest_tri_model is not None:
        #         grouped_files[closest_tri_model].append(dl2_file)

        # Filter out empty groups
        grouped_files = {
            model: files for model, files in grouped_files.items() if files
        }
        n = []
        models = []
        for tri_model, files in grouped_files.items():
            output_directory = tri_model.project_directories.dl2_post_processed_data_directory
            n.append(len(files))
            models.append(tri_model.project_directories.tri_model_nickname)
            print(
                f"Processing {len(files)} files 🧠🧠🧠 CTLearnTriModelManager ▮ {tri_model.direction_model.model_nickname} ▮ {tri_model.energy_model.model_nickname} ▮ {tri_model.type_model.model_nickname} ▮"
            )
            tri_model_file = f"{output_directory}/tri_model_{tri_model.direction_model.model_nickname}.pkl"
            tri_model.dl2_data_files = files

            use_cluster = tri_model.cluster_configuration.use_cluster
            tri_model.cluster_configuration.use_cluster = False  # if some DL2 files were not processed, they will be processed in the same job as the plotting job, and not submit multiple new jobs
            os.makedirs(output_directory, exist_ok=True)
            with open(tri_model_file, "wb") as f:
                pickle.dump(tri_model, f)
            tri_model.cluster_configuration.use_cluster = use_cluster
            # print(edep_cuts)
            cmd = f"plot_dl2 --stereo_tri_model {tri_model_file} --output_directory {output_directory} --gammaness_cut {gammaness_cut} --edep_cuts={edep_cuts}"
            print(cmd)
            sbatch_file = tri_model.cluster_configuration.write_sbatch_script(
                f"dl2_plots_{tri_model.direction_model.model_nickname}",
                cmd,
                output_directory,
                use_gpu_cscs=False,
            )
            if self.cluster_configuration.use_cluster:
                os.system(f"sbatch {sbatch_file}")
            else:
                os.system(cmd)
        if len(self.tri_models) > 1:
            plt.bar(models, n) 
            plt.xlabel("CTLearn TriModel")
            plt.ylabel("Number of DL2 files")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig("dl2_files_per_model.png")
            plt.show()

    def get_tri_model_by_nickname(self, tri_model_nickname):
        """
        Get a tri-model by its nickname.

        Parameters
        ----------
        nickname : str
            The nickname of the tri-model to retrieve.

        Returns
        -------
        CTLearnTriModel
            The tri-model with the specified nickname, or None if not found.
        """
        for tri_model in self.tri_models:
            if tri_model.project_directories.tri_model_nickname == tri_model_nickname:
                return tri_model
        return None
