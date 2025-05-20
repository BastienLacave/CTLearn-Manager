import os

import astropy.units as u
import ctadata
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
    CTLearnManagerStyle,
)

__all__ = ['TriModelCollection']

class TriModelCollection:
    
    def __init__(self, tri_models: list, cluster_configuration:ClusterConfiguration=ClusterConfiguration(), model_labels: list[str]=None):
        self.tri_models = tri_models
        self.cluster_configuration = cluster_configuration
        for tri_model in self.tri_models:
            tri_model.cluster_configuration = cluster_configuration
        telescope_ids = [tri_model.telescope_ids for tri_model in self.tri_models]
        telescope_names = [tri_model.telescope_names for tri_model in self.tri_models]
        stereos = [tri_model.stereo for tri_model in self.tri_models]
        if model_labels is not None:
            assert len(model_labels) == len(self.tri_models), "Model labels must be the same length as the number of tri models."
            self.model_labels = model_labels
        else:
            self.model_labels = [f'Model_{j}'for j in range(len(self.tri_models))]
        assert len(set(stereos)) == 1, "All stereos in the collection must be the same."
        set_mpl_style()
        # assert len(set(telescope_ids)) == 1, "All telescope_ids in the collection must be the same."
        # assert len(set(telescope_names)) == 1, "All telescope_names in the collection must be the same."

    def predict_lstchain_run(self, run: int, output_dir: str, DL1_data_dir=None, overwrite=False, plot=False, batch_size=64):
        os.makedirs(output_dir, exist_ok=True)
        if self.cluster_configuration.cluster == 'cscs':
            if DL1_data_dir is None:
                DL1_data_dir = "/pnfs/cta.cscs.ch/lst/DL1/"
            input_files, v = get_files_cscs(run, DL1_data_dir)
            scratch_dir = os.getenv('SCRATCH')
            scratch_dl1_dir = f"{scratch_dir}/ctlearn_manager_dl1_from_dcache/{run:05d}/{v}/tailcut84/"
            os.system(f"mkdir -p {scratch_dl1_dir}")
            current_directory = os.getcwd()
            print(f"DL1 files will be copied to {scratch_dl1_dir}\n")
            for dcache_file in input_files:
                input_file = f"{scratch_dl1_dir}/{dcache_file.split('/')[-1]}"
                if not os.path.exists(input_file):
                    print(f"⌛ Copying {dcache_file} to {scratch_dl1_dir}")
                    ctadata.fetch_and_save_file_or_dir(dcache_file)
                    os.system(f"mv {current_directory}/{dcache_file.split('/')[-1]} {scratch_dl1_dir}/{dcache_file.split('/')[-1]}")
                # print(f"Predicting {input_file}")
                subrun = int(input_file.split('.')[-2])
                output_file = f"{output_dir}/LST-1.Run{run:05d}.{subrun:04d}.dl2.h5"
                self.predict_lstchain_data(input_file, output_file, config_dir=output_dir, overwrite=overwrite, run=run, subrun=subrun, plot=plot, batch_size=batch_size)

        elif self.cluster_configuration.cluster == 'lst-cluster':
            if DL1_data_dir is None:
                DL1_data_dir = "/fefs/aswg/data/real/DL1/"
            input_files = get_files_LST_cluster(run, DL1_data_dir)
            for input_file in input_files:
                print(f"Predicting {input_file}")
                subrun = int(input_file.split('.')[-2])
                output_file = f"{output_dir}/LST-1.Run{run:05d}.{subrun:04d}.dl2.h5"
                self.predict_lstchain_data(input_file, output_file, config_dir=output_dir, overwrite=overwrite, run=run, subrun=subrun, plot=plot, batch_size=batch_size)
        else:
            raise ValueError(f"To predict LST data run-wise, the cluster must be either 'cscs' or 'lst-cluster'. Current cluster : {self.cluster_configuration.cluster}")
         
    def predict_lstchain_data(self, input_file, output_file, pointing_table='/dl1/event/telescope/parameters/LST_LSTCam', config_dir=None, overwrite=False, run=None, subrun=None, plot=False, batch_size=64):
        closest_tri_model = self.find_closest_model_to(input_file, pointing_table, plot=plot)
        if os.path.exists(output_file) and not overwrite:
            print(f"⚠️ Output file already exists and overwrite is set to False : {output_file}")
            return
        if closest_tri_model is not None:
            closest_tri_model.predict_lstchain_data(input_file, output_file, config_dir=config_dir, overwrite=overwrite, run=run, subrun=subrun, pointing_table=pointing_table, batch_size=batch_size)
        else:
            return
        
    def predict_data(self, input_file, output_file, pointing_table='dl0/monitoring/subarray/pointing', config_dir=None, overwrite=False, plot=False):
        closest_tri_model = self.find_closest_model_to(input_file, pointing_table, plot=plot)
        if closest_tri_model is not None:
            closest_tri_model.predict_data(input_file, output_file, config_dir=config_dir, overwrite=overwrite, pointing_table=pointing_table)
        else:
            return
        
    def find_closest_model_to(self, input_file, pointing_table, plot=False, alt_key='alt_tel', az_key='az_tel', verbose=True):
        import astropy.units as u

        from ctlearn_manager.utils.utils import get_avg_pointing
        try:
            avg_data_ze, avg_data_az = get_avg_pointing(input_file, pointing_table=pointing_table, alt_key=alt_key, az_key=az_key)
        except:
            print(f"⚠️ Pointing not found at {pointing_table}, skipping : {input_file}")
            return
        
        avg_model_azs = []
        avg_model_zes = []
        for tri_model in self.tri_models:
            avg_model_azs.append(np.mean(tri_model.direction_model.validity.azimuth_range).to(u.deg).value)
            avg_model_zes.append(np.mean(tri_model.direction_model.validity.zenith_range).to(u.deg).value)
        avg_model_azs = np.array(avg_model_azs) * u.deg
        avg_model_zes = np.array(avg_model_zes) * u.deg
        # angular_distance_matrix = angular_distance(avg_data_ze, avg_data_az, avg_model_zes, avg_model_azs)
        closest_model_index = np.argmin(angular_distance(avg_data_ze, avg_data_az, avg_model_zes, avg_model_azs))
        closest_model = self.tri_models[closest_model_index]

        if verbose:
            print(f"📁 File : {input_file.split('/')[-1]}      📡 Pointing : ({avg_data_ze.value:.3f}, {avg_data_az.value:.3f})      🧠 Closest Model : ({np.mean(closest_model.direction_model.validity.zenith_range).value:.3f}, {np.mean(closest_model.direction_model.validity.azimuth_range).value:.3f})")
        # print(f"｜📡 Average pointing of {input_file.split('/')[-1]} : ({avg_data_ze:3f}, {avg_data_az:3f})")
        # print(f"｜🔍 Closest model avg node : ({np.mean(closest_model.direction_model.validity.zenith_range).value}, {np.mean(closest_model.direction_model.validity.azimuth_range).value})")
        # print(f"｜🧠 Using models {closest_model.direction_model.model_nickname}, {closest_model.energy_model.model_nickname} and {closest_model.type_model.model_nickname}")
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            closest_model.direction_model.plot_zenith_azimuth_ranges(ax)
            ax.scatter(avg_data_az.to(u.rad), avg_data_ze, label='Average pointing', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3])
            ax.legend()
            plt.show()
        return closest_model

    def plot_zenith_azimuth_ranges(self, plot_testing_nodes=True):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        for tri_model in self.tri_models:
            tri_model.direction_model.plot_zenith_azimuth_ranges(ax, plot_testing_nodes=plot_testing_nodes)
        plt.show()

    @u.quantity_input(zenith=u.deg,azimuth=u.deg) 
    def plot_energy_resolution_DL2(self, cuts: Cuts=DefaultCuts.GH_0_9.value, zenith: float=None, azimuth: float=None, ylim=None, particle_type: ParticleType=ParticleType.GAMMA_POINT, figsize=None, plot_RF=False, compare_with: str=None):
        compare_with_index = [i for i, label in enumerate(self.model_labels) if label == compare_with]
        if compare_with is not None:
            fig, (ax, ax_rel) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
            ax_rel.set_xlabel("True Energy (TeV)")
            ax_rel.set_ylabel("Rel. Impr. (%)")
            ax_rel.grid(True, linestyle='--', alpha=0.5)
            ax_rel.set_xscale('log')
            ax_rel.set_ymargin(0.05)
            ax_rel.set_yticks([0, 10, 20, 30, 40, 50])
        else:
            fig, ax = plt.subplots()
        cuts.plot_cuts_info_plt(ax)
        if plot_RF and cuts.cut_type == CutType.EFFICIENCY_OPTIMIZED and zenith is not None:
            from astropy.io import fits
            import importlib
            import importlib.resources as pkg_resources
            module_name = f"ctlearn_manager.resources.irfs.LST1"
            RF_bechmpark = importlib.import_module(module_name)
            available_zeniths = [10.00, 23.63, 32.06, 43.20]
            closest_zenith = min(available_zeniths, key=lambda x: abs(x - zenith.value))
            
            with pkg_resources.path(RF_bechmpark, f'irfs_zen_{closest_zenith:.2f}_gh-eff_{cuts.efficiency_gammaness}.fits.gz') as irf_file:
                # irf_file = "/users/blacave/PhD/Software/CTLearn-Manager/src/ctlearn_manager/resources/irfs/LST1/irfs_zen_10.00_gh-eff_0.7.fits.gz"
                hudl = fits.open(irf_file)  
                # plt.plot(hudl['ANGULAR_RESOLUTION'].data['true_energy_center'],hudl['ANGULAR_RESOLUTION'].data['angular_resolution'])
                RF_e = hudl['ENERGY_BIAS_RESOLUTION'].data['true_energy_center']
                RF_e_res = hudl['ENERGY_BIAS_RESOLUTION'].data['resolution']
                l = f'RF {closest_zenith:.1f}°'
                if f"{zenith.value:.2f}" == f"{closest_zenith:.2f}":
                    l = 'RF'
                ax.plot(RF_e, RF_e_res, label=l, color='k', zorder=0)
        if zenith is not None and azimuth is not None:
            zeniths = np.array([zenith.value]) * zenith.unit
            azimuths = np.array([azimuth.value]) * azimuth.unit
            text_color=CTLearnManagerStyle.ctlearn_accent_2.value
            background_color=CTLearnManagerStyle.ctlearn_accent_1.value
            ax.text(
            0.02, 0.02, f"Pointing: ({zenith.value:.1f}, {azimuth.value:.1f})°",
            transform=ax.transAxes,
            fontsize=9,
            color=text_color,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor=background_color, alpha=0.2),
            )
            if compare_with is not None:
                # ax.set_xscale(ax_rel.get_xscale())
                ax.set_xticks([])
                ax.set_xlabel("")
                fig.subplots_adjust(hspace=0)
                if len(compare_with_index) > 0:
                    ref_e_bins, ref_e_res_err = self.tri_models[compare_with_index[0]].get_energy_resolution_DL2(zenith=zenith, azimuth=azimuth, cuts=cuts, particle_type=particle_type)
                    ref_e = (ref_e_bins[:-1] + ref_e_bins[1:]) / 2
                    ref_e_res = [e_r[0] for e_r in ref_e_res_err]
                elif compare_with == 'RF' and plot_RF:
                    ref_e = RF_e
                    ref_e_res = RF_e_res
                    ax_rel.plot(ref_e, [0] * len(ref_e), label=f"{compare_with} vs {compare_with}", color='k', zorder=0)
                for tri_model, label in tqdm(zip(self.tri_models, self.model_labels), desc="Plotting energy resolution improvment", unit="model", total=len(self.tri_models)):
                    try:
                        e_bins, e_res_err = tri_model.get_energy_resolution_DL2(zenith=zenith, azimuth=azimuth, cuts=cuts, particle_type=particle_type)
                    except:
                        continue
                    e = (e_bins[:-1] + e_bins[1:]) / 2
                    e_res = [e_r[0] for e_r in e_res_err]
                    if not np.array_equal(e, ref_e):
                        ref_e_res_interp = np.interp(e, ref_e, ref_e_res)
                    else:
                        ref_e_res_interp = ref_e_res
                    relative_improvement = 100 * (np.array(ref_e_res_interp) - np.array(e_res)) / np.array(ref_e_res_interp)
                    ax_rel.plot(e, relative_improvement, label=f"{label} vs {compare_with}")

        else:
            if compare_with is not None:
                raise ValueError("If you want to compare with another model, you need to provide zenith and azimuth.")
            zeniths = None
            azimuths = None

        
        for tri_model, label in tqdm(zip(self.tri_models, self.model_labels), desc="Plotting energy resolution", unit="model", total=len(self.tri_models)):
            l = tri_model.energy_model.model_nickname if label is None else label
            tri_model.plot_energy_resolution_DL2(zeniths=zeniths, azimuths=azimuths, cuts=[cuts], ylim=ylim, particle_type=particle_type, ax=ax, figsize=figsize, label=l)
        
        if compare_with is not None:
            ax_rel.set_xlim(ax.get_xlim())
            ax_rel.set_ylim(bottom=0)
        ax.legend()
        plt.tight_layout()
        plt.subplots_adjust(hspace=.0)
        plt.show()

    def plot_angular_resolution_DL2(self, cuts: Cuts=DefaultCuts.GH_0_9.value, zenith: float=None, azimuth: float=None, ylim=None, particle_type: ParticleType=ParticleType.GAMMA_POINT, figsize=None, plot_RF=False, compare_with: str=None):
        compare_with_index = [i for i, label in enumerate(self.model_labels) if label == compare_with]
        if compare_with is not None:
            fig, (ax, ax_rel) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
            ax_rel.set_xlabel("True Energy (TeV)")
            ax_rel.set_ylabel("Rel. Impr. (%)")
            ax_rel.grid(True, linestyle='--', alpha=0.5)
            ax_rel.set_xscale('log')
            ax_rel.set_ymargin(0.05)
            ax_rel.set_yticks([0, 10, 20, 30, 40, 50])
        else:
            fig, ax = plt.subplots()
        stored_efficiency_theta = cuts.efficiency_theta
        cuts.efficiency_theta = None
        cuts.plot_cuts_info_plt(ax)
        cuts.efficiency_theta = stored_efficiency_theta
        if plot_RF and cuts.cut_type == CutType.EFFICIENCY_OPTIMIZED and zenith is not None:
            from astropy.io import fits
            import importlib
            import importlib.resources as pkg_resources
            module_name = f"ctlearn_manager.resources.irfs.LST1"
            RF_bechmpark = importlib.import_module(module_name)
            available_zeniths = [10.00, 23.63, 32.06, 43.20]
            closest_zenith = min(available_zeniths, key=lambda x: abs(x - zenith.value))
            with pkg_resources.path(RF_bechmpark, f'irfs_zen_{closest_zenith:.2f}_gh-eff_{cuts.efficiency_gammaness}.fits.gz') as irf_file:
                # irf_file = "/users/blacave/PhD/Software/CTLearn-Manager/src/ctlearn_manager/resources/irfs/LST1/irfs_zen_10.00_gh-eff_0.7.fits.gz"
                hudl = fits.open(irf_file)  
                RF_e = hudl['ANGULAR_RESOLUTION'].data['true_energy_center']
                RF_ang_res = hudl['ANGULAR_RESOLUTION'].data['angular_resolution']
                l = f'RF {closest_zenith:.1f}°'
                if f"{zenith.value:.2f}" == f"{closest_zenith:.2f}":
                    l = 'RF'
                ax.plot(RF_e, RF_ang_res, label=l, color='k', zorder=0)
            # ax.plot(hudl['ENERGY_BIAS_RESOLUTION'].data['true_energy_center'],hudl['ENERGY_BIAS_RESOLUTION'].data['resolution'], label='RF', color='k', zorder=0)
        if zenith is not None and azimuth is not None:
            zeniths = np.array([zenith.value]) * zenith.unit
            azimuths = np.array([azimuth.value]) * azimuth.unit
            text_color=CTLearnManagerStyle.ctlearn_accent_2.value
            background_color=CTLearnManagerStyle.ctlearn_accent_1.value
            ax.text(
            0.02, 0.02, f"Pointing: ({zenith.value:.1f}, {azimuth.value:.1f})°",
            transform=ax.transAxes,
            fontsize=9,
            color=text_color,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor=background_color, alpha=0.2),
            )
            if compare_with is not None:
                # ax.set_xscale(ax_rel.get_xscale())
                ax.set_xticks([])
                ax.set_xlabel("")
                fig.subplots_adjust(hspace=0)
                if len(compare_with_index) > 0:
                    ref_e_bins, ref_ang_res_err = self.tri_models[compare_with_index[0]].get_angular_resolution_DL2(zenith=zenith, azimuth=azimuth, cuts=cuts, particle_type=particle_type)
                    ref_e = (ref_e_bins[:-1].value + ref_e_bins[1:].value) / 2
                    ref_ang_res = [e_r[0].value for e_r in ref_ang_res_err]
                elif compare_with == 'RF' and plot_RF:
                    ref_e = RF_e
                    ref_ang_res = RF_ang_res
                    ax_rel.plot(ref_e, [0] * len(ref_e), label=f"{compare_with} vs {compare_with}", color='k', zorder=0)
                for tri_model, label in tqdm(zip(self.tri_models, self.model_labels), desc="Plotting angular resolution improvment", unit="model", total=len(self.tri_models)):
                    try:
                        e_bins, e_res_err = tri_model.get_angular_resolution_DL2(zenith=zenith, azimuth=azimuth, cuts=cuts, particle_type=particle_type)
                    except:
                        continue
                    e = (e_bins[:-1].value + e_bins[1:].value) / 2
                    e_res = [e_r[0].value for e_r in e_res_err]
                    if not np.array_equal(e, ref_e):
                        ref_e_res_interp = np.interp(e, ref_e, ref_ang_res)
                    else:
                        ref_e_res_interp = ref_ang_res
                    relative_improvement = 100 * (np.array(ref_e_res_interp) - np.array(e_res)) / np.array(ref_e_res_interp)
                    ax_rel.plot(e, relative_improvement, label=f"{label} vs {compare_with}")
                    # ax_rel.text(e[np.where(relative_improvement == np.max(relative_improvement))][0], np.max(relative_improvement), f"{int(np.max(relative_improvement))}", fontsize=8)
        else:
            zeniths = None
            azimuths = None
        for tri_model, label in tqdm(zip(self.tri_models, self.model_labels), desc="Plotting angular resolution", unit="model", total=len(self.tri_models)):
            l = tri_model.direction_model.model_nickname if label is None else label
            tri_model.plot_angular_resolution_DL2(zeniths=zeniths, azimuths=azimuths, cuts=[cuts], ylim=ylim, particle_type=particle_type, ax=ax, figsize=figsize, label=l)
        
        if compare_with is not None:
            ax_rel.set_xlim(ax.get_xlim())
            ax_rel.set_ylim(bottom=0)
        ax.legend()
        plt.tight_layout()
        plt.subplots_adjust(hspace=.0)
        plt.show()

    def plot_cuts(self, cuts: Cuts=DefaultCuts.EFF_70.value):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        cuts.plot_cuts_info_plt(axs[0])
        cuts.plot_cuts_info_plt(axs[1])
        for tri_model, label in tqdm(zip(self.tri_models, self.model_labels), desc="Plotting cuts", unit="model"):
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


    def plot_everything_dl2(self, output_directory: str, dl2_files: list[str], gammaness_cut: float=0.9, edep_cuts: bool=False, pointing_table: str='/dl1/monitoring/telescope/pointing/tel_001'):
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

        grouped_files = {tri_model: [] for tri_model in self.tri_models}

        for dl2_file in tqdm(dl2_files, desc="Grouping DL2 files per model", unit="file"):
            closest_tri_model = self.find_closest_model_to(dl2_file, pointing_table=pointing_table, plot=False, alt_key='altitude', az_key='azimuth', verbose=False)
            if closest_tri_model is not None:
                grouped_files[closest_tri_model].append(dl2_file)

        # Filter out empty groups
        grouped_files = {model: files for model, files in grouped_files.items() if files}

        for tri_model, files in grouped_files.items():
            print(f"Processing {len(files)} files 🧠🧠🧠 CTLearnTriModelManager ▮ {tri_model.direction_model.model_nickname} ▮ {tri_model.energy_model.model_nickname} ▮ {tri_model.type_model.model_nickname} ▮")
            tri_model_file = f"{output_directory}/tri_model_{tri_model.direction_model.model_nickname}.pkl"
            tri_model.dl2_data_files = files

            use_cluster = tri_model.cluster_configuration.use_cluster
            tri_model.cluster_configuration.use_cluster = False # if some DL2 files were not processed, they will be processed in the same job as the plotting job, and not submit multiple new jobs
            with open(tri_model_file, 'wb') as f:
                pickle.dump(tri_model, f)
            tri_model.cluster_configuration.use_cluster = use_cluster
            print(edep_cuts)
            cmd = f"plot_dl2 --stereo_tri_model {tri_model_file} --output_directory {output_directory} --gammaness_cut {gammaness_cut} --edep_cuts={edep_cuts}"
            print(cmd)
            sbatch_file = tri_model.cluster_configuration.write_sbatch_script(f"dl2_plots_{tri_model.direction_model.model_nickname}", cmd, output_directory, use_gpu_cscs=False)
            if self.cluster_configuration.use_cluster:
                os.system(f"sbatch {sbatch_file}")
            else:
                os.system(cmd)