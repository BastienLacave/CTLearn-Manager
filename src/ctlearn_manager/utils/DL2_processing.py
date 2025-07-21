import os
import pickle

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from pyirf.statistics import li_ma_significance
from tqdm import tqdm

from ..tri_model import CTLearnTriModelManager
from ..tri_model_collection import TriModelCollection
from ..utils.utils import (
    Cuts,
    CutType,
    DefaultCuts,
    calc_flux_for_N_sigma,
    find_68_percent_range,
    ExportCurves,
    CurveType,
    calc_flux_for_N_sigma_array,
    get_avg_pointing,
    ParticleType,
    get_color,
)
import h5py


class DL2DataProcessor:
    """
    A class to process DL2 data and perform various analyses such as plotting theta^2 distributions and computing on-off counts.

    Attributes
    ----------
    DL2_files : list
        List of DL2 file paths to be processed.
    CTLearnTriModelManager : CTLearnTriModelManager
        An instance of CTLearnTriModelManager containing telescope information.
    source_position : SkyCoord
        The sky coordinates of the source position. Default is the Crab Nebula.
    telescope_ids : list
        List of telescope IDs from CTLearnTriModelManager.
    telescope_names : list
        List of telescope names from CTLearnTriModelManager.
    stereo : bool
        Indicates if stereo mode is used.
    gammaness_cut : float
        The gammaness cut value for event selection. Default is 0.9.
    reconstruction_method : str
        The method used for reconstruction. Default is "CTLearn".
    reco_field_suffix : str
        Suffix for the reconstruction field, based on stereo mode.
    telescope_location : EarthLocation
        The location of the telescope, if LST1 is in the telescope names.
    reco_directions : list
        List of reconstructed sky directions.
    pointings : list
        List of pointing directions.
    dl2s : list
        List of loaded DL2 data.
    dl2s_cuts : list
        List of DL2 data after applying cuts.

    Methods
    -------
    __init__(self, DL2_files, CTLearnTriModelManager, gammaness_cut=0.9, source_position=SkyCoord.from_name("Crab")):
        Initializes the DL2DataProcessor with the given parameters and processes the DL2 data.
    process_DL2_data(self):
        Processes the DL2 data files, applying cuts and computing sky positions.
    plot_theta2_distribution(self, bins, n_off=3):
        Plots the theta^2 distribution for the processed DL2 data.
    compute_off_regions(self, pointing, n_off):
        Computes the off-source regions for background estimation.
    compute_eff_time(self, events):
        Computes the effective observation time and elapsed time from the event data.
    compute_on_off_counts(self, events, reco_coord, pointing_coord, n_off, theta2_cut=0.04*u.deg**2, gcut=0.5, E_min=0, E_max=100, I_min=None, I_max=None):
        Computes the on-source and off-source counts, as well as the Li & Ma significance.
    """

    def __init__(
        self,
        DL2_files: list[str],
        CTLearn_TriModel_Manager: CTLearnTriModelManager or TriModelCollection,
        cuts: list[Cuts] = [DefaultCuts.GH_0_9.value],
        source_position=SkyCoord.from_name("Crab"),
        pointing_table="dl1/monitoring/telescope/pointing/tel_001",
        default_E_bins=np.logspace(
            np.log10(0.02), np.log10(20), int((np.log10(20) - np.log10(0.02)) * 5 + 1)
        )
        * u.TeV,
    ):
        self.DL2_files = np.sort(DL2_files)
        if isinstance(CTLearn_TriModel_Manager, CTLearnTriModelManager):
            self.CTLearnTriModelCollection = TriModelCollection(
                [CTLearn_TriModel_Manager],
                cluster_configuration=CTLearn_TriModel_Manager.cluster_configuration,
                allow_muliple_projects=False,
            )
        else:
            assert CTLearn_TriModel_Manager.allow_muliple_projects == False, "CTLearnTriModelManager must be a single project."
            self.CTLearnTriModelCollection = CTLearn_TriModel_Manager
        self.source_position = source_position
        self.telscope_names = self.CTLearnTriModelCollection.tri_models[
            0
        ].telescope_names
        self.stereo = self.CTLearnTriModelCollection.tri_models[0].stereo
        self.cuts = cuts
        # self.gammaness_cut = gammaness_cut
        self.pointing_table = pointing_table
        self.reconstruction_method = "CTLearn"
        self.reco_field_suffix = (
            self.reconstruction_method
            if self.stereo
            else f"{self.reconstruction_method}_tel"
        )
        self.telescope_id = (
            self.CTLearnTriModelCollection.tri_models[0].telescope_ids
            if self.stereo
            else self.CTLearnTriModelCollection.tri_models[0].telescope_ids[0]
        )  # FIXME other telescopes ?
        # self.irfs = CTLearnTriModelManager.irfs
        self.CTLearn = True
        # self.edep_cuts = edep_cuts
        # print(self.edep_cuts)
        
        self.set_keys()

        if any("LST" in name and "1" in name for name in self.telscope_names):
            # print("LST1 is in the telescope names")
            self.telescope_location = EarthLocation(
                lon=-17.89149701 * u.deg,
                lat=28.76152611 * u.deg,
                # height of central pin + distance from pin to elevation axis
                height=2184 * u.m + 15.883 * u.m,
            )
        
        if self.CTLearn:
            self.dl2_processed_dir = self.CTLearnTriModelCollection.tri_models[0].project_directories.dl2_post_processed_data_directory
        else:
            self.dl2_processed_dir = self.CTLearnTriModelCollection.tri_models[0].project_directories.dl2_post_processed_data_rf_directory

        if not os.path.exists(self.dl2_processed_dir):
            os.makedirs(self.dl2_processed_dir, exist_ok=True)

        self.process_DL2_data()
        self.load_processed_data()

        import concurrent.futures

        self.cut_file_theta_cuts = []
        self.cut_file_gammaness_cuts = []

        def extract_cuts(args):
            model, file, cut = args
            zenith, azimuth = model.project_directories.get_available_MC_directions(ParticleType.GAMMA_POINT)
            file_zenith, file_azimuth = get_avg_pointing(
                file,
                self.pointing_table,
                alt_key=self.pointing_alt_key,
                az_key=self.pointing_az_key,
            )
            zenith = np.asarray(zenith)
            azimuth = np.asarray(azimuth)
            # Compute angular distance between file pointing and all available MC directions
            distances = np.sqrt((zenith - file_zenith.value) ** 2 + (azimuth - file_azimuth.value) ** 2)
            closest_idx = np.argmin(distances)
            cuts_file = model.project_directories.get_irf_files(
                zenith[closest_idx] * u.deg, azimuth[closest_idx] * u.deg, cut
            )['cuts_file']
            with fits.open(cuts_file, mode="readonly") as hdul:
                theta_cut = hdul["RAD_MAX"].data["cut"]
                gammaness_cut = hdul["GH_CUTS"].data["cut"]
            return theta_cut, gammaness_cut

        for i, cut in enumerate(self.cuts):
            if (
                cut.cut_type == CutType.EFFICIENCY_OPTIMIZED
                or cut.cut_type == CutType.SENSITIVITY_OPTIMIZED
            ):
                file_args = [(model, file, cut) for model, file in zip(self.corresponding_models, self.DL2_files)]
                file_theta_cuts = []
                file_gammaness_cuts = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = list(tqdm(executor.map(extract_cuts, file_args), desc=f"Extracting cuts {cut.get_label()}", total=len(file_args)))
                for theta_cut, gammaness_cut in results:
                    file_theta_cuts.append(theta_cut)
                    file_gammaness_cuts.append(gammaness_cut)
                self.cut_file_theta_cuts.append(file_theta_cuts)
                self.cut_file_gammaness_cuts.append(file_gammaness_cuts)


        # self.cut_file_theta_cuts = []
        # self.cut_file_gammaness_cuts = []
        # for i, cut in enumerate(self.cuts):
        #     if (
        #         cut.cut_type == CutType.EFFICIENCY_OPTIMIZED
        #         or cut.cut_type == CutType.SENSITIVITY_OPTIMIZED
        #     ):
        #         file_theta_cuts = []
        #         file_gammaness_cuts = []
        #         for model, file in zip(self.corresponding_models, self.DL2_files):
        #             zenith, azimuth = model.project_directories.get_available_MC_directions(ParticleType.GAMMA_POINT)
        #             file_zenith, file_azimuth = get_avg_pointing(file, 
        #                 self.pointing_table, 
        #                 alt_key=self.pointing_alt_key, 
        #                 az_key=self.pointing_az_key,
        #             )

        #             # Find the closest available MC direction (zenith, azimuth) to the file's average pointing
        #             zenith = np.asarray(zenith)
        #             azimuth = np.asarray(azimuth)
        #             # Compute angular distance between file pointing and all available MC directions
        #             distances = np.sqrt((zenith - file_zenith.value) ** 2 + (azimuth - file_azimuth.value) ** 2)
        #             closest_idx = np.argmin(distances)
        #             cuts_file = model.project_directories.get_irf_files(zenith[closest_idx] * u.deg, azimuth[closest_idx] * u.deg, cut)['cuts_file']
        #             with fits.open(cuts_file, mode="readonly") as hdul:
        #                 file_theta_cuts.append(hdul["RAD_MAX"].data["cut"])
        #                 file_gammaness_cuts.append(hdul["GH_CUTS"].data["cut"])
        #         self.cut_file_theta_cuts.append(file_theta_cuts)
        #         self.cut_file_gammaness_cuts.append(file_gammaness_cuts)

        # print(self.cut_file_theta_cuts)
        # print("Shape of self.file_theta_cuts:", np.shape(self.cut_file_theta_cuts))


        E_bins_tot = np.empty(len(self.cuts), dtype=object)
        # GH_cuts_tot = np.empty(len(self.cuts), dtype=object)
        # Theta_cuts_tot = np.empty(len(self.cuts), dtype=object)
        for i, cut in enumerate(self.cuts):
            if (
                cut.cut_type == CutType.EFFICIENCY_OPTIMIZED
                or cut.cut_type == CutType.SENSITIVITY_OPTIMIZED
            ):
                
                # for modelindex, file in zip(self.corresponding_model_indexs, self.DL2_files):
                # print(self.edep_cuts)
                # get E bins from IRFs cuts file
                zenith, azimuth = self.CTLearnTriModelCollection.tri_models[
                    0 # FIXME
                ].project_directories.get_available_MC_directions(ParticleType.GAMMA_POINT)

                cuts_file = self.CTLearnTriModelCollection.tri_models[
                    0 # FIXME
                ].direction_model.project_directories.get_irf_files(zenith[0], azimuth[0], cut)['cuts_file']
                with fits.open(cuts_file, mode="readonly") as hdul:
                    E_bins = hdul["GH_CUTS"].data["low"]
                    E_bins = np.append(E_bins, hdul["GH_CUTS"].data["high"][-1]) * u.TeV
                    E_bins_tot[i] = E_bins

                    # GH_cuts = hdul["GH_CUTS"].data["cut"]
                    # GH_cuts_tot[i] = GH_cuts

                    # Theta_cuts = hdul["RAD_MAX"].data["cut"]
                    # Theta_cuts_tot[i] = Theta_cuts
            else:
                E_bins = default_E_bins
                E_bins_tot[i] = E_bins
        self.E_bins = E_bins_tot
        # self.GH_cuts = GH_cuts_tot
        # self.Theta_cuts = Theta_cuts_tot
        
        # set_mpl_style()

    def set_keys(self):
        self.gammaness_key = (
            f"{self.reco_field_suffix}_prediction"  # if self.CTLearn else "gammaness"
        )
        self.energy_key = (
            f"{self.reco_field_suffix}_energy"  # if self.CTLearn else "reco_energy"
        )
        self.intensity_key = "hillas_intensity"  # if self.CTLearn else "intensity"
        self.reco_alt_key = (
            f"{self.reco_field_suffix}_alt"  # if self.CTLearn else "reco_alt"
        )
        self.reco_az_key = (
            f"{self.reco_field_suffix}_az"  # if self.CTLearn else "reco_az"
        )
        self.pointing_alt_key = "altitude"  # if self.CTLearn else "alt_tel"
        self.pointing_az_key = "azimuth"  # if self.CTLearn else "az_tel"
        self.time_key = "time"  # if self.CTLearn else "dragon_time"

    def process_DL2_data(self):
        import concurrent.futures

        def process_one(DL2_file):
            dl2_processed_dir = self.dl2_processed_dir
            dl2_output_file = os.path.join(dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_dl2_processed.pkl'))
            reco_output_file = os.path.join(dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_reco_directions.pkl'))
            pointing_output_file = os.path.join(dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_pointings.pkl'))
            I_g_on_counts_output_file = os.path.join(dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_I_g_on_counts.pkl'))
            I_g_off_counts_output_file = os.path.join(dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_I_g_off_counts.pkl'))

            # Skip if all outputs exist
            if all(os.path.exists(f) for f in [reco_output_file, pointing_output_file, dl2_output_file, I_g_on_counts_output_file, I_g_off_counts_output_file]):
                return DL2_file, True

            # Cluster or local processing
            try:
                if self.CTLearnTriModelCollection.cluster_configuration.use_cluster:
                    processor_file = f"{dl2_processed_dir}/{os.path.basename(DL2_file)}_processor.pkl"
                    with open(processor_file, "wb") as f:
                        pickle.dump(self, f)
                    self.CTLearnTriModelCollection.cluster_configuration.write_sbatch_script(
                        f"process_dl2_{os.path.basename(DL2_file)}",
                        f"process_dl2_file {DL2_file} {processor_file}",
                        dl2_processed_dir,
                        use_gpu_cscs=False,
                    )
                    os.system(f"sbatch {dl2_processed_dir}/process_dl2_{os.path.basename(DL2_file)}.sh")
                else:
                    processor_file = f"{dl2_processed_dir}/{os.path.basename(DL2_file)}_processor.pkl"
                    with open(processor_file, "wb") as f:
                        pickle.dump(self, f)
                    print(f"process_dl2_file {DL2_file} {processor_file}")
                    os.system(f"process_dl2_file {DL2_file} {processor_file}")
                return DL2_file, True
            except Exception as e:
                print(f"Error processing {DL2_file}: {e}")
                return DL2_file, False

        # Parallel processing of all files
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_one, self.DL2_files))

        # Optionally, you can check which files failed
        failed = [DL2_file for DL2_file, success in results if not success]
        if failed:
            print(f"Failed to process files: {failed}")
                    
    def load_processed_data(self):
        from tqdm import tqdm
        import concurrent.futures

        n_files = len(self.DL2_files)
        self.reco_directions = np.empty(n_files, dtype=object)
        self.pointings = np.empty(n_files, dtype=object)
        self.dl2s = np.empty(n_files, dtype=object)
        self.cuts_masks = np.empty(n_files, dtype=object)
        self.cuts_masks_gammaness_only = np.empty(n_files, dtype=object)
        self.I_g_on_counts = np.empty(n_files, dtype=object)
        self.I_g_off_counts = np.empty(n_files, dtype=object)
        self.corresponding_models = np.empty(n_files, dtype=CTLearnTriModelManager)
        self.corresponding_model_indexs = np.empty(n_files, dtype=int)
        failed_mask = np.ones(n_files, dtype=bool)

        def load_one(i_DL2_file):
            i, DL2_file = i_DL2_file
            try:
                # Find corresponding model
                if self.CTLearn:
                    corresponding_model = self.CTLearnTriModelCollection.find_closest_model_to(
                        DL2_file,
                        self.pointing_table,
                        alt_key=self.pointing_alt_key,
                        az_key=self.pointing_az_key,
                        verbose=False,
                    )
                    if corresponding_model is None:
                        return i, False
                else:
                    corresponding_model = self.CTLearnTriModelCollection.tri_models[0]

                self.corresponding_models[i] = corresponding_model

                # File paths
                dl2_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_dl2_processed.pkl'))
                reco_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_reco_directions.pkl'))
                pointing_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_pointings.pkl'))
                I_g_on_counts_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_I_g_on_counts.pkl'))
                I_g_off_counts_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_I_g_off_counts.pkl'))

                # Load files
                if not (os.path.exists(reco_output_file) and os.path.exists(pointing_output_file) and os.path.exists(dl2_output_file)):
                    return i, False

                with open(reco_output_file, "rb") as f:
                    transformed_reco_dict = pickle.load(f)
                with open(pointing_output_file, "rb") as f:
                    transformed_pointing_dict = pickle.load(f)
                transformed_reco = SkyCoord(
                    ra=transformed_reco_dict["ra"] * u.deg,
                    dec=transformed_reco_dict["dec"] * u.deg,
                    frame=self.source_position,
                )
                transformed_pointing = SkyCoord(
                    ra=transformed_pointing_dict["ra"] * u.deg,
                    dec=transformed_pointing_dict["dec"] * u.deg,
                    frame=self.source_position,
                )
                self.reco_directions[i] = transformed_reco
                self.pointings[i] = transformed_pointing

                with open(dl2_output_file, "rb") as f:
                    dl2 = pickle.load(f)
                if self.gammaness_key in dl2.colnames:
                    dl2 = dl2[dl2[self.gammaness_key] > 0]
                    cut_mask = np.empty(len(self.cuts), dtype=object)
                    cut_mask_gammaness_only = np.empty(len(self.cuts), dtype=object)
                    for j, cut in enumerate(self.cuts):
                        if cut.cut_type in [CutType.GLOBAL]:
                            mask = dl2[self.gammaness_key] > cut.gammaness_cut
                            cut_mask[j] = mask
                            cut_mask_gammaness_only[j] = mask
                        else:
                            mask = self.get_energy_dependent_mask_data(
                                dl2, corresponding_model, transformed_reco, cuts=cut
                            )
                            mask_gam = self.get_energy_dependent_mask_data(
                                dl2, corresponding_model, transformed_reco, False, cuts=cut
                            )
                            cut_mask[j] = mask
                            cut_mask_gammaness_only[j] = mask_gam
                else:
                    cut_mask = [np.ones(len(dl2), dtype=bool)]
                    cut_mask_gammaness_only = [np.ones(len(dl2), dtype=bool)]
                self.cuts_masks[i] = cut_mask
                self.cuts_masks_gammaness_only[i] = cut_mask_gammaness_only
                self.dl2s[i] = dl2

                # On-off counts
                if os.path.exists(I_g_on_counts_output_file) and os.path.exists(I_g_off_counts_output_file):
                    with open(I_g_on_counts_output_file, "rb") as f:
                        self.I_g_on_counts[i] = pickle.load(f)
                    with open(I_g_off_counts_output_file, "rb") as f:
                        self.I_g_off_counts[i] = pickle.load(f)
                else:
                    return i, False

                return i, True
            except Exception as e:
                print(f"Error loading {DL2_file}: {e}")
                return i, False

        # Parallel loading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(load_one, enumerate(self.DL2_files)), total=n_files, desc="Loading processed data"))

        for i, success in results:
            failed_mask[i] = success

        # Filter arrays
        self.reco_directions = self.reco_directions[failed_mask]
        self.pointings = self.pointings[failed_mask]
        self.dl2s = self.dl2s[failed_mask]
        self.cuts_masks = self.cuts_masks[failed_mask]
        self.cuts_masks_gammaness_only = self.cuts_masks_gammaness_only[failed_mask]
        self.I_g_on_counts = self.I_g_on_counts[failed_mask]
        self.I_g_off_counts = self.I_g_off_counts[failed_mask]
        self.corresponding_models = self.corresponding_models[failed_mask]
        self.DL2_files = self.DL2_files[failed_mask]

    def get_energy_dependent_mask_data(
        self,
        data,
        tri_model: CTLearnTriModelManager,
        reco_coord,
        theta_cut=True,
        cuts: Cuts = DefaultCuts.EFF_70.value,
    ):
        from astropy.io import fits

        zenith, azimuth = tri_model.project_directories.get_available_MC_directions(ParticleType.GAMMA_POINT)
        cuts_file = tri_model.project_directories.get_irf_files(zenith[0], azimuth[0], cuts)['cuts_file']

        with fits.open(cuts_file) as hdul:
            gammaness_cuts = hdul["GH_CUTS"].data["cut"]
            energy_low = hdul["GH_CUTS"].data["low"]
            energy_high = hdul["GH_CUTS"].data["high"]
            theta_cuts = hdul["RAD_MAX"].data["cut"]

            # Compute separation only once
            if "angular_separation" not in data.colnames:
                data["angular_separation"] = reco_coord.separation(self.source_position)

            energy = data[self.energy_key]
            gammaness = data[self.gammaness_key]
            separation = data["angular_separation"]

            # Vectorized mask creation
            mask = np.zeros(len(data), dtype=bool)
            for E_min, E_max, gcut, tcut in zip(energy_low, energy_high, gammaness_cuts, theta_cuts):
                e_mask = (energy > E_min) & (energy < E_max)
                g_mask = gammaness > gcut
                if theta_cut:
                    t_mask = separation < tcut
                    mask |= (e_mask & g_mask & t_mask)
                else:
                    mask |= (e_mask & g_mask)
            return mask

    def plot_theta2_distribution(self, bins=25, n_off=3, output_file=None, cuts_index=0):
        import matplotlib.pyplot as plt
        import concurrent.futures

        angle2_bins = np.linspace(0, 0.4, bins)
        angle2_center = (angle2_bins[:-1] + angle2_bins[1:]) / 2
        h_on = np.zeros(bins - 1)
        h_off = np.zeros(bins - 1)
        t_eff = 0 * u.h
        t_elapsed = 0 * u.h

        def process_file(args):
            reco_direction, pointing_direction, dl2, cuts_mask = args
            cuts_mask = cuts_mask[cuts_index]
            reco_direction = reco_direction[cuts_mask]
            pointing_direction = pointing_direction[cuts_mask]
            t_eff_temp, t_elapsed_temp = self.compute_eff_time(dl2)
            dl2 = dl2[cuts_mask]
            on_count_temp, off_count_temp, on_separation_temp, all_off_separation_temp, _ = self.compute_on_off_counts(
                dl2,
                reco_direction,
                pointing_direction,
                n_off=n_off,
                theta2_cut=0.04 * u.deg**2,
                gcut=0,
                E_min=0,
                E_max=1000,
                I_min=None,
                I_max=None,
            )
            h_on_temp, _ = np.histogram(on_separation_temp.to(u.deg).value ** 2, bins=angle2_bins)
            h_off_temp, _ = np.histogram(all_off_separation_temp.to(u.deg).value ** 2, bins=angle2_bins)
            return (on_count_temp, off_count_temp, h_on_temp, h_off_temp, t_eff_temp, t_elapsed_temp)

        file_args = list(zip(self.reco_directions, self.pointings, self.dl2s, self.cuts_masks_gammaness_only))
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for result in tqdm(executor.map(process_file, file_args), total=len(file_args), desc="Computing on-off counts"):
                results.append(result)

        on_count_tot = sum(r[0] for r in results)
        off_count_tot = sum(r[1] for r in results)
        for r in results:
            h_on += r[2]
            h_off += r[3] / n_off
            t_eff += r[4]
            t_elapsed += r[5]

        lima_signi = li_ma_significance(
            np.float64(on_count_tot), np.float64(off_count_tot), alpha=1 / n_off
        )
        fig, ax = plt.subplots()
        self.cuts[cuts_index].plot_cuts_info_plt(ax)
        label = (
            "$t_{eff}$ = "
            + f"{t_elapsed.to(u.h):.2f}"
            + "\n$N_{on}$ = "
            + f"{on_count_tot}\t"
            + r"$\overline{N}_{off}$ = "
            + f"{(off_count_tot / n_off):.1f}"
            + "\n$N_{excess}$ = "
            + f"{(on_count_tot - off_count_tot / n_off):.1f}\t"
            + r"$\sigma_{Li&Ma}$ = "
            + f"{lima_signi:.2f}"
        )
        props = dict(
            boxstyle="round",
            facecolor=get_color("surface"),
            alpha=0.2,
            edgecolor="none",
        )
        plt.text(
            0.12,
            0.96,
            label,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=props,
            color=get_color("on_surface"),
        )
        # plt.plot(angle2_center, h_off, label="off source", zorder=0, color=get_color("ctlearn_1"))
        # plt.errorbar(
        #     angle2_center,
        #     h_on,
        #     yerr=np.sqrt(h_on),
        #     label="On source",
        #     zorder=0,
        #     color=get_color("ctlearn_accent_1"),
        #     marker="o",
        #     ls="none",)
        # plt.errorbar(
        #     angle2_center,
        #     h_off,
        #     yerr=np.sqrt(h_off),
        #     label="Off source",
        #     zorder=0,
        #     color=get_color("ctlearn_1")
        #     , marker="o", ls="none")
        plt.scatter(
            angle2_center,
            h_on,
            label="On source",
            zorder=2,
            color=get_color("ctlearn_accent_1"),
            marker="o",
            s=20,
        )
        plt.scatter(
            angle2_center,
            h_off,
            label="Off source",
            zorder=2,
            color=get_color("ctlearn_1"),
            marker="o",
            s=20,
        )
        plt.fill_between(angle2_center, h_off - np.sqrt(h_off), h_off + np.sqrt(h_off), color=get_color("ctlearn_1"), alpha=0.3, zorder=1, edgecolor="none")
        plt.fill_between(angle2_center, h_on - np.sqrt(h_on), h_on + np.sqrt(h_on), color=get_color("ctlearn_accent_1"), alpha=0.3, zorder=1, edgecolor="none")
        # plt.plot(angle2_center, h_on, label="on source", color=get_color("ctlearn_accent_1"))
        plt.xlim(0, 0.4)
        plt.axvline(0.04, color=get_color('on_background'), linestyle="--")
        plt.legend()
        plt.xlabel(r"Separation [deg$^2$]")
        plt.ylabel("Counts")
        plt.title(f"{self.telscope_names[0]} Crab Nebula with {self.reconstruction_method}")
        if output_file is not None:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()

    def compute_off_regions(self, pointing, n_off):
        center = pointing  # SkyCoord(ra=10*u.degree, dec=20*u.degree)
        # ra_axis = pointing.directional_offset_by(0, 0.5*u.deg)
        # source = self.source_position #SkyCoord(ra=11*u.degree, dec=20*u.degree)
        angle_source = pointing.position_angle(self.source_position)
        radius = center.separation(self.source_position)
        angles = np.linspace(0, 2 * np.pi, n_off + 1, endpoint=False)  # + np.pi/(n_off)

        new_ra = []
        new_dec = []
        for i, angle in enumerate(angles):
            position_off = center.directional_offset_by(
                angle_source + Angle(angle, "rad"), radius
            )
            new_ra.append(position_off.ra.degree)
            new_dec.append(position_off.dec.degree)
        off_regions = SkyCoord(ra=new_ra[1:] * u.degree, dec=new_dec[1:] * u.degree)
        return off_regions
    
    def compute_eff_time(self, events):
        # Extract timestamps and delta_t as numpy arrays
        if self.CTLearn:
            timestamp = np.asarray(events[self.time_key].to_value("unix"))
        else:
            timestamp = np.asarray(events[self.time_key])

        delta_t = np.asarray(events["delta_t"])

        # Ensure units are attached only once
        if not isinstance(timestamp, u.Quantity):
            timestamp = timestamp * u.s
        if not isinstance(delta_t, u.Quantity):
            delta_t = delta_t * u.s

        # Fast diff and mask for elapsed time (DAQ breaks >0.01s)
        time_diff = np.diff(timestamp)
        t_elapsed = np.sum(time_diff[time_diff < 0.01 * u.s])

        # Mask for valid delta_t (exclude first event and DAQ breaks)
        valid_delta = (delta_t > 0.0 * u.s) & (delta_t < 0.01 * u.s)
        delta_t_valid = delta_t[valid_delta]

        if delta_t_valid.size == 0:
            # Avoid division by zero if no valid delta_t
            return 0 * u.h, 0 * u.h

        dead_time = np.min(delta_t_valid)
        mean_delta = np.mean(delta_t_valid)
        # Avoid division by zero in rate calculation
        denom = mean_delta - dead_time
        if denom <= 0:
            rate = 0
        else:
            rate = 1 / denom

        t_eff = t_elapsed / (1 + rate * dead_time)
        return t_eff.to(u.h), t_elapsed.to(u.h)

    def compute_on_off_counts_array(
        self,
        events,
        reco_coord,
        pointing_coord,
        n_off,
        theta2_cut=0.04 * u.deg**2,
        gcut=0.5,
        E_min=0,
        E_max=100,
    ):
        theta2_cut = np.atleast_1d(theta2_cut)
        gcut = np.atleast_1d(gcut)
        n_theta = len(theta2_cut)
        n_g = len(gcut)

        # --- Mask in energy first ---
        if hasattr(E_min, "value"):
            E_min = E_min.value
        if hasattr(E_max, "value"):
            E_max = E_max.value
        energy_mask = (events[self.energy_key] > E_min) & (events[self.energy_key] < E_max)
        if np.sum(energy_mask) == 0:
            # No events in this energy bin
            shape = (n_g, n_theta)
            return np.zeros(shape), np.zeros(shape), None, None, np.zeros(shape)

        # Mask all arrays
        events = events[energy_mask]
        print("Energy mask applied, remaining events:", len(events))
        gammaness = events[self.gammaness_key]
        reco_coord = reco_coord[energy_mask]
        pointing_coord = pointing_coord[energy_mask]

        # --- Compute ON separations ---
        on_sep = reco_coord.separation(self.source_position).to(u.deg).value  # (N_events,)

        # Prepare 3D mask for ON: (n_g, n_theta, N_events)
        gammaness_2d = gammaness[None, None, :]  # (1, 1, N_events)
        gcut_2d = gcut[:, None, None]            # (n_g, 1, 1)
        on_sep_2d = on_sep[None, None, :]        # (1, 1, N_events)
        t2_sqrt = np.sqrt(theta2_cut)[None, :, None]  # (1, n_theta, 1)

        g_mask = gammaness_2d > gcut_2d          # (n_g, 1, N_events)
        theta_mask = on_sep_2d < t2_sqrt         # (1, n_theta, N_events)
        on_mask = g_mask & theta_mask            # (n_g, n_theta, N_events)
        on_count = np.sum(on_mask, axis=2)       # (n_g, n_theta)

        # --- Compute OFF separations ---
        off_regions = self.compute_off_regions(pointing_coord, n_off)
        # Vectorized: stack all off regions and compute separations in one call
        off_sep = np.array([
            reco_coord.separation(off_regions[i]).to(u.deg).value for i in range(n_off)
        ])  # (n_off, N_events)

        # Prepare for broadcasting
        off_sep_3d = off_sep[None, None, :, :]  # (1, 1, n_off, N_events)
        t2_sqrt_3d = t2_sqrt[:, :, None]        # (1, n_theta, 1)
        g_mask_3d = g_mask[:, :, None, :]       # (n_g, 1, 1, N_events)

        # Mask for all off regions at once
        theta_mask_off = off_sep_3d < t2_sqrt_3d  # (1, n_theta, n_off, N_events)
        off_mask = g_mask_3d & theta_mask_off     # (n_g, n_theta, n_off, N_events)
        off_count = np.sum(off_mask, axis=(2, 3)) # (n_g, n_theta)

        alpha = 1 / n_off
        significance_lima = li_ma_significance(on_count, off_count, alpha)

        return on_count, off_count, None, None, significance_lima

    def compute_on_off_counts_array_old(
        self,
        events,
        reco_coord,
        pointing_coord,
        n_off,
        theta2_cut=0.04 * u.deg**2,
        gcut=0.5,
        E_min=0,
        E_max=100,
    ):
        theta2_cut = np.atleast_1d(theta2_cut)
        gcut = np.atleast_1d(gcut)
        n_theta = len(theta2_cut)
        n_g = len(gcut)

        # Precompute separations once
        on_separation_all = reco_coord.separation(self.source_position)
        off_regions = self.compute_off_regions(pointing_coord, n_off)
        off_separation_all = [reco_coord.separation(off_regions[i]) for i in range(n_off)]

        # Precompute energy mask once
        energy_mask = (events[self.energy_key] > E_min.value) & (events[self.energy_key] < E_max.value)
        gammaness = events[self.gammaness_key][energy_mask]
        on_sep = on_separation_all[energy_mask].value  # in deg
        # print("3D mask preparation...")
        # Prepare 3D mask for ON: (n_g, n_theta, N_events)
        gammaness_2d = gammaness[None, None, :]  # shape (1, 1, N_events)
        gcut_2d = gcut[:, None, None]            # shape (n_g, 1, 1)
        on_sep_2d = on_sep[None, None, :]        # shape (1, 1, N_events)
        t2_sqrt = np.sqrt(theta2_cut)[None, :, None]  # shape (1, n_theta, 1)
        # print("3D mask preparation done.")
        g_mask = gammaness_2d > gcut_2d          # (n_g, 1, N_events)
        theta_mask = on_sep_2d < t2_sqrt         # (1, n_theta, N_events)
        on_mask = g_mask & theta_mask            # (n_g, n_theta, N_events)
        on_count = np.sum(on_mask, axis=2)       # (n_g, n_theta)
        # print("ON regions: sum over all regions")
        # OFF regions: sum over all regions
        off_count = np.zeros((n_g, n_theta), dtype=int)
        for off_sep_all in off_separation_all:
            off_sep = off_sep_all[energy_mask].value
            off_sep_2d = off_sep[None, None, :]      # (1, 1, N_events)
            theta_mask_off = off_sep_2d < t2_sqrt    # (1, n_theta, N_events)
            off_mask = g_mask & theta_mask_off       # (n_g, n_theta, N_events)
            off_count += np.sum(off_mask, axis=2)    # (n_g, n_theta)
        # print("OFF regions: sum over all regions")
        alpha = 1 / n_off
        significance_lima = li_ma_significance(on_count, off_count, alpha)

        return on_count, off_count, None, None, significance_lima

    def compute_on_off_counts(
        self,
        events,
        reco_coord,
        pointing_coord,
        n_off,
        theta2_cut=0.04 * u.deg**2,
        gcut=0.5,
        E_min=0,
        E_max=100,
        I_min=None,
        I_max=None,
    ):
        if gcut is None:
            gcut = 0
        if I_min == None or I_max == None:
            mask = (
                (events[self.energy_key] > E_min)
                & (events[self.energy_key] < E_max)
                & (events[self.gammaness_key] > gcut)
            )  # TODO GCUT can be non in case of edep cut
        else:
            mask = (
                (events["hillas_intensity"] > I_min)
                & (events["hillas_intensity"] < I_max)
                & (events[self.gammaness_key] > gcut)
            )

        # ON
        on_separation = reco_coord.separation(self.source_position)[mask]
        # on_count = len(on_separation[on_separation < np.sqrt(theta2_cut)])
        on_count = np.count_nonzero(on_separation < np.sqrt(theta2_cut))
        # sum_norm_on = len(on_separation[(on_separation > norm_theta[0]) & (on_separation < norm_theta[1])])

        # # OFF
        # off_regions = self.compute_off_regions(pointing_coord, n_off)
        # off_count = 0
        # # sum_norm_off = 0
        # all_off_separation = []
        # for i in range(n_off):
        #     off_separation = reco_coord.separation(off_regions[i])[mask]
        #     all_off_separation.append(off_separation)
        #     off_count += len(off_separation[off_separation < np.sqrt(theta2_cut)])
        #     # sum_norm_off += len(off_separation[(off_separation > norm_theta[0]) & (off_separation < norm_theta[1])])
        # all_off_separation = np.array(all_off_separation).flatten() * u.deg

        # OFF (vectorized)
        off_regions = self.compute_off_regions(pointing_coord, n_off)
        # Stack all off regions and compute separations in one call
        off_separations = np.array([
            reco_coord.separation(off_region)[mask].value for off_region in off_regions
        ])  # shape (n_off, N_events)
        off_count = np.count_nonzero(off_separations < np.sqrt(theta2_cut.value))
        all_off_separation = off_separations.flatten() * u.deg

        # alpha = sum_norm_on / sum_norm_off
        alpha = 1 / n_off
        # stat = WStatCountsStatistic(n_on=on_count, n_off=off_count, alpha=alpha)
        # significance_lima = stat.sqrt_ts
        significance_lima = li_ma_significance(on_count, off_count, alpha)
        # print(f"Significance: {significance_lima:.2f}")
        # N_excess = on_count - alpha*off_count

        return on_count, off_count, on_separation, all_off_separation, significance_lima

    def plot_skymap(self, output_file=None, cuts_index=0):
        import matplotlib.pyplot as plt
        import concurrent.futures
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.xlabel("RA (deg)")
        plt.ylabel("Dec (deg)")
        if len(self.DL2_files) == 1:
            plt.title(f"Sky Map for {self.DL2_files[0].split('/')[-1]}")
        else:
            plt.title("Sky Map")

        # Prepare arguments for parallel processing
        file_args = list(
            zip(
                self.reco_directions,
                self.cuts_masks_gammaness_only,
                self.dl2s,
                self.pointings,
            )
        )

        def extract_coords(args):
            reco, cuts_mask, dl2, pointing = args
            cuts_mask = cuts_mask[cuts_index]
            ra = reco[cuts_mask].ra.deg
            dec = reco[cuts_mask].dec.deg
            pointing_ra = pointing[cuts_mask].ra.deg
            pointing_dec = pointing[cuts_mask].dec.deg
            return ra, dec, pointing_ra, pointing_dec

        # Parallel extraction of coordinates
        ra_values = []
        dec_values = []
        pointings_ra = []
        pointings_dec = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for ra, dec, pra, pdec in executor.map(extract_coords, file_args):
                ra_values.append(ra)
                dec_values.append(dec)
                pointings_ra.append(pra)
                pointings_dec.append(pdec)

        # Flatten arrays for plotting
        ra_values = np.concatenate(ra_values)
        dec_values = np.concatenate(dec_values)
        pointings_ra = np.concatenate(pointings_ra)
        pointings_dec = np.concatenate(pointings_dec)

        self.cuts[cuts_index].plot_cuts_info_plt(
            ax,
            text_color=get_color("ctlearn_highlight"),
            background_color=get_color("ctlearn_1"),
        )
        plt.hist2d(ra_values, dec_values, bins=100, cmap="viridis", zorder=0)
        # Add colorbar with same height as the plot
        im = plt.gca().collections[0]
        # Make colorbar the same height as the axes (not the whole figure)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, aspect=10)
        cbar.set_label("Counts")

        # Plot pointings and off regions (not parallelized, usually fast)
        for pointing, cuts_mask in zip(self.pointings, self.cuts_masks):
            cuts_mask = cuts_mask[cuts_index]
            pointing = pointing[cuts_mask]
            if len(pointing) == 0:
                continue
            off_regions = self.compute_off_regions(pointing[0], n_off=3)
            plt.scatter(
                pointing.ra.deg,
                pointing.dec.deg,
                label="pointing",
                color=get_color("ctlearn_accent_1"),
                marker="x",
            )
            for off_region in off_regions:
                off_circle = plt.Circle(
                    (off_region.ra.deg, off_region.dec.deg),
                    radius=0.2,
                    color="w",
                    fill=False,
                    lw=1,
                    ls="--",
                    alpha=0.9,
                )
                plt.gca().add_artist(off_circle)

        on_circle = plt.Circle(
            (self.source_position.ra.deg, self.source_position.dec.deg),
            radius=0.2,
            color="w",
            fill=False,
            lw=1,
        )
        plt.gca().add_artist(on_circle)
        plt.gca().set_aspect("equal", adjustable="box")

        if output_file is not None:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
    
    def optimize_cuts_on_crab(self, n_off=3, output_suffix=""):
        """
        Compute and store optimal gammaness/theta2 cuts for even and odd events for each energy bin.
        """
        import concurrent.futures
        from astropy.coordinates import concatenate

        E_bins = self.E_bins[0]
        gammaness_bins = np.linspace(0, 1, 201) #2001
        theta2bins = np.linspace(0, 0.6, 601) # 6001
        n_bins = len(E_bins) - 1

        # Prepare storage
        best_gammaness_even = np.zeros(n_bins)
        best_theta2_even = np.zeros(n_bins)
        best_gammaness_odd = np.zeros(n_bins)
        best_theta2_odd = np.zeros(n_bins)

        def mask_events(args):
            reco_direction, pointing_direction, dl2 = args
            even_mask = dl2["event_id"] % 2 == 0
            odd_mask = dl2["event_id"] % 2 == 1
            return (
                dl2[even_mask], dl2[odd_mask],
                reco_direction[even_mask], reco_direction[odd_mask],
                pointing_direction[even_mask], pointing_direction[odd_mask]
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(mask_events, zip(self.reco_directions, self.pointings, self.dl2s)), desc="Masking events", total=len(self.reco_directions)))

        even_dl2_temp, odd_dl2_temp, even_reco_temp, odd_reco_temp, even_pointing_temp, odd_pointing_temp = zip(*results)
        # print(even_reco_temp)
        print("Concatenating even_dl2...", flush=True)
        even_dl2 = np.concatenate(even_dl2_temp)
        print(f"# of even events: {len(even_dl2)}", flush=True)
        print("Concatenating odd_dl2...", flush=True)
        odd_dl2 = np.concatenate(odd_dl2_temp)
        print(f"# of odd events: {len(odd_dl2)}", flush=True)
        
        # even_reco = np.concatenate(list(tqdm(even_reco, desc="even_reco")))
        # print("Concatenating odd_reco...")
        # odd_reco = np.concatenate(list(tqdm(odd_reco, desc="odd_reco")))
        # print("Concatenating even_pointing...")
        # even_pointing = np.concatenate(list(tqdm(even_pointing, desc="even_pointing")))
        # print("Concatenating odd_pointing...")
        # odd_pointing = np.concatenate(list(tqdm(odd_pointing, desc="odd_pointing")))
        print(even_reco_temp)
        if len(even_reco_temp) > 1:
            print("Concatenating even_reco...", flush=True)
            even_reco = concatenate(even_reco_temp)
            print("Concatenating odd_reco...", flush=True)
            odd_reco = concatenate(odd_reco_temp)
            print("Concatenating even_pointing...", flush=True)
            even_pointing = concatenate(even_pointing_temp)
            print("Concatenating odd_pointing...", flush=True)
            odd_pointing = concatenate(odd_pointing_temp)
        else:
            even_reco = even_reco_temp[0]
            odd_reco = odd_reco_temp[0]
            even_pointing = even_pointing_temp[0]
            odd_pointing = odd_pointing_temp[0]

        # even_reco = SkyCoord([coord for coords in even_reco for coord in coords])
        # odd_reco = SkyCoord([coord for coords in odd_reco for coord in coords])
        # even_pointing = SkyCoord([coord for coords in even_pointing for coord in coords])
        # odd_pointing = SkyCoord([coord for coords in odd_pointing for coord in coords])

        def process_bin(args):
            i, E_min, E_max = args
            # Compute on/off counts for all grid points (vectorized)
            # print(even_reco)
            print(f"Processing bin {i} : [{E_min}, {E_max}]", flush=True)
            # EVEN
            _on, _off, _, _, _ = self.compute_on_off_counts_array(
                even_dl2, even_reco, even_pointing, n_off,
                theta2_cut=theta2bins, gcut=gammaness_bins, E_min=E_min, E_max=E_max
            )
            _nexcess = _on - _off / n_off
            flux_even, _ = calc_flux_for_N_sigma_array(
                5, _nexcess, _off, 3, 0.002, 10, 1, 50.0 * u.h, 50.0 * u.h, cond=True
            )

            # ODD
            _on, _off, _, _, _ = self.compute_on_off_counts_array(
                odd_dl2, odd_reco, odd_pointing, n_off,
                theta2_cut=theta2bins, gcut=gammaness_bins, E_min=E_min, E_max=E_max
            )
            _nexcess = _on - _off / n_off
            flux_odd, _ = calc_flux_for_N_sigma_array(
                5, _nexcess, _off, 3, 0.002, 10, 1, 50.0 * u.h, 50.0 * u.h, cond=True
            )

            # Find minimum
            min_idx_even = np.unravel_index(np.nanargmin(flux_even), flux_even.shape)
            min_idx_odd = np.unravel_index(np.nanargmin(flux_odd), flux_odd.shape)
            # print(f"Min flux even: {flux_even[min_idx_even]:.2e} at gammaness={gammaness_bins[min_idx_even[0]]:.2f}, theta2={theta2bins[min_idx_even[1]]:.2f}")
            # print(f"Min flux odd: {flux_odd[min_idx_odd]:.2e} at gammaness={gammaness_bins[min_idx_odd[0]]:.2f}, theta2={theta2bins[min_idx_odd[1]]:.2f}")
            return (
                i,
                gammaness_bins[min_idx_even[0]], theta2bins[min_idx_even[1]],
                gammaness_bins[min_idx_odd[0]], theta2bins[min_idx_odd[1]]
            )

        # Prepare arguments for parallel processing
        bin_args = [(i, E_min, E_max) for i, (E_min, E_max) in enumerate(zip(E_bins[:-1], E_bins[1:]))]
        bin_args.reverse()
        # print(bin_args)
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     results = list(tqdm(executor.map(process_bin, bin_args), total=n_bins, desc="Finding optimal cuts", unit="bins"))
        results = []
        for bin_arg in tqdm(bin_args, desc="Finding optimal cuts", total=n_bins, unit="bins"):
            i, g_even, t_even, g_odd, t_odd = process_bin(bin_arg)
            print(f"Gcuts {g_even:.2f}\t{g_odd:.2f}\tTheta2 {t_even:.2f}\t{t_odd:.2f}")
            results.append((i, g_even, t_even, g_odd, t_odd))

        # Store results
        for j, g_even, t_even, g_odd, t_odd in results:
            # print(g_even, g_odd, t_even, t_odd)
            best_gammaness_even[j] = g_even
            best_theta2_even[j] = t_even
            best_gammaness_odd[j] = g_odd
            best_theta2_odd[j] = t_odd
        # print(best_gammaness_even)
        # print(best_gammaness_odd)
        # Save to disk
        # Save to HDF5 file
        output_file = f"{self.CTLearnTriModelCollection.project_directories.dl2_post_processed_data_directory}/crab_optimized_cuts_{len(self.DL2_files)}_files{output_suffix}.h5"
        with h5py.File(output_file, "w") as f:
            f.create_dataset("even/gammaness", data=best_gammaness_even)
            f.create_dataset("even/theta2", data=best_theta2_even)
            f.create_dataset("even/E_bins", data=E_bins)
            f.create_dataset("odd/gammaness", data=best_gammaness_odd)
            f.create_dataset("odd/theta2", data=best_theta2_odd)
            f.create_dataset("odd/E_bins", data=E_bins)


    @staticmethod
    def read_cuts_optimized_on_crab_from_h5(h5_filename):
        """
        Read optimal gammaness/theta2 cuts for even and odd events from an HDF5 file.
        Returns a dict with keys: 'even' and 'odd', each containing a dict with keys 'gammaness', 'theta2', 'E_bins'.
        """
        cuts = {}
        with h5py.File(h5_filename, "r") as f:
            cuts["even"] = {
            "gammaness": f["even/gammaness"][:],
            "theta2": f["even/theta2"][:],
            "E_bins": f["even/E_bins"][:],
            }
            cuts["odd"] = {
            "gammaness": f["odd/gammaness"][:],
            "theta2": f["odd/theta2"][:],
            "E_bins": f["odd/E_bins"][:],
            }
        return cuts
        
    def plot_cuts_optimized_on_crab(self, cuts_h5_file):
        import matplotlib.pyplot as plt
        cuts = self.read_cuts_optimized_on_crab_from_h5(cuts_h5_file)
        gcuts_even, tcuts_even, E_bins = cuts["even"]["gammaness"], cuts["even"]["theta2"], cuts["even"]["E_bins"]
        gcuts_odd, tcuts_odd = cuts["odd"]["gammaness"], cuts["odd"]["theta2"]
        E_center = (E_bins[:-1] + E_bins[1:]) / 2
        # for i in range(len(E_center)):
        #     print(f"Energy: {E_center[i]:.2f} TeV, {gcuts_even[i]:.2f},{gcuts_odd[i]:.2f}, Even Theta2 Cut: {tcuts_even[i]:.2f}, Odd Theta2 Cut: {tcuts_odd[i]:.2f}")
        plt.scatter(E_center, gcuts_even, label="Even Cuts")
        plt.scatter(E_center, gcuts_odd, label="Odd Cuts", marker='x')
        plt.legend()
        plt.xlabel("Energy [TeV]")
        plt.ylabel("Gammaness Cut")
        plt.xscale("log")
        plt.show()
        plt.scatter(E_center, np.sqrt(tcuts_even), label="Even Cuts")
        plt.scatter(E_center, np.sqrt(tcuts_odd), label="Odd Cuts", marker='x')
        plt.legend()
        plt.xlabel("Energy [TeV]")
        plt.ylabel("Theta Cut [deg]")
        plt.xscale("log")
        plt.show()

    def plot_sensitivity(self, n_off=3, ax=None, label="CTLearn", output_file=None, export_to_h5: str=None,
        import_from_h5: str = None,
        import_label: str = None,
        optimized_on_crab: bool = False):
        import matplotlib.pyplot as plt
        import concurrent.futures

        export_curves = ExportCurves(export_to_h5)
        if import_from_h5 is not None:
            import_curves = ExportCurves(import_from_h5, export_mode=False, import_label=import_label)
            for curve_type in import_curves.curve_types:
                if curve_type not in [CurveType.SENSITIVITY_DATA.value]:
                    raise ValueError(f"Imported curves are not of type GH-cuts or theta-cuts : {curve_type}")
        if ax is None:
            fig, ax = plt.subplots()
        if len(self.cuts) == 1:
            self.cuts[0].plot_cuts_info_plt(ax)

        if not optimized_on_crab:
            for i, cut in enumerate(self.cuts):
                E_bins = self.E_bins[i]
                match cut.cut_type:
                    case CutType.EFFICIENCY_OPTIMIZED | CutType.SENSITIVITY_OPTIMIZED:
                        # GH_cuts = self.GH_cuts[i]
                        Theta_cuts = self.cut_file_theta_cuts[i]
                    case _:
                        # GH_cuts = [cut.gammaness_cut] * len(E_bins)
                        if cut.theta_cut is None:
                            Theta_cuts = [[0.2] * len(E_bins)] * len(self.DL2_files)
                        else:
                            Theta_cuts = [[cut.theta_cut] * len(E_bins)] * len(self.DL2_files)
                on_count = np.zeros(len(E_bins) - 1)
                off_count = np.zeros(len(E_bins) - 1)
                t_eff = 0 * u.h
                t_elapsed = 0 * u.h

                def process_file(args):
                    reco_direction, pointing_direction, dl2, cuts_mask, theta_cuts = args
                    cuts_mask = cuts_mask[i]
                    reco_direction = reco_direction[cuts_mask]
                    pointing_direction = pointing_direction[cuts_mask]
                    t_eff_temp, t_elapsed_temp = self.compute_eff_time(dl2)
                    dl2 = dl2[cuts_mask]
                    on_count_arr = np.zeros(len(E_bins) - 1)
                    off_count_arr = np.zeros(len(E_bins) - 1)
                    for j, E_min, E_max, Theta_cut in zip(
                        range(len(E_bins) - 1), E_bins[:-1], E_bins[1:], theta_cuts
                    ):
                        on_count_temp, off_count_temp, _, _, _ = self.compute_on_off_counts(
                            dl2,
                            reco_direction,
                            pointing_direction,
                            n_off=n_off,
                            theta2_cut=(Theta_cut**2) * u.deg**2,
                            gcut=None,
                            E_min=E_min,
                            E_max=E_max,
                            I_min=None,
                            I_max=None,
                        )
                        on_count_arr[j] = on_count_temp
                        off_count_arr[j] = off_count_temp / n_off
                    return on_count_arr, off_count_arr, t_eff_temp, t_elapsed_temp

                file_args = list(zip(self.reco_directions, self.pointings, self.dl2s, self.cuts_masks_gammaness_only, Theta_cuts))
                results = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for result in tqdm(executor.map(process_file, file_args), total=len(file_args), desc=f"Computing sensitivity [{cut.get_label()}]"):
                        results.append(result)

                for r in results:
                    on_count += r[0]
                    off_count += r[1]
                    t_eff += r[2]
                    t_elapsed += r[3]

                nexcess = on_count - off_count

                min_signi = 3
                min_exc = 0.002
                min_off_events = 10
                backg_syst = 0.01
                obs_time = 50.0 * u.h

                flux_factor, lima_signi = calc_flux_for_N_sigma(
                    5,
                    nexcess,
                    off_count,
                    min_signi,
                    min_exc,
                    min_off_events,
                    1,
                    obs_time,
                    t_eff,
                    cond=True,
                )
                flux_minus, lima_signi_minus = calc_flux_for_N_sigma(
                    5,
                    nexcess + backg_syst * off_count + (nexcess + 2 * off_count) ** 0.5,
                    off_count,
                    min_signi,
                    min_exc,
                    min_off_events,
                    1,
                    obs_time,
                    t_eff,
                    cond=True,
                )
                flux_plus, lima_signi_plus = calc_flux_for_N_sigma(
                    5,
                    nexcess - backg_syst * off_count - (nexcess + 2 * off_count) ** 0.5,
                    off_count,
                    min_signi,
                    min_exc,
                    min_off_events,
                    1,
                    obs_time,
                    t_eff,
                    cond=True,
                )
                mask = np.ones(len(flux_factor), dtype=bool)
                E = (E_bins[:-1] + E_bins[1:]) / 2

                if len(self.cuts) > 1:
                    ax.plot(
                        E[mask],
                        flux_factor[mask] * 100,
                        marker="o",
                        label=cut.get_label(),
                        zorder=10,
                        ls="--",
                    )
                else:
                    ax.plot(
                        E[mask],
                        flux_factor[mask] * 100,
                        marker="o",
                        label=label,
                        zorder=10,
                        ls="--",
                    )
                ax.fill_between(
                    E[mask].value,
                    flux_minus[mask] * 100,
                    flux_plus[mask] * 100,
                    alpha=0.2,
                    zorder=0,
                    edgecolor="none"
                )
                export_curves.add_curve(
                    E[mask],
                    flux_factor[mask] * 100,
                    CurveType.SENSITIVITY_DATA,
                    cuts=cut,
                )

        # ...rest of the function unchanged (optimized_on_crab branch, plotting, export, etc.)...
        else:
            from concurrent.futures import ThreadPoolExecutor

            def process_energy_bin(args):
                j, E_min, E_max, even_dl2, reco_direction_even, pointing_direction_even, odd_dl2, reco_direction_odd, pointing_direction_odd, n_off, theta2bins, gammaness_bins = args
                # Even
                on_count_temp_even, off_count_temp_even, _, _, _ = self.compute_on_off_counts_array(
                    even_dl2,
                    reco_direction_even,
                    pointing_direction_even,
                    n_off,
                    theta2_cut=theta2bins,
                    gcut=gammaness_bins,
                    E_min=E_min,
                    E_max=E_max,
                )
                # Odd
                on_count_temp_odd, off_count_temp_odd, _, _, _ = self.compute_on_off_counts_array(
                    odd_dl2,
                    reco_direction_odd,
                    pointing_direction_odd,
                    n_off,
                    theta2_cut=theta2bins,
                    gcut=gammaness_bins,
                    E_min=E_min,
                    E_max=E_max,
                )
                return j, on_count_temp_even, off_count_temp_even, on_count_temp_odd, off_count_temp_odd
            # Split the cuts into two groups: even and odd, optimize cuts and apply on the other group
            E_bins = self.E_bins[0]
            gammaness_bins = np.linspace(0, 1, 2001)
            alphabins = np.linspace(0, 60, 3001)
            theta2bins = np.linspace(0, 0.6, 6001)
            # E_bins = np.linspace(-2.0, 2.2, 22) # 5 per decade

            optimization_matrix_size = (len(E_bins) - 1, len(gammaness_bins), len(theta2bins))
            on_count = [np.zeros(optimization_matrix_size), np.zeros(optimization_matrix_size)]
            off_count = [np.zeros(optimization_matrix_size), np.zeros(optimization_matrix_size)]
            t_eff = [0 * u.h, 0 * u.h]
            t_elapsed = [0 * u.h, 0 * u.h]

            for reco_direction, pointing_direction, dl2 in tqdm(
                    zip(self.reco_directions, self.pointings, self.dl2s),
                    desc=f"Computing sensitivity using optimized_on_crab",
                    total=len(self.reco_directions),
                ):
                even_mask = dl2["event_id"] % 2 == 0
                odd_mask = dl2["event_id"] % 2 == 1
                even_dl2 = dl2[even_mask]
                odd_dl2 = dl2[odd_mask]
                reco_direction_even = reco_direction[even_mask]
                reco_direction_odd = reco_direction[odd_mask]
                pointing_direction_even = pointing_direction[even_mask]
                pointing_direction_odd = pointing_direction[odd_mask]
                t_eff_temp_even, t_elapsed_temp_even = self.compute_eff_time(dl2)
                t_eff_temp_odd, t_elapsed_temp_odd = self.compute_eff_time(dl2)
                t_eff[0] += t_eff_temp_even
                t_eff[1] += t_eff_temp_odd
                t_elapsed[0] += t_elapsed_temp_even
                t_elapsed[1] += t_elapsed_temp_odd

                # Prepare arguments for parallel processing
                energy_args = [
                    (
                        j, E_min, E_max,
                        even_dl2, reco_direction_even, pointing_direction_even,
                        odd_dl2, reco_direction_odd, pointing_direction_odd,
                        n_off, theta2bins, gammaness_bins
                    )
                    for j, E_min, E_max in zip(range(len(E_bins) - 1), E_bins[:-1], E_bins[1:])
                ]

                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(process_energy_bin, energy_args))

                for j, on_count_temp_even, off_count_temp_even, on_count_temp_odd, off_count_temp_odd in results:
                    on_count[0][j] += on_count_temp_even
                    off_count[0][j] += off_count_temp_even / n_off
                    on_count[1][j] += on_count_temp_odd
                    off_count[1][j] += off_count_temp_odd / n_off


            nexcess = [on_count[0] - off_count[0], on_count[1] - off_count[1]]
            min_signi = 3  # below this value (significance of the test source, Crab, for the *actual* observation
            min_exc = 0.002  # in fraction of off. Below this we ignore the corresponding cut combination.
            min_off_events = 10 
            backg_syst = 0.01

            GH_cuts = [np.empty(len(E_bins) - 1), np.empty(len(E_bins) - 1)]
            Theta_cuts = GH_cuts
            for i in range(len(E_bins) - 1):
                flux_factor_even, _ = calc_flux_for_N_sigma_array(
                    5,
                    nexcess[0],
                    off_count[0],
                    min_signi,
                    min_exc,
                    min_off_events,
                    1,
                    obs_time,
                    t_eff,
                    cond=True,
                )
                flux_factor_odd, _ = calc_flux_for_N_sigma_array(
                    5,
                    nexcess[1],
                    off_count[1],
                    min_signi,
                    min_exc,
                    min_off_events,
                    1,
                    obs_time,
                    t_eff,
                    cond=True,
                )

                

                min_flux_index_even = np.argmin(flux_factor_even)
                print(min_flux_index_even)
                min_flux_index_odd = np.argmin(flux_factor_odd)
                print(min_flux_index_odd)
        
                GH_cuts[0][i] = gammaness_bins[min_flux_index_even]
                GH_cuts[1][i] = gammaness_bins[min_flux_index_odd]
                
                Theta_cuts[0][i] = theta2bins[min_flux_index_even]
                Theta_cuts[1][i] = theta2bins[min_flux_index_odd]

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Reco Energy [TeV]")
        ax.set_ylabel("Differential sensitivity [% Obs. Flux.]")
        # ax.set_xlim(0.03, 2)
        # ax.set_ylim(2, 60)
        # ax.set_yticks([1, 10])
        # ax.set_yticklabels(['1', '10'])
        ax.set_title("Differential sensitivity")
        if export_to_h5 is not None:
            export_curves.export()
        if import_from_h5 is not None:
            import_curves.plot_curves(axs = [ax] * int(len(import_curves.x_values)))
        ax.legend()
        

        plt.tight_layout()
        if output_file is not None:
            plt.savefig(output_file)
            plt.close()
        else:
            if ax is None:
                plt.show()

    def plot_PSF(self, n_off=3, ax=None, label="CTLearn", output_file=None, plot_MC: list[str]=[], export_to_h5: str=None,
        import_from_h5: str = None,
        import_label: str = None, ylim=(0, 0.6)):
        import matplotlib.pyplot as plt
        import concurrent.futures

        export_curves = ExportCurves(export_to_h5)
        if import_from_h5 is not None:
            import_curves = ExportCurves(import_from_h5, export_mode=False, import_label=import_label)
            for curve_type in import_curves.curve_types:
                if curve_type not in [CurveType.PSF_DATA.value]:
                    raise ValueError(f"Imported curves are not of type PSF-data : {curve_type}")
        if ax is None:
            fig, ax = plt.subplots()
        if len(self.cuts) == 1:
            self.cuts[0].plot_cuts_info_plt(ax)

        for i, cut in enumerate(self.cuts):
            stored_efficiency_theta = cut.efficiency_theta
            cut.efficiency_theta = None
            E_bins = self.E_bins[i]
            match cut.cut_type:
                case CutType.EFFICIENCY_OPTIMIZED | CutType.SENSITIVITY_OPTIMIZED:
                    # GH_cuts = self.GH_cuts[i]
                    Theta_cuts = self.cut_file_theta_cuts[i]
                case _:
                    # GH_cuts = [cut.gammaness_cut] * len(E_bins)
                    if cut.theta_cut is None:
                        Theta_cuts = [[0.2] * len(E_bins)] * len(self.DL2_files)
                    else:
                        Theta_cuts = [[cut.theta_cut] * len(E_bins)] * len(self.DL2_files)
            angle_bins = np.linspace(0, 0.4, 25)
            h_on = np.zeros((len(E_bins) - 1, len(angle_bins) - 1))
            h_off = np.zeros((len(E_bins) - 1, len(angle_bins) - 1))
            t_eff = 0 * u.h
            t_elapsed = 0 * u.h

            def process_file(args):
                reco_direction, pointing_direction, dl2, cuts_mask, theta_cuts = args
                cuts_mask = cuts_mask[i]
                reco_direction = reco_direction[cuts_mask]
                pointing_direction = pointing_direction[cuts_mask]
                t_eff_temp, t_elapsed_temp = self.compute_eff_time(dl2)
                dl2 = dl2[cuts_mask]
                h_on_file = np.zeros((len(E_bins) - 1, len(angle_bins) - 1))
                h_off_file = np.zeros((len(E_bins) - 1, len(angle_bins) - 1))
                for j, E_min, E_max, Theta_cut in zip(
                    range(len(E_bins) - 1), E_bins[:-1], E_bins[1:], theta_cuts
                ):
                    on_count_temp, off_count_temp, on_separation_temp, all_off_separation_temp, _ = self.compute_on_off_counts(
                        dl2,
                        reco_direction,
                        pointing_direction,
                        n_off=n_off,
                        theta2_cut=(Theta_cut**2) * u.deg**2,
                        gcut=None,
                        E_min=E_min,
                        E_max=E_max,
                        I_min=None,
                        I_max=None,
                    )
                    h_on_temp, _ = np.histogram(on_separation_temp.to(u.deg).value ** 2, bins=angle_bins)
                    h_off_temp, _ = np.histogram(all_off_separation_temp.to(u.deg).value ** 2, bins=angle_bins)
                    h_on_file[j] += h_on_temp
                    h_off_file[j] += h_off_temp / n_off
                return h_on_file, h_off_file, t_eff_temp, t_elapsed_temp

            file_args = list(zip(self.reco_directions, self.pointings, self.dl2s, self.cuts_masks_gammaness_only, Theta_cuts))
            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for result in tqdm(executor.map(process_file, file_args), total=len(file_args), desc=f"Computing PSF [{cut.get_label()}]"):
                    results.append(result)

            for r in results:
                h_on += r[0]
                h_off += r[1]
                t_eff += r[2]
                t_elapsed += r[3]

            nexcess = h_on - h_off

            psf = np.zeros(len(E_bins) - 1)
            psf_min = np.zeros(len(E_bins) - 1)
            psf_max = np.zeros(len(E_bins) - 1)
            for k, E_min, E_max in zip(range(len(E_bins) - 1), E_bins[:-1], E_bins[1:]):
                psf[k] = find_68_percent_range(nexcess[k], angle_bins) ** 0.5
                psf_max[k] = (
                    find_68_percent_range(
                        nexcess[k]
                        + 0.01 * h_off[k]
                        + np.sqrt(nexcess[k] + 2 * h_off[k]),
                        angle_bins,
                    )
                    ** 0.5
                )
                psf_min[k] = (
                    find_68_percent_range(
                        nexcess[k]
                        - 0.01 * h_off[k]
                        - np.sqrt(nexcess[k] + 2 * h_off[k]),
                        angle_bins,
                    )
                    ** 0.5
                )

            E = (E_bins[:-1] + E_bins[1:]) / 2
            if len(self.cuts) > 1:
                ax.plot(
                    E.value, psf, marker="o", label=cut.get_label(), zorder=10, ls="--"
                )
            else:
                ax.plot(E.value, psf, marker="o", label=label, zorder=10, ls="--")
            ax.fill_between(
                E.value,
                psf - 1 / np.sqrt(np.sum(h_on, axis=1)),
                psf + 1 / np.sqrt(np.sum(h_on, axis=1)),
                alpha=0.3,
                zorder=0,
                color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i],
                edgecolor="none"
            )
            export_curves.add_curve(
                E.value,
                psf,
                CurveType.PSF_DATA,
                cuts=cut,
            )
            cut.efficiency_theta = stored_efficiency_theta


        # ...rest of the MC plotting and export code unchanged...
        for i, cut in enumerate(self.cuts):
            # stored_efficiency_theta = cut.efficiency_theta
            # cut.efficiency_theta = None
            for tri_model_nickname in tqdm(plot_MC, desc="Plotting MC curves"):
                    if tri_model_nickname in self.CTLearnTriModelCollection.tri_model_nicknames:
                        tri_model = self.CTLearnTriModelCollection.get_tri_model_by_nickname(tri_model_nickname)
                        coords = tri_model.get_available_MC_directions(verbose=False)
                        if len(coords) > 0:
                            for zenith, azimuth in coords:
                                try:
                                    e_bins, ang_res_err = tri_model.get_angular_resolution_DL2(
                                        zenith = zenith,
                                        azimuth = azimuth,
                                        cuts = cut,
                                    )
                                    e = (e_bins[:-1].value + e_bins[1:].value) / 2
                                    ang_res = [e_r[0].value for e_r in ang_res_err]
                                    ax.plot(e, ang_res, label=f"MC ({zenith.value:.1f}, {azimuth.value:.1f})° | {cut.get_label()}", zorder=10)
                                except:
                                    print(f"IRFs not found for {tri_model.project_directories.tri_model_nickname}: ({zenith.value:.1f}, {azimuth.value:.1f})°, {cut.get_label()}. Skipping.")
                    else:
                        print(f"Model {plot_MC} not found in CTLearnTriModelCollection. Skipping MC curves.")
            # cut.efficiency_theta = stored_efficiency_theta
        if export_to_h5 is not None:
            export_curves.export()
        if import_from_h5 is not None:
            import_curves.plot_curves(axs = [ax] * int(len(import_curves.x_values)))
        ax.legend()
        ax.set_xscale("log")
        ax.set_ylabel("68% cont. [deg]")
        ax.set_xlabel("Reco Energy [TeV]")
        ax.set_ylim(ylim)
        ax.set_title("Point Spread Function")

        plt.tight_layout()
        if output_file is not None:
            plt.savefig(output_file)
            plt.close()
        else:
            if ax is None:
                plt.show()

    def get_gammaness_cuts_for_efficiencies(
        self, MC_dl2, efficiencies, E_min=None, E_max=None, I_min=None, I_max=None
    ):
        gammaness_cuts = np.empty(len(efficiencies), dtype=float)
        for i, efficiency in enumerate(efficiencies):
            if E_min is not None and E_max is not None:
                mask = (MC_dl2[self.energy_key] > E_min) & (
                    MC_dl2[self.energy_key] < E_max
                )
            elif I_min is not None and I_max is not None:
                mask = (MC_dl2["hillas_intensity"] > I_min) & (
                    MC_dl2["hillas_intensity"] < I_max
                )
            else:
                mask = np.ones(len(MC_dl2), dtype=bool)

            sorted_gammaness = np.sort(MC_dl2[self.gammaness_key][mask])
            cut_index = int((1 - efficiency) * len(sorted_gammaness))
            gammaness_cut = sorted_gammaness[cut_index]
            gammaness_cuts[i] = gammaness_cut
        return gammaness_cuts

    def get_efficiency_for_gamaness_cuts(
        self, MC_dl2, gammaness_cuts, E_min=None, E_max=None, I_min=None, I_max=None
    ):
        efficiencies = np.empty(len(gammaness_cuts), dtype=float)
        for i, gammaness_cut in enumerate(gammaness_cuts):
            if E_min is not None and E_max is not None:
                mask = (MC_dl2[self.energy_key] > E_min) & (
                    MC_dl2[self.energy_key] < E_max
                )
            elif I_min is not None and I_max is not None:
                mask = (MC_dl2["hillas_intensity"] > I_min) & (
                    MC_dl2["hillas_intensity"] < I_max
                )
            else:
                mask = np.ones(len(MC_dl2), dtype=bool)

            mask &= MC_dl2[self.gammaness_key] > gammaness_cut
            efficiency = len(MC_dl2[mask]) / len(MC_dl2)
            efficiencies[i] = efficiency
        return efficiencies

    def plot_bkg_discrimination_capability(
        self, n_off=3, axs=None, label="CTLearn", output_file=None
    ):
        gammaness_cuts = np.arange(0, 1.05, 0.05)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LogFormatterExponent, LogLocator



        if axs is None:
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # , sharey=True)
        intensity_ranges = [(50, 200), (200, 800), (800, 3200), (3200, np.inf)]
        # for ax, (I_min, I_max) in zip(axs, intensity_ranges):
        #     excess_counts = []
        #     off_counts = []
        #     for gcut in tqdm(gammaness_cuts, desc=f"RComputing excesses for [{I_min} - {I_max}] p.e."):
        #         total_excess = 0
        #         total_off = 0
        #         for reco_direction, pointing_direction, dl2 in zip(self.reco_directions, self.pointings, self.dl2s):
        #             on_count, off_count, _, _, _ = self.compute_on_off_counts(
        #                 dl2,
        #                 reco_direction,
        #                 pointing_direction,
        #                 n_off=n_off,
        #                 theta2_cut=0.04 * u.deg ** 2,
        #                 gcut=gcut,
        #                 E_min=None,
        #                 E_max=None,
        #                 I_min=I_min,
        #                 I_max=I_max
        #             )
        #             total_excess += on_count - off_count / n_off
        #             total_off += off_count / n_off

        #         excess_counts.append(total_excess)
        #         off_counts.append(total_off)

        #     ax.plot(off_counts, excess_counts, marker='o', linestyle='-',)
        #     ax.set_xlabel('Background Counts')
        #     ax.set_title(f'[{I_min} - {I_max}] p.e.')
        # print(self.I_g_on_counts)
        I_g_on_counts_tot = np.sum(self.I_g_on_counts, axis=0)
        I_g_off_counts_tot = np.sum(self.I_g_off_counts, axis=0)
        

        for i, ax, (I_min, I_max) in zip(
            range(len(intensity_ranges)), axs, intensity_ranges
        ):
            I_g_on_counts_tot[i][I_g_on_counts_tot[i] == 0] = np.nan
            I_g_off_counts_tot[i][I_g_on_counts_tot[i] == 0] = np.nan
            ax.plot(
                I_g_off_counts_tot[i],
                I_g_on_counts_tot[i],
                marker="o",
                linestyle="-",
                label=label,
            )
            ax.set_xlabel("Background Counts")
            ax.set_title(f"[{I_min} - {I_max}] p.e.")
            # ax.set_xscale('log')
            # ax.set_xlim(left=0.1)
            # Plot statistical uncertainty for ON counts
            # ax.fill_between(
            #     I_g_off_counts_tot[i],
            #     I_g_on_counts_tot[i] - np.sqrt(I_g_on_counts_tot[i]),
            #     I_g_on_counts_tot[i] + np.sqrt(I_g_on_counts_tot[i]),
            #     alpha=0.3,
            # )

        axs[0].set_ylabel("Excess Counts")
        axs[0].legend()
        plt.suptitle(
            "Excess Counts vs Background Counts for Different Intensity Ranges"
        )
        axs[2].set_xscale("log")
        axs[3].set_xscale("log")
        # from matplotlib.ticker import LogFormatterExponent
        # # ax.yaxis.set_major_locator(LogLocator(base=10.0))
        # ax.yaxis.set_major_formatter(LogFormatterExponent(base=10.0))
        # for ax in axs:
        #     ax.set_xscale("log")
        #     ax.set_yscale("log")
        #     ax.xaxis.set_major_locator(LogLocator(base=10.0))
        #     ax.xaxis.set_major_formatter(LogFormatterExponent(base=10.0))
        #     ax.yaxis.set_major_locator(LogLocator(base=10.0))
        #     ax.yaxis.set_major_formatter(LogFormatterExponent(base=10.0))

        if output_file is not None:
            plt.savefig(output_file)
            plt.close()
        else:
            if axs is None:
                plt.show()

    def plot_excess_vs_background_rates(self, n_off=3, output_file=None):
        gammaness_cuts = np.arange(0, 1.05, 0.05)
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # , sharey=True)
        intensity_ranges = [(50, 200), (200, 800), (800, 3200), (3200, np.inf)]
        total_t_eff = 0 * u.h
        # for ax, (I_min, I_max) in zip(axs, intensity_ranges):
        #     excess_rates = []
        #     background_rates = []
        #     for gcut in gammaness_cuts:
        #         total_excess = 0
        #         total_off = 0
        #         total_t_eff = 0 * u.h
        #         for reco_direction, pointing_direction, dl2 in zip(self.reco_directions, self.pointings, self.dl2s):
        #             on_count, off_count, _, _, _ = self.compute_on_off_counts(
        #                 dl2,
        #                 reco_direction,
        #                 pointing_direction,
        #                 n_off=n_off,
        #                 theta2_cut=0.04 * u.deg ** 2,
        #                 gcut=gcut,
        #                 E_min=None,
        #                 E_max=None,
        #                 I_min=I_min,
        #                 I_max=I_max
        #             )
        for dl2 in self.dl2s:
            t_eff, _ = self.compute_eff_time(dl2)
            # total_excess += ((on_count - off_count / n_off) / t_eff.to(u.s)).value
            # total_off += (off_count / n_off / t_eff.to(u.s)).value
            total_t_eff += t_eff

            # excess_rates.append(total_excess)
            # background_rates.append(total_off)
            # print(excess_rates)
            # print(background_rates)

            # ax.plot(background_rates, excess_rates, marker='o', linestyle='-')
            # ax.set_xlabel('Background Rate [Hz]')
            # ax.set_title(f'[{I_min} - {I_max}] p.e.')

        I_g_on_counts_tot = np.sum(self.I_g_on_counts, axis=0)
        I_g_off_counts_tot = np.sum(self.I_g_off_counts, axis=0)

        for i, ax, (I_min, I_max) in zip(
            range(len(intensity_ranges)), axs, intensity_ranges
        ):
            ax.plot(
                I_g_off_counts_tot[i] / total_t_eff,
                I_g_on_counts_tot[i] / total_t_eff,
                marker="o",
                linestyle="-",
            )
            ax.set_xlabel("Background Counts")
            ax.set_title(f"[{I_min} - {I_max}] p.e.")
            ax.set_xscale("log")

        axs[0].set_ylabel("Excess Rate [Hz]")
        plt.suptitle("Excess Rate vs Background Rate for Different Intensity Ranges")
        if output_file is not None:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()

    def plot_excess_and_background_rates_vs_energy(
        self, n_off=3, output_file=None, cuts_index=0
    ):
        import matplotlib.pyplot as plt

        E_bins = self.E_bins[cuts_index]
        # if self.cuts[cuts_index].cut_type == CutType.EFFICIENCY_OPTIMIZED or self.cuts[cuts_index].cut_type == CutType.SENSITIVITY_OPTIMIZED:
        #     E_bins = self.E_bins[cuts_index]
        # else:
        #     E_bins = np.logspace(np.log10(0.03), np.log10(2), 10) * u.TeV
        fig, ax = plt.subplots()
        excess_rates = np.zeros(len(E_bins) - 1)
        background_rates = np.zeros(len(E_bins) - 1)
        t_eff = 0 * u.h
        self.cuts[cuts_index].plot_cuts_info_plt(ax)

        for reco_direction, pointing_direction, dl2 in tqdm(
            zip(self.reco_directions, self.pointings, self.dl2s),
            desc="Computing excess and background rates",
            total=len(self.reco_directions),
            disable=self.CTLearnTriModelCollection.cluster_configuration.use_cluster,
        ):
            for i, E_min, E_max in zip(range(len(E_bins) - 1), E_bins[:-1], E_bins[1:]):
                on_count, off_count, _, _, _ = self.compute_on_off_counts(
                    dl2,
                    reco_direction,
                    pointing_direction,
                    n_off=n_off,
                    theta2_cut=0.04 * u.deg**2,
                    gcut=self.cuts[cuts_index].gammaness_cut,
                    E_min=E_min,
                    E_max=E_max,
                )
                t_eff_temp, _ = self.compute_eff_time(dl2)
                excess_rates[i] += (
                    (on_count - off_count / n_off) / t_eff_temp.to(u.s)
                ).value
                background_rates[i] += (off_count / n_off / t_eff_temp.to(u.s)).value
                t_eff += t_eff_temp

        E = (E_bins[:-1] + E_bins[1:]) / 2

        plt.plot(E.value, excess_rates, marker="o", linestyle="-", label="Excess Rate")
        plt.plot(
            E.value,
            background_rates,
            marker="o",
            linestyle="-",
            label="Background Rate",
        )
        plt.xlabel("Reco Energy [TeV]")
        plt.ylabel("Rate [Hz]")
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Excess and Background Rates vs Energy")
        plt.legend()

        plt.tight_layout()
        if output_file is not None:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()

    def plot_gammaness_distribution(self, output_file=None):
        import matplotlib.pyplot as plt

        gammaness_values = []
        for dl2 in self.dl2s:
            # Extracting the gammaness values
            gammaness_values.extend(dl2[self.gammaness_key])

        # Plotting the histograms
        plt.hist(
            gammaness_values,
            bins=100,
            range=(0, 1),
            histtype="step",
            density=False,
            lw=2,
            label="Real data",
        )
        plt.xlabel("Gammaness")
        plt.ylabel("Counts")
        plt.legend()

        plt.tight_layout()
        if output_file is not None:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()

    def plot_energy_distribution(self, output_file=None, bins=None, gammaness_cut=0.9):
        import matplotlib.pyplot as plt

        energy_values = []
        for dl2 in self.dl2s:
            # Extracting the energy values
            energy_values.extend(
                dl2[self.energy_key][dl2[self.gammaness_key] > gammaness_cut]
            )

        # Plotting the histograms
        if bins is None:
            bins = np.logspace(
                np.log10(min(energy_values)), np.log10(max(energy_values)), 100
            )
        plt.hist(
            energy_values,
            bins=bins,
            histtype="step",
            density=False,
            lw=2,
            label=f"Real data gcut {gammaness_cut}",
        )
        plt.xlabel("Energy [TeV]")
        plt.xscale("log")
        plt.ylabel("Counts")
        plt.legend()

        plt.tight_layout()
        if output_file is not None:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()

    def plot_everything(self, output_directory: str, suffix: str = ""):
        self.plot_sensitivity(
            output_file=f"{output_directory}/sensitivity_{suffix}.png"
        )
        self.plot_gammaness_distribution(
            output_file=f"{output_directory}/gammaness_distribution_{suffix}.png"
        )
        self.plot_skymap(output_file=f"{output_directory}/skymap_{suffix}.png")
        self.plot_theta2_distribution(
            25, output_file=f"{output_directory}/theta2_distribution_{suffix}.png"
        )
        self.plot_bkg_discrimination_capability(
            output_file=f"{output_directory}/bkg_discrimination_capability_{suffix}.png"
        )
        self.plot_excess_vs_background_rates(
            output_file=f"{output_directory}/excess_vs_background_rates_{suffix}.png"
        )
        self.plot_excess_and_background_rates_vs_energy(
            output_file=f"{output_directory}/excess_and_background_rates_vs_energy_{suffix}.png"
        )
        self.plot_PSF(output_file=f"{output_directory}/psf_{suffix}.png")
