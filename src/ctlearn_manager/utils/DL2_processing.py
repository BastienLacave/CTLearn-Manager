import faulthandler
import multiprocessing as mp
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.io import fits
from pyirf.statistics import li_ma_significance
from tqdm import tqdm

from ..tri_model import CTLearnTriModelManager
from ..tri_model_collection import TriModelCollection
from ..utils.utils import (
    CurveType,
    Cuts,
    CutType,
    DefaultCuts,
    ExportCurves,
    ParticleType,
    calc_flux_for_N_sigma,
    calc_flux_for_N_sigma_array,
    find_68_percent_range,
    get_avg_pointing,
    get_color,
    get_closest_rf_irf_files,
)


def extract_cuts(args):
    model, file, cut, pointing_table, pointing_alt_key, pointing_az_key = args
    # zenith, azimuth = model.project_directories.get_available_MC_directions(ParticleType.GAMMA_POINT)
    # file_zenith, file_azimuth = get_avg_pointing(
    #     file,
    #     pointing_table,
    #     alt_key=pointing_alt_key,
    #     az_key=pointing_az_key,
    # )
    # zenith = np.asarray(zenith)
    # azimuth = np.asarray(azimuth)
    # Compute angular distance between file pointing and all available MC directions
    # distances = np.sqrt((zenith - file_zenith.value) ** 2 + (azimuth - file_azimuth.value) ** 2)
    # closest_idx = np.argmin(distances)
    # cuts_file = model.project_directories.get_irf_files(
    #     zenith[closest_idx] * u.deg, azimuth[closest_idx] * u.deg, cut
    # )['cuts_file']
    zenith, azimuth = get_avg_pointing(file, pointing_table, pointing_alt_key, pointing_az_key)
    cuts_file = model.project_directories.get_closest_irf_files(zenith.value, azimuth.value, cut)['cuts_file'] 
    with fits.open(cuts_file, mode="readonly") as hdul:
        theta_cut = hdul["RAD_MAX"].data["cut"]
        gammaness_cut = hdul["GH_CUTS"].data["cut"]
    return theta_cut, gammaness_cut

def get_energy_dependent_mask_data(
    DL2_file,pointing_table,pointing_alt_key, pointing_az_key, CTLearn,
    data,
    source_position,
    energy_key,
    gammaness_key,
    tri_model: CTLearnTriModelManager,
    reco_coord,
    theta_cut=True,
    cuts: Cuts = DefaultCuts.EFF_70.value,
):
    from astropy.io import fits

    # zenith, azimuth = tri_model.project_directories.get_available_MC_directions(ParticleType.GAMMA_POINT)
    # cuts_file = tri_model.project_directories.get_irf_files(zenith[0], azimuth[0], cuts)['cuts_file'] # FIXME
    if CTLearn:
        zenith, azimuth = get_avg_pointing(DL2_file, pointing_table, pointing_alt_key, pointing_az_key)
        cuts_file = tri_model.project_directories.get_closest_irf_files(zenith.value, azimuth.value, cuts)['cuts_file'] 
    else:
        zenith, azimuth = get_avg_pointing(DL2_file, pointing_table, pointing_alt_key, pointing_az_key)
        cuts_file = get_closest_rf_irf_files(zenith.value, cuts)

    with fits.open(cuts_file) as hdul:
        if CTLearn:
            gammaness_cuts = hdul["GH_CUTS"].data["cut"]
            energy_low = hdul["GH_CUTS"].data["low"]
            energy_high = hdul["GH_CUTS"].data["high"]
            theta_cuts = hdul["RAD_MAX"].data["cut"]
        else:
            gammaness_cuts = hdul["GH_CUTS"].data["GH_CUTS"]
            energy_low = hdul["GH_CUTS"].data["true_energy_low"]
            energy_high = hdul["GH_CUTS"].data["true_energy_high"]
            theta_cuts = hdul["THETA_CUTS"].data["cut"]

        # Compute separation only once
        if "angular_separation" not in data.colnames:
            data["angular_separation"] = reco_coord.separation(source_position)

        energy = data[energy_key]
        gammaness = data[gammaness_key]
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
    
def compute_eff_time(events, CTLearn, time_key):
    # Extract timestamps and delta_t as numpy arrays
    if CTLearn:
        # print(events[time_key])
        timestamp = np.asarray(events[time_key].to_value("unix"))
    else:
        timestamp = np.asarray(events[time_key])

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

def load_one_worker(args):
    (
        i, CTLearn, pointing_table, pointing_alt_key, pointing_az_key,
        DL2_file, CTLearnTriModelCollection, source_position,
        gammaness_key, energy_key, intensity_key, intensity_cut,
        cuts, dl2_processed_dir, global_gammaness_cut, time_key
    ) = args
    # try:
    # Find corresponding model
    if CTLearn:
        corresponding_model = CTLearnTriModelCollection.find_closest_model_to(
            DL2_file,
            pointing_table,
            alt_key=pointing_alt_key,
            az_key=pointing_az_key,
            verbose=False,
        )
        if corresponding_model is None:
            return i, False
    else:
        corresponding_model = CTLearnTriModelCollection.tri_models[0]

    # File paths
    dl2_output_file = os.path.join(dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_dl2_processed.pkl'))
    reco_output_file = os.path.join(dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_reco_directions.pkl'))
    pointing_output_file = os.path.join(dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_pointings.pkl'))
    I_g_on_counts_output_file = os.path.join(dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_I_g_on_counts.pkl'))
    I_g_off_counts_output_file = os.path.join(dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_I_g_off_counts.pkl'))

    # Load files
    if not (os.path.exists(reco_output_file) and os.path.exists(pointing_output_file) and os.path.exists(dl2_output_file)):
        return (i, None, None, None, None, None, None, None, None, False, 0, 0, 0)

    with open(reco_output_file, "rb") as f:
        transformed_reco_dict = pickle.load(f)
    with open(pointing_output_file, "rb") as f:
        transformed_pointing_dict = pickle.load(f)
    transformed_reco = SkyCoord(
        ra=transformed_reco_dict["ra"] * u.deg,
        dec=transformed_reco_dict["dec"] * u.deg,
        frame=source_position,
    )
    transformed_pointing = SkyCoord(
        ra=transformed_pointing_dict["ra"] * u.deg,
        dec=transformed_pointing_dict["dec"] * u.deg,
        frame=source_position,
    )
    

    with open(dl2_output_file, "rb") as f:
        dl2 = pickle.load(f)
    # print(dl2.colnames)
    eff_time, _ = compute_eff_time(dl2, CTLearn, time_key)
    if gammaness_key in dl2.colnames:
        n_before = len(dl2)
        garbage_mask = (dl2[gammaness_key] > global_gammaness_cut) & (dl2[energy_key] > 0) & (dl2[intensity_key] > intensity_cut)
        dl2 = dl2[garbage_mask]
        n_after = len(dl2)
        # print(f"Cut {n_before - n_after} events based on intensity cuts.")
        cut_mask = np.empty(len(cuts), dtype=object)
        cut_mask_gammaness_only = np.empty(len(cuts), dtype=object)
        for j, cut in enumerate(cuts):
            if cut.cut_type in [CutType.GLOBAL]:
                mask = dl2[gammaness_key] > cut.gammaness_cut
                cut_mask[j] = mask
                cut_mask_gammaness_only[j] = mask
            else:
                mask = get_energy_dependent_mask_data(
                    DL2_file, pointing_table, pointing_alt_key, pointing_az_key, CTLearn,
                    dl2, source_position,
                    energy_key,
                    gammaness_key, corresponding_model, transformed_reco[garbage_mask], cuts=cut
                )
                mask_gam = get_energy_dependent_mask_data(
                    DL2_file, pointing_table, pointing_alt_key, pointing_az_key, CTLearn,
                    dl2, source_position,
                    energy_key,
                    gammaness_key, corresponding_model, transformed_reco[garbage_mask], False, cuts=cut
                )
                cut_mask[j] = mask
                cut_mask_gammaness_only[j] = mask_gam
    else:
        cut_mask = [np.ones(len(dl2), dtype=bool)]
        cut_mask_gammaness_only = [np.ones(len(dl2), dtype=bool)]
    reco_direction = transformed_reco[garbage_mask]
    pointing = transformed_pointing[garbage_mask]

    # On-off counts
    if os.path.exists(I_g_on_counts_output_file) and os.path.exists(I_g_off_counts_output_file):
        with open(I_g_on_counts_output_file, "rb") as f:
            I_g_on_counts = pickle.load(f)
        with open(I_g_off_counts_output_file, "rb") as f:
            I_g_off_counts = pickle.load(f)
    else:
        return (i, None, None, None, None, None, None, None, None, False, 0, 0, 0)

    return (i, reco_direction, pointing, dl2, cut_mask, cut_mask_gammaness_only, I_g_on_counts, I_g_off_counts, corresponding_model, True, n_before, n_after, eff_time)


def process_gt_tuple(args):
    i_g, i_theta, gcut, theta2_cut, gammaness, on_sep, off_sep = args
    gcut_val = gcut[i_g]
    theta2_val = theta2_cut[i_theta]
    t2_sqrt = np.sqrt(theta2_val)
    g_mask = gammaness > gcut_val
    theta_mask = on_sep < t2_sqrt
    on_count = np.count_nonzero(g_mask & theta_mask)
    # OFF
    off_mask = np.zeros(off_sep.shape, dtype=bool)
    for i_off in range(off_sep.shape[0]):
        off_mask[i_off] = g_mask & (off_sep[i_off] < t2_sqrt)
    off_count = np.sum(off_mask)
    return i_g, i_theta, on_count, off_count

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
    plot_theta2_distribution(self, bins, n_off=5):
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
        intensity_cut: int=80,
        global_gammaness_cut: float=0.,
        # max_theta2: float=0.2,
        workers=None
    ):
        self.workers = workers
        mp.set_start_method("fork", force=True)
        faulthandler.enable()
        self.DL2_files = np.sort(DL2_files)
        self.intensity_cut = intensity_cut
        self.global_gammaness_cut = global_gammaness_cut
        # self.max_theta2 = max_theta2
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


        self.cut_file_theta_cuts = []
        self.cut_file_gammaness_cuts = []

        

        for i, cut in enumerate(self.cuts):
            if (
                cut.cut_type == CutType.EFFICIENCY_OPTIMIZED
                or cut.cut_type == CutType.SENSITIVITY_OPTIMIZED
            ):
                file_args = [
                    (model, file, cut, self.pointing_table, self.pointing_alt_key, self.pointing_az_key)
                    for model, file in zip(self.corresponding_models, self.DL2_files)
                ]
                file_theta_cuts = []
                file_gammaness_cuts = []
                with ProcessPoolExecutor(self.workers) as executor:
                    results = list(tqdm(executor.map(extract_cuts, file_args), desc=f"Creating masks [{cut.get_label()}]", total=len(file_args)))
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
        self.pointing_alt_key = "altitude" if self.CTLearn else "alt_tel"
        self.pointing_az_key = "azimuth" if self.CTLearn else "az_tel"
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

        # def load_one(i_DL2_file):
        #     i, DL2_file = i_DL2_file
        #     # try:
        #     # Find corresponding model
        #     if self.CTLearn:
        #         corresponding_model = self.CTLearnTriModelCollection.find_closest_model_to(
        #             DL2_file,
        #             self.pointing_table,
        #             alt_key=self.pointing_alt_key,
        #             az_key=self.pointing_az_key,
        #             verbose=False,
        #         )
        #         if corresponding_model is None:
        #             return i, False
        #     else:
        #         corresponding_model = self.CTLearnTriModelCollection.tri_models[0]

        #     self.corresponding_models[i] = corresponding_model

        #     # File paths
        #     dl2_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_dl2_processed.pkl'))
        #     reco_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_reco_directions.pkl'))
        #     pointing_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_pointings.pkl'))
        #     I_g_on_counts_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_I_g_on_counts.pkl'))
        #     I_g_off_counts_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_I_g_off_counts.pkl'))

        #     # Load files
        #     if not (os.path.exists(reco_output_file) and os.path.exists(pointing_output_file) and os.path.exists(dl2_output_file)):
        #         return i, False

        #     with open(reco_output_file, "rb") as f:
        #         transformed_reco_dict = pickle.load(f)
        #     with open(pointing_output_file, "rb") as f:
        #         transformed_pointing_dict = pickle.load(f)
        #     transformed_reco = SkyCoord(
        #         ra=transformed_reco_dict["ra"] * u.deg,
        #         dec=transformed_reco_dict["dec"] * u.deg,
        #         frame=self.source_position,
        #     )
        #     transformed_pointing = SkyCoord(
        #         ra=transformed_pointing_dict["ra"] * u.deg,
        #         dec=transformed_pointing_dict["dec"] * u.deg,
        #         frame=self.source_position,
        #     )
            

        #     with open(dl2_output_file, "rb") as f:
        #         dl2 = pickle.load(f)
        #     # print(dl2.colnames)
        #     if self.gammaness_key in dl2.colnames:
        #         n_before = len(dl2)
        #         garbage_mask = (dl2[self.gammaness_key] > 0) & (dl2[self.energy_key] > 0) & (dl2[self.intensity_key] > self.intensity_cut)
        #         dl2 = dl2[garbage_mask]
        #         n_after = len(dl2)
        #         print(f"Cut {n_before - n_after} events based on intensity cuts.")
        #         cut_mask = np.empty(len(self.cuts), dtype=object)
        #         cut_mask_gammaness_only = np.empty(len(self.cuts), dtype=object)
        #         for j, cut in enumerate(self.cuts):
        #             if cut.cut_type in [CutType.GLOBAL]:
        #                 mask = dl2[self.gammaness_key] > cut.gammaness_cut
        #                 cut_mask[j] = mask
        #                 cut_mask_gammaness_only[j] = mask
        #             else:
        #                 mask = self.get_energy_dependent_mask_data(
        #                     dl2, corresponding_model, transformed_reco[garbage_mask], cuts=cut
        #                 )
        #                 mask_gam = self.get_energy_dependent_mask_data(
        #                     dl2, corresponding_model, transformed_reco[garbage_mask], False, cuts=cut
        #                 )
        #                 cut_mask[j] = mask
        #                 cut_mask_gammaness_only[j] = mask_gam
        #     else:
        #         cut_mask = [np.ones(len(dl2), dtype=bool)]
        #         cut_mask_gammaness_only = [np.ones(len(dl2), dtype=bool)]
        #     self.cuts_masks[i] = cut_mask
        #     self.cuts_masks_gammaness_only[i] = cut_mask_gammaness_only
        #     self.dl2s[i] = dl2
        #     self.reco_directions[i] = transformed_reco[garbage_mask]
        #     self.pointings[i] = transformed_pointing[garbage_mask]

        #     # On-off counts
        #     if os.path.exists(I_g_on_counts_output_file) and os.path.exists(I_g_off_counts_output_file):
        #         with open(I_g_on_counts_output_file, "rb") as f:
        #             self.I_g_on_counts[i] = pickle.load(f)
        #         with open(I_g_off_counts_output_file, "rb") as f:
        #             self.I_g_off_counts[i] = pickle.load(f)
        #     else:
        #         return i, False

        #     return i, True
        #     # except Exception as e:
        #     #     print(f"Error loading {DL2_file}: {e}")
        #     #     return i, False


        n_files = len(self.DL2_files)
        results = []
        args_list = [
            (
                i,
                self.CTLearn,
                self.pointing_table,
                self.pointing_alt_key,
                self.pointing_az_key,
                DL2_file,
                self.CTLearnTriModelCollection,
                self.source_position,
                self.gammaness_key,
                self.energy_key,
                self.intensity_key,
                self.intensity_cut,
                self.cuts,
                self.dl2_processed_dir,
                self.global_gammaness_cut,
                self.time_key,
            )
            for i, DL2_file in enumerate(self.DL2_files)
        ]

        with ProcessPoolExecutor(self.workers) as executor:
            for result in tqdm(executor.map(load_one_worker, args_list), total=n_files, desc="Loading processed data"):
                results.append(result)
        # Parallel loading
        # with ProcessPoolExecutor() as executor:
        #     results = list(tqdm(executor.map(load_one, enumerate(self.DL2_files)), total=n_files, desc="Loading processed data"))
        
        

        # for i, success in results:
        #     failed_mask[i] = success
        n_saved_tot = 0
        n_tot = 0
        self.effective_times = np.empty(n_files, dtype=object)
        for res in results:
            i, reco_direction, pointing, dl2, cuts_mask, cuts_mask_gammaness_only, I_g_on_counts, I_g_off_counts, corresponding_model, success, n_before, n_after, eff_time = res
            failed_mask[i] = success
            if success:
                n_saved_tot += n_before - n_after
                n_tot += n_before
                self.reco_directions[i] = reco_direction
                self.pointings[i] = pointing
                self.dl2s[i] = dl2
                self.cuts_masks[i] = cuts_mask
                self.cuts_masks_gammaness_only[i] = cuts_mask_gammaness_only
                self.I_g_on_counts[i] = I_g_on_counts
                self.I_g_off_counts[i] = I_g_off_counts
                self.corresponding_models[i] = corresponding_model
                self.effective_times[i] = eff_time

        print(f"Cut {n_saved_tot} ({(n_saved_tot/n_tot * 100):.2f}%) events in total from intensity cut {self.intensity_cut}p.e. and global gammaness cut {self.global_gammaness_cut}.")
        print(f"Remaining {n_tot - n_saved_tot} ({((n_tot - n_saved_tot)/n_tot * 100):.2f}%) events in total after cuts.")
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
        self.effective_times = self.effective_times[failed_mask]
        self.effective_time = np.sum(self.effective_times) if len(self.effective_times) > 0 else 0 * u.h
        # print(self.effective_time)
        # print(self.effective_times)
        print(f"Total effective time: {self.effective_time.to(u.h):.2f}")



    def plot_theta2_distribution(self, bins=25, n_off=5, output_file=None, cuts_index=0):
        import concurrent.futures

        import matplotlib.pyplot as plt

        angle2_bins = np.linspace(0, 0.4, bins)
        angle2_center = (angle2_bins[:-1] + angle2_bins[1:]) / 2
        h_on = np.zeros(bins - 1)
        h_off = np.zeros(bins - 1)
        t_eff = self.effective_time
        # t_elapsed = 0 * u.h

        def process_file(args):
            reco_direction, pointing_direction, dl2, cuts_mask = args
            cuts_mask = cuts_mask[cuts_index]
            reco_direction = reco_direction[cuts_mask]
            pointing_direction = pointing_direction[cuts_mask]
            # t_eff_temp, t_elapsed_temp = self.compute_eff_time(dl2, self.CTLearn, self.time_key)
            dl2 = dl2[cuts_mask]
            on_count_temp, off_count_temp, on_separation_temp, all_off_separation_temp, _ = self.compute_on_off_counts(
                dl2,
                reco_direction,
                pointing_direction,
                n_off=n_off,
                theta2_cut=0.04 * u.deg**2, # FIXME
                gcut=0,
                E_min=0,
                E_max=10000,
                I_min=None,
                I_max=None,
            )
            h_on_temp, _ = np.histogram(on_separation_temp.to(u.deg).value ** 2, bins=angle2_bins)
            h_off_temp, _ = np.histogram(all_off_separation_temp.to(u.deg).value ** 2, bins=angle2_bins)
            return (on_count_temp, off_count_temp, h_on_temp, h_off_temp)

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

        lima_signi = li_ma_significance(
            np.float64(on_count_tot), np.float64(off_count_tot), alpha= 1 / n_off
        )
        fig, ax = plt.subplots()
        self.cuts[cuts_index].plot_cuts_info_plt(ax)
        label = (
            "$t_{eff}$ = "
             f"{t_eff.to(u.h):.2f}"
             "\n$N_{on}$ = "
             f"{on_count_tot}\t"
             r"$\overline{N}_{off}$ = "
             f"{(off_count_tot / n_off):.1f}"
             "\n$N_{excess}$ = "
             f"{(on_count_tot - off_count_tot / n_off):.1f}\t"
             r"$\sigma_{Li&Ma}$ = "
             f"{lima_signi:.2f}"
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
    
    def compute_on_off_counts_array_nul(
        self,
        events,
        on_sep,
        off_sep,
        theta2_cut,
        gcut,
        E_min=0,
        E_max=100,
    ):
        theta2_cut = np.atleast_1d(theta2_cut)
        gcut = np.atleast_1d(gcut)
        n_theta = len(theta2_cut)
        n_g = len(gcut)

        if hasattr(E_min, "value"):
            E_min = E_min.value
        if hasattr(E_max, "value"):
            E_max = E_max.value
        energy_mask = (events[self.energy_key] > E_min) & (events[self.energy_key] < E_max)
        if np.sum(energy_mask) == 0:
            shape = (n_g, n_theta)
            return np.zeros(shape), np.zeros(shape)

        events = events[energy_mask]
        gammaness = events[self.gammaness_key]
        on_sep = on_sep[energy_mask]
        off_sep = off_sep[:, energy_mask]

        
        # Prepare all bin indices
        bin_indices = [(i_g, i_theta, gcut, theta2_cut, gammaness, on_sep, off_sep) for i_g in range(n_g) for i_theta in range(n_theta)]
        on_count = np.zeros((n_g, n_theta), dtype=int)
        off_count = np.zeros((n_g, n_theta), dtype=int)

        with ProcessPoolExecutor(self.workers) as executor:
            for i_g, i_theta, on_c, off_c in tqdm(executor.map(process_gt_tuple, bin_indices), desc="Computing on-off counts", total=len(bin_indices)):
                on_count[i_g, i_theta] = on_c
                off_count[i_g, i_theta] = off_c

        return on_count, off_count

    def compute_on_off_counts_array(
        self,
        events,
        # reco_coord,
        # pointing_coord,
        # n_off,
        on_sep,
        off_sep,
        theta2_cut,
        gcut,
        E_min=0,
        E_max=100,
    ):
        # import cProfile, pstats
        # profiler = cProfile.Profile()
        # profiler.enable()
        theta2_cut = np.atleast_1d(theta2_cut)
        gcut = np.atleast_1d(gcut)
        n_theta = len(theta2_cut)
        n_g = len(gcut)
        
        # print(gcut)
        # print(theta2_cut)

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
        # print("Energy mask applied, remaining events:", len(events))
        # gammaness = events[self.gammaness_key]
        # on_sep = on_sep[energy_mask]
        # off_sep = off_sep[:,energy_mask]
        # reco_coord = reco_coord[energy_mask]
        # pointing_coord = pointing_coord[energy_mask]

        # --- Compute ON separations ---
        # on_sep = reco_coord.separation(self.source_position).to(u.deg).value  # (N_events,)

        # Prepare 3D mask for ON: (n_g, n_theta, N_events)
        gammaness_2d = events[self.gammaness_key][None, None, :]  # (1, 1, N_events)
        gcut_2d = gcut[:, None, None]            # (n_g, 1, 1)
        on_sep_2d = on_sep[energy_mask][None, None, :]        # (1, 1, N_events)
        t2_sqrt = np.sqrt(theta2_cut)[None, :, None]  # (1, n_theta, 1)

        g_mask = gammaness_2d > gcut_2d          # (n_g, 1, N_events)
        theta_mask = on_sep_2d < t2_sqrt         # (1, n_theta, N_events)
        on_mask = g_mask & theta_mask            # (n_g, n_theta, N_events)
        on_count = np.sum(on_mask, axis=2)       # (n_g, n_theta)

        # --- Compute OFF separations ---
        # off_regions = self.compute_off_regions(pointing_coord, n_off)
        # # Vectorized: stack all off regions and compute separations in one call
        # off_sep = np.array([
        #     reco_coord.separation(off_regions[i]).to(u.deg).value for i in range(n_off)
        # ])  # (n_off, N_events)
        # off_sep = reco_coord.separation(off_regions[:, None]).to(u.deg).value  # shape: (n_off, N_events)

        # Prepare for broadcasting
        off_sep_3d = off_sep[:,energy_mask][None, None, :, :]  # (1, 1, n_off, N_events)
        t2_sqrt_3d = t2_sqrt[:, :, None]        # (1, n_theta, 1)
        g_mask_3d = g_mask[:, :, None, :]       # (n_g, 1, 1, N_events)

        # Mask for all off regions at once
        theta_mask_off = off_sep_3d < t2_sqrt_3d  # (1, n_theta, n_off, N_events)
        off_mask = g_mask_3d & theta_mask_off     # (n_g, n_theta, n_off, N_events)
        off_count = np.sum(off_mask, axis=(2, 3)) # (n_g, n_theta)

        # profiler.disable()
        # stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.print_stats()  # Print top 20 time-consuming functions
        return on_count, off_count

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
        import concurrent.futures

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
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
        ax.hist2d(ra_values, dec_values, bins=100, zorder=0)
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
                print("No pointings available for this cuts index, skipping plotting.")
                continue
            off_regions = self.compute_off_regions(pointing[0], n_off=5)
            ax.scatter(
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
                ax.add_artist(off_circle)

        on_circle = plt.Circle(
            (self.source_position.ra.deg, self.source_position.dec.deg),
            radius=0.2,
            color="w",
            fill=False,
            lw=1,
        )
        ax.add_artist(on_circle)
        ax.set_aspect("equal", adjustable="box")

        if output_file is not None:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
    
    def optimize_cuts_on_crab(self, n_off=5, output_suffix="", max_gammaness_cut=1.0, max_theta2_cut=0.2, gcut_step=0.01, theta2_cut_step=0.001, E_bins=None):
        """
        Compute and store optimal gammaness/theta2 cuts for even and odd events for each energy bin.
        """
        import concurrent.futures
        import csv

        from astropy.coordinates import concatenate

        assert max_gammaness_cut > self.global_gammaness_cut, "max_gammaness_cut must be greater than global_gammaness_cut"

        if E_bins is None:
            E_bins = self.E_bins[0]
            # import importlib.resources as pkg_resources
            # from .. import resources
            # with pkg_resources.path(resources, "sensitivity_src_indep.txt") as text_path:
            #     data = np.loadtxt(text_path)

            # energy_med = data[:,0] 
            # E_bins 
        gammaness_bins_tot = np.arange(self.global_gammaness_cut, max_gammaness_cut + 0.00001, gcut_step)
        theta2bins_tot = np.arange(0, max_theta2_cut + 0.00001, theta2_cut_step)
        # Split gammaness_bins and theta2bins into arrays of 50 elements or less
        def split_bins(bins, max_size=50):
            return [bins[i:i+max_size] for i in range(0, len(bins), max_size)]

        gammaness_bins_split = split_bins(gammaness_bins_tot, 50)
        theta2bins_split = split_bins(theta2bins_tot, 50)
        # gammaness_bins = np.linspace(self.global_gammaness_cut, 1, int((1-self.global_gammaness_cut)*201)) #2001
        # theta2bins = np.linspace(0, self.max_theta2, 301) # 6001
        n_bins = len(E_bins) - 1
        print(f"Parameter space: {len(gammaness_bins_tot)} x {len(theta2bins_tot)}", flush=True)

        def mask_events(args):
            reco_direction, pointing_direction, dl2 = args
            even_mask = dl2["event_id"] % 2 == 0
            odd_mask = dl2["event_id"] % 2 == 1
            if self.time_key in dl2.colnames:
                dl2.remove_column(self.time_key)
            return (
                dl2[even_mask], dl2[odd_mask],
                reco_direction[even_mask], reco_direction[odd_mask],
                pointing_direction[even_mask], pointing_direction[odd_mask],
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(mask_events, zip(self.reco_directions, self.pointings, self.dl2s)), desc="Masking events", total=len(self.reco_directions)))

        even_dl2_temp, odd_dl2_temp, even_reco_temp, odd_reco_temp, even_pointing_temp, odd_pointing_temp = zip(*results)
        t_eff_temp = self.effective_time

        print("Concatenating even_dl2...", flush=True)
        even_dl2 = np.concatenate(even_dl2_temp)
        print(f"# of even events: {len(even_dl2)}", flush=True)
        print("Concatenating odd_dl2...", flush=True)
        odd_dl2 = np.concatenate(odd_dl2_temp)
        print(f"# of odd events: {len(odd_dl2)}", flush=True)
    
        # print(even_reco_temp)
        if len(even_reco_temp) > 1:
            # print("Concatenating even_reco...", flush=True)
            even_reco = concatenate(even_reco_temp)
            # print("Concatenating odd_reco...", flush=True)
            odd_reco = concatenate(odd_reco_temp)
            # print("Concatenating even_pointing...", flush=True)
            even_pointing = concatenate(even_pointing_temp)
            # print("Concatenating odd_pointing...", flush=True)
            odd_pointing = concatenate(odd_pointing_temp)
        else:
            even_reco = even_reco_temp[0]
            odd_reco = odd_reco_temp[0]
            even_pointing = even_pointing_temp[0]
            odd_pointing = odd_pointing_temp[0]

        
        def plot_flux_map(E_min, E_max, flux, min_idx, min_off_mask, excess_vs_bkg_mask, even=True):
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap, LogNorm
            cmap_min_off = ListedColormap([get_color('ctlearn_accent_1'), 'none'])
            cmap_excess = ListedColormap([get_color('ctlearn_accent_2'), 'none'])
            im = plt.imshow(
                flux,
                aspect='auto',
                origin='lower',
                cmap='viridis',
                norm=LogNorm(vmin=np.nanmin(flux[flux > 0]), vmax=np.nanmax(flux)),
                extent=[
                    theta2bins_tot[0], theta2bins_tot[-1],
                    gammaness_bins_tot[0], gammaness_bins_tot[-1]
                ],
            )
            cbar = plt.colorbar(im, label='Flux Factor [a.u.]')
            cbar.set_ticks([])
            cbar.ax.tick_params(which='minor', length=0)
            cbar.ax.set_yticklabels([])
            cbar.ax.set_xticks([])
            cbar.ax.set_yticks([])
            # Remove minor tick labels
            from matplotlib.ticker import NullLocator
            cbar.ax.yaxis.set_minor_locator(NullLocator())
            cbar.ax.yaxis.set_minor_formatter(lambda *args, **kwargs: "")
            # plt.colorbar(label='Flux Factor')
            plt.title(f"{'Even' if even else 'Odd'} [{E_min.value:.3f}, {E_max.value:.3f}] TeV")
            plt.xlabel('Theta2 Cut')
            plt.ylabel('Gammaness Cut')

            plt.tight_layout()
            if min_idx != -1:
                plt.scatter(
                    theta2bins_tot[min_idx[1]],
                    gammaness_bins_tot[min_idx[0]],
                    color='red',
                    marker='o',
                    s=120,
                    label='Min Flux',
                    facecolors='none',
                    edgecolors='red'
                )
            mask =(~min_off_mask).astype(float)
            mask[min_off_mask] = np.nan  # Set False to nan (transparent)
            

            # Use imshow with a transparent colormap and hatch for True area
            # cmap = ListedColormap(['g', 'none'])  # All transparent
            im = plt.imshow(
                mask,
                aspect='auto',
                origin='lower',
                extent=[
                    theta2bins_tot[0], theta2bins_tot[-1],
                    gammaness_bins_tot[0], gammaness_bins_tot[-1]
                ],
                interpolation='nearest',
                alpha=0.3,  # Fully transparent
                cmap=cmap_min_off
            )
            
            # Create a mask array for plotting
            mask = (~excess_vs_bkg_mask).astype(float)
            mask[excess_vs_bkg_mask] = np.nan  # Set False to nan (transparent)

            # Use imshow with a transparent colormap and hatch for True area
            # cmap = ListedColormap(['r', 'none'])  # All transparent
            im = plt.imshow(
                mask,
                aspect='auto',
                origin='lower',
                extent=[
                    theta2bins_tot[0], theta2bins_tot[-1],
                    gammaness_bins_tot[0], gammaness_bins_tot[-1]
                ],
                interpolation='nearest',
                alpha=0.3,  # Fully transparent
                cmap=cmap_excess
            )
            plt.legend(
                handles=[
                    mpatches.Patch(color=get_color('ctlearn_accent_1'), label='Bkg. < 10ev.'),
                    mpatches.Patch(color=get_color('ctlearn_accent_2'), label='Exc. < 5% bkg.'),
                    plt.Line2D([], [], color='red', marker='o', linestyle='None',markerfacecolor='none',
                    markeredgecolor='red', label='Min Flux', markeredgewidth=2)
                ],
                loc='upper right',
                # title=f"[{E_min:.3f}, {E_max:.3f}] TeV"
            )
            plt.savefig(f"{self.CTLearnTriModelCollection.project_directories.plots_directory}/flux_{'Even' if even else 'Odd'}_{len(self.DL2_files)}_files{output_suffix}_g{self.global_gammaness_cut}_{max_gammaness_cut}_t{max_theta2_cut}_{E_min:.4f}_{E_max:.4f}.png")
            plt.show()
            plt.close()
        
        import time
        time_i = time.time()
        on_sep_even = even_reco.separation(self.source_position).to(u.deg).value  # (N_events,)
        off_regions = self.compute_off_regions(even_pointing, n_off)
        # Vectorized: stack all off regions and compute separations in one call
        off_sep_even = np.array([
            even_reco.separation(off_regions[i]).to(u.deg).value for i in range(n_off)
        ])  # (n_off, N_events)
        print(f"Time for EVEN sep: {time.time() - time_i:.2f} seconds", flush=True)

        time_i = time.time()
        on_sep_odd = odd_reco.separation(self.source_position).to(u.deg).value  # (N_events,)
        off_regions = self.compute_off_regions(odd_pointing, n_off)
        # Vectorized: stack all off regions and compute separations in one call
        off_sep_odd = np.array([
            odd_reco.separation(off_regions[i]).to(u.deg).value for i in range(n_off)
        ])  # (n_off, N_events)
        print(f"Time for ODD sep: {time.time() - time_i:.2f} seconds", flush=True)

        def process_bin(args):
            
            i, E_min, E_max = args
            
            # Compute on/off counts for all grid points (vectorized)
            # print(even_reco)
            # print(f"Processing bin {i} : [{E_min:.4f}, {E_max:.4f}]", flush=True)
            min_flux_even = []
            gammaness_cuts_even = []
            theta2_cuts_even = []
            min_flux_odd = []
            gammaness_cuts_odd = []
            theta2_cuts_odd = []

            for gammaness_bins in tqdm(gammaness_bins_split, desc=f"Processing gammaness bins for bin {i}"):
                for theta2bins in theta2bins_split:
                    # print(f"{len(gammaness_bins)} x {len(theta2bins)}", flush=True)
                    time_i = time.time()
                    # EVEN
                    _on, _off = self.compute_on_off_counts_array(
                        even_dl2, on_sep_even, off_sep_even,
                        theta2_cut=theta2bins, gcut=gammaness_bins, E_min=E_min, E_max=E_max
                    )
                    # _nexcess = _on - _off# / 
                    # print(f"Time for EVEN bin: {time.time() - time_i:.2f} seconds", flush=True)
                    flux_even, signi_even, min_off_mask, excess_vs_bkg_mask = calc_flux_for_N_sigma_array(
                        N_sigma = 5, 
                        on_counts = _on, 
                        off_counts = _off, 
                        min_signi = 3, 
                        min_excess = 0.05, 
                        min_off_events = 10, 
                        alpha = 1 / n_off, # Already devided by n_off in compute_on_off_counts_array
                        target_obs_time = 50.0 * u.h, 
                        actual_obs_time = t_eff_temp / 2, 
                        cond=True
                    )
                    # print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
                    try:
                        min_idx_even = np.unravel_index(np.nanargmin(flux_even), flux_even.shape)
                    except ValueError as e:
                        if "All-NaN slice encountered" in str(e):
                            min_idx_even = -1
                        else:
                            raise
                    # plot_flux_map(E_min, E_max, flux_even, min_idx_even, min_off_mask, excess_vs_bkg_mask, even=True)

                    # ODD
                    # print(f"ODD counts {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
                    time_i = time.time()
                    _on, _off = self.compute_on_off_counts_array(
                        odd_dl2, on_sep_odd, off_sep_odd,
                        theta2_cut=theta2bins, gcut=gammaness_bins, E_min=E_min, E_max=E_max
                    )
                    # print(f"Time for ODD bin: {time.time() - time_i:.2f} seconds", flush=True)
                    # _nexcess = _on - _off #/ n_off
                    # t_eff_temp, _ = self.compute_eff_time(odd_dl2)
                    flux_odd, signi_odd, min_off_mask, excess_vs_bkg_mask  = calc_flux_for_N_sigma_array(
                        N_sigma = 5, 
                        on_counts = _on, 
                        off_counts = _off, 
                        min_signi = 3, 
                        min_excess = 0.05, 
                        min_off_events = 10, 
                        alpha = 1/n_off, 
                        target_obs_time = 50.0 * u.h, 
                        actual_obs_time = t_eff_temp / 2, 
                        cond=True
                    )
                    # print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
                    try:
                        min_idx_odd = np.unravel_index(np.nanargmin(flux_odd), flux_odd.shape)
                    except ValueError as e:
                        if "All-NaN slice encountered" in str(e):
                            min_idx_odd = -1
                        else:
                            raise
                    # plot_flux_map(E_min, E_max, flux_odd, min_idx_odd, min_off_mask, excess_vs_bkg_mask, even=False)
                    # min_idx_odd = np.unravel_index(np.nanargmin(flux_odd), flux_odd.shape)

                    if min_idx_even == -1 or min_idx_odd == -1:
                        print(f"Min flux not found for bin {i}, returning NaN.")
                        return (
                            i,
                            np.nan, np.nan,
                            np.nan, np.nan
                        )
                    min_flux_even.append(flux_even[min_idx_even])
                    gammaness_cuts_even.append(gammaness_bins[min_idx_even[0]])
                    theta2_cuts_even.append(theta2bins[min_idx_even[1]])
                    min_flux_odd.append(flux_odd[min_idx_odd])
                    gammaness_cuts_odd.append(gammaness_bins[min_idx_odd[0]])
                    theta2_cuts_odd.append(theta2bins[min_idx_odd[1]])
            # print(f"Min flux even: {flux_even[min_idx_even]:.2e} at gammaness={gammaness_bins[min_idx_even[0]]:.2f}, theta2={theta2bins[min_idx_even[1]]:.4f}")
            # print(f"Min flux odd: {flux_odd[min_idx_odd]:.2e} at gammaness={gammaness_bins[min_idx_odd[0]]:.2f}, theta2={theta2bins[min_idx_odd[1]]:.4f}")
                print(gammaness_cuts_even)
            min_flux_even_index = np.nanargmin(min_flux_even)
            min_flux_odd_index = np.nanargmin(min_flux_odd)
            print(min_flux_even_index, min_flux_odd_index)
            return (
                i,
                gammaness_cuts_even[min_flux_even_index], theta2_cuts_even[min_flux_even_index],
                gammaness_cuts_odd[min_flux_odd_index], theta2_cuts_odd[min_flux_odd_index]
            )

        # Prepare arguments for parallel processing
        bin_args = [(i, E_min, E_max) for i, (E_min, E_max) in enumerate(zip(E_bins[:-1], E_bins[1:]))]
        bin_args.reverse()
        # print(bin_args)
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     results = list(tqdm(executor.map(process_bin, bin_args), total=n_bins, desc="Finding optimal cuts", unit="bins"))
        output_file = f"{self.dl2_processed_dir}/crab_optimized_cuts_{len(self.DL2_files)}_files{output_suffix}.csv"

        # Step 1: Read existing bin indices
        existing_bins = set()
        try:
            with open(output_file, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    existing_bins.add(int(row["bin_index"]))
        except FileNotFoundError:
            # File does not exist yet, will be created below
            pass

        # Step 2: Write header if file does not exist
        if not os.path.exists(output_file):
            with open(output_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["bin_index", "E_min", "E_max", "even_gammaness", "even_theta2", "odd_gammaness", "odd_theta2"])


        # results = []
        for bin_arg in tqdm(bin_args, desc="Finding optimal cuts", total=n_bins, unit="bins"):
            i, _, _ = bin_arg
            if i in existing_bins:
                print(f"Bin {i} already in {output_file}, skipping.", flush=True)
                continue
            i, g_even, t_even, g_odd, t_odd = process_bin(bin_arg)
            print(f"Gcuts {g_even:.2f}\t{g_odd:.2f}\tTheta2 {t_even:.4f}\t{t_odd:.4f}", flush=True)
            # results.append((i, g_even, t_even, g_odd, t_odd))
            with open(output_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    i,
                    E_bins[i],
                    E_bins[i + 1],
                    g_even,
                    t_even,
                    g_odd,
                    t_odd,
                ])
            print(f"Saved cuts for bin {i} to {output_file}", flush=True)

        # Store results
        # for j, g_even, t_even, g_odd, t_odd in results:
        #     # print(g_even, g_odd, t_even, t_odd)
        #     best_gammaness_even[j] = g_even
        #     best_theta2_even[j] = t_even
        #     best_gammaness_odd[j] = g_odd
        #     best_theta2_odd[j] = t_odd
        # print(best_gammaness_even)
        # print(best_gammaness_odd)
        # Save to disk
        # Save to HDF5 file
        
        # with open(output_file, "w", newline="") as csvfile:
        #     writer = csv.writer(csvfile)
        #     # Write header
        #     writer.writerow(["bin_index", "E_min", "E_max", "even_gammaness", "even_theta2", "odd_gammaness", "odd_theta2"])
        #     for i in range(len(E_bins) - 1):
        #         writer.writerow([
        #             i,
        #             E_bins[i],
        #             E_bins[i + 1],
        #             best_gammaness_even[i],
        #             best_theta2_even[i],
        #             best_gammaness_odd[i],
        #             best_theta2_odd[i],
        #         ])
        # print(f"Optimized cuts saved to {output_file}")

    @staticmethod
    def read_cuts_optimized_on_crab_from_csv(csv_filename):
        """
        Read optimal gammaness/theta2 cuts for even and odd events from a CSV file.
        Returns a dict with keys: 'even' and 'odd', each containing a dict with keys 'gammaness', 'theta2', 'E_bins'.
        The bins are sorted by bin_index.
        """
        import csv

        import numpy as np

        rows = []
        with open(csv_filename, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rows.append(row)

        # Sort rows by bin_index
        rows.sort(key=lambda r: int(r["bin_index"]))

        even_gammaness = []
        even_theta2 = []
        odd_gammaness = []
        odd_theta2 = []
        E_bins_low = []
        E_bins_high = []

        for row in rows:
            E_min = float(row["E_min"].split()[0])
            E_max = float(row["E_max"].split()[0])
            E_bins_low.append(E_min)
            E_bins_high.append(E_max)
            even_gammaness.append(float(row["even_gammaness"]))
            even_theta2.append(float(row["even_theta2"]))
            odd_gammaness.append(float(row["odd_gammaness"]))
            odd_theta2.append(float(row["odd_theta2"]))

        # Reconstruct E_bins array
        E_bins = np.array(E_bins_low + [E_bins_high[-1]])
        cuts = {
            "even": {
                "gammaness": np.array(even_gammaness),
                "theta2": np.array(even_theta2),
                "E_bins": E_bins,
            },
            "odd": {
                "gammaness": np.array(odd_gammaness),
                "theta2": np.array(odd_theta2),
                "E_bins": E_bins,
            },
        }
        return cuts

    def plot_cuts_optimized_on_crab(self, cuts_h5_file):
        import matplotlib.pyplot as plt
        cuts = self.read_cuts_optimized_on_crab_from_csv(cuts_h5_file)
        gcuts_even, tcuts_even, E_bins = cuts["even"]["gammaness"], cuts["even"]["theta2"], cuts["even"]["E_bins"]
        gcuts_odd, tcuts_odd = cuts["odd"]["gammaness"], cuts["odd"]["theta2"]
        E_center = (E_bins[:-1] + E_bins[1:]) / 2
        # for i in range(len(E_center)):
        #     print(f"Energy: {E_center[i]:.2f} TeV, {gcuts_even[i]:.2f},{gcuts_odd[i]:.2f}, Even Theta2 Cut: {tcuts_even[i]:.2f}, Odd Theta2 Cut: {tcuts_odd[i]:.2f}")

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        
        # Gammaness cuts subplot
        axs[0].scatter(E_center, gcuts_even, label="Even Cuts")
        axs[0].scatter(E_center, gcuts_odd, label="Odd Cuts", marker='x')
        axs[0].legend()
        axs[0].set_xlabel("Energy [TeV]")
        axs[0].set_ylabel("Gammaness Cut")
        axs[0].set_xscale("log")
        axs[0].set_title("Gammaness Cuts Optimized on Crab")

        # Theta cuts subplot
        axs[1].scatter(E_center, np.sqrt(tcuts_even), label="Even Cuts")
        axs[1].scatter(E_center, np.sqrt(tcuts_odd), label="Odd Cuts", marker='x')
        axs[1].legend()
        axs[1].set_xlabel("Energy [TeV]")
        axs[1].set_ylabel("Theta Cut [deg]")
        axs[1].set_xscale("log")
        axs[1].set_title("Theta Cuts Optimized on Crab")

        for tri_model in self.CTLearnTriModelCollection.tri_models:
            zenith, azimuth = tri_model.project_directories.get_available_MC_directions(ParticleType.GAMMA_POINT)
            for z, a in zip(zenith, azimuth):
                cuts_file = tri_model.project_directories.get_irf_files(z, a, DefaultCuts.EFF_70.value)['cuts_file'] 
                with fits.open(cuts_file) as hdul:
                    axs[0].plot(
                            hdul["GH_CUTS"].data["center"],
                            hdul["GH_CUTS"].data["cut"],
                            color=get_color('on_background'),
                            alpha=0.5,
                            zorder=0,
                            # label=DefaultCuts.EFF_70.value.get_label(),
                        )
                    axs[1].plot(
                            hdul["RAD_MAX"].data["center"],
                            hdul["RAD_MAX"].data["cut"],
                            color=get_color('on_background'),
                            alpha=0.5,
                            zorder=0,
                            # label=DefaultCuts.EFF_70.value.get_label(),
                        )
        
        axs[0].plot([], [], color=get_color('on_background'), alpha=0.5, label=f"MC cuts {DefaultCuts.EFF_70.value.get_label()}")
        axs[0].legend()
        axs[1].plot([], [], color=get_color('on_background'), alpha=0.5, label=f"MC cuts {DefaultCuts.EFF_70.value.get_label()}")
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    def plot_sensitivity(self, n_off=5, ax=None, label="CTLearn", output_file=None, export_to_h5: str=None,
        import_from_h5: str = None,
        import_label: str = None,
        optimized_on_crab: bool = False, output_suffix: str=''):
        import concurrent.futures

        import matplotlib.pyplot as plt

        export_curves = ExportCurves(export_to_h5)
        if import_from_h5 is not None:
            import_curves = ExportCurves(import_from_h5, export_mode=False, import_label=import_label)
            for curve_type in import_curves.curve_types:
                if curve_type not in [CurveType.SENSITIVITY_DATA.value]:
                    raise ValueError(f"Imported curves are not of type GH-cuts or theta-cuts : {curve_type}")
        if ax is None:
            fig, ax = plt.subplots()
        if len(self.cuts) == 1 and not optimized_on_crab:
            self.cuts[0].plot_cuts_info_plt(ax)

        # ...rest of the function unchanged (optimized_on_crab branch, plotting, export, etc.)...
        if optimized_on_crab:
            # print('Cuts omptimized on Crab')
            cuts_h5_file = f"{self.dl2_processed_dir}/crab_optimized_cuts_{len(self.DL2_files)}_files{output_suffix}.csv"
            cuts = self.read_cuts_optimized_on_crab_from_csv(cuts_h5_file)
            gcuts_even, tcuts_even, E_bins = cuts["even"]["gammaness"], np.sqrt(cuts["even"]["theta2"]), cuts["even"]["E_bins"]
            gcuts_odd, tcuts_odd = cuts["odd"]["gammaness"], np.sqrt(cuts["odd"]["theta2"])
            # E_center = (E_bins[:-1] + E_bins[1:]) / 2
            on_count = np.zeros(len(E_bins) - 1)
            off_count = np.zeros(len(E_bins) - 1)
            t_eff = self.effective_time
            # t_elapsed = 0 * u.h
            # for dl2 in self.dl2s:
            #     _t_eff, _t_elapsed = self.compute_eff_time(dl2)
            #     t_eff += _t_eff
            #     t_elapsed += _t_elapsed

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
                results = list(executor.map(mask_events, zip(self.reco_directions, self.pointings, self.dl2s)))

            even_dl2s, odd_dl2s, even_recos, odd_recos, even_pointings, odd_pointings = zip(*results)
            # Prepare per-file gammaness/theta cuts for even and odd events
            # For each dl2, create boolean masks for even and odd events using the optimized gammaness cuts
            # Build a single boolean mask for even and odd events for each dl2


            def process_file(args):
                even_reco_direction, even_pointing_direction, even_dl2, odd_reco_direction, odd_pointing_direction, odd_dl2, even_gcuts, odd_gcuts, even_theta_cuts, odd_theta_cuts = args
                # EVEN
                # even_t_eff_temp, even_t_elapsed_temp = self.compute_eff_time(even_dl2)
                even_on_count_arr = np.zeros(len(E_bins) - 1)
                even_off_count_arr = np.zeros(len(E_bins) - 1)
                for j, E_min, E_max, gcut, Theta_cut in zip(
                    range(len(E_bins) - 1), E_bins[:-1], E_bins[1:], odd_gcuts, odd_theta_cuts # Apply odd cuts to even events (as per the optimization on Crab)
                ):
                    on_count_temp, off_count_temp, _, _, _ = self.compute_on_off_counts(
                        even_dl2,
                        even_reco_direction,
                        even_pointing_direction,
                        n_off=n_off,
                        theta2_cut=(Theta_cut**2) * u.deg**2,
                        gcut=gcut,
                        E_min=E_min,
                        E_max=E_max,
                        I_min=None,
                        I_max=None,
                    )
                    even_on_count_arr[j] = on_count_temp
                    even_off_count_arr[j] = off_count_temp / n_off
                # ODD
                # odd_t_eff_temp, odd_t_elapsed_temp = self.compute_eff_time(odd_dl2)
                odd_on_count_arr = np.zeros(len(E_bins) - 1)
                odd_off_count_arr = np.zeros(len(E_bins) - 1)
                for j, E_min, E_max, gcut, Theta_cut in zip(
                    range(len(E_bins) - 1), E_bins[:-1], E_bins[1:], even_gcuts, even_theta_cuts # Apply even cuts to odd events (as per the optimization on Crab)
                ):
                    on_count_temp, off_count_temp, _, _, _ = self.compute_on_off_counts(
                        odd_dl2,
                        odd_reco_direction,
                        odd_pointing_direction,
                        n_off=n_off,
                        theta2_cut=(Theta_cut**2) * u.deg**2,
                        gcut=gcut,
                        E_min=E_min,
                        E_max=E_max,
                        I_min=None,
                        I_max=None,
                    )
                    odd_on_count_arr[j] = on_count_temp
                    odd_off_count_arr[j] = off_count_temp / n_off


                return even_on_count_arr + odd_on_count_arr, even_off_count_arr + odd_off_count_arr#, (even_t_eff_temp + odd_t_eff_temp) / 2, (even_t_elapsed_temp + odd_t_elapsed_temp) / 2

            file_args = list(zip(even_recos, even_pointings, even_dl2s, 
                                 odd_recos, odd_pointings, odd_dl2s, 
                                 [gcuts_even] * len(self.DL2_files), [gcuts_odd] * len(self.DL2_files),
                                 [tcuts_even] * len(self.DL2_files), [tcuts_odd] * len(self.DL2_files)))
            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for result in tqdm(executor.map(process_file, file_args), total=len(file_args), desc="Computing sensitivity [Optimized on Crab]"):
                    results.append(result)

            for r in results:
                on_count += r[0]
                off_count += r[1]
                # t_eff += r[2]
                # t_elapsed += r[3]
            print(t_eff)

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
                cond=False,
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
                cond=False,
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
                cond=False,
            )
            # mask = np.ones(len(flux_factor), dtype=bool)
            # Mask where flux_minus is negative or -inf, and handle upper limits
            mask = np.isfinite(flux_minus) & np.isfinite(flux_plus) & np.isfinite(flux_factor)
            E = (E_bins[:-1] + E_bins[1:]) / 2
            ax.plot(
                E[mask],
                flux_factor[mask] * 100,
                marker="o",
                label="Optimized on Crab",
                zorder=10,
                ls="--",
            )
            ax.fill_between(
                E[mask],
                flux_minus[mask] * 100,
                flux_plus[mask] * 100,
                alpha=0.2,
                zorder=0,
                edgecolor="none"
            )
            # ax.text(
            #     0.98,
            #     0.02,
            #     "Optimized on Crab",
            #     transform=ax.transAxes,
            #     fontsize=9,
            #     color=get_color("on_surface"),
            #     verticalalignment="bottom",
            #     horizontalalignment="right",
            #     bbox=dict(
            #         boxstyle="round,pad=0.3",
            #         edgecolor="none",
            #         facecolor=get_color("surface"),
            #         alpha=0.2,
            #     ),
            # )
            
            # export_curves.add_curve(
            #     E[mask],
            #     flux_factor[mask] * 100,
            #     CurveType.SENSITIVITY_DATA,
            #     cuts=cut,
            # )



        # if not optimized_on_crab:
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
            t_eff = self.effective_time
            t_elapsed = 0 * u.h

            def process_file(args):
                reco_direction, pointing_direction, dl2, cuts_mask, theta_cuts = args
                cuts_mask = cuts_mask[i]
                reco_direction = reco_direction[cuts_mask]
                pointing_direction = pointing_direction[cuts_mask]
                # t_eff_temp, t_elapsed_temp = self.compute_eff_time(dl2)
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
                return on_count_arr, off_count_arr,#t_eff_temp, t_elapsed_temp

            file_args = list(zip(self.reco_directions, self.pointings, self.dl2s, self.cuts_masks_gammaness_only, Theta_cuts))
            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for result in tqdm(executor.map(process_file, file_args), total=len(file_args), desc=f"Computing sensitivity [{cut.get_label()}]"):
                    results.append(result)

            for r in results:
                on_count += r[0]
                off_count += r[1]
                # t_eff += r[2]
                # t_elapsed += r[3]

            nexcess = on_count - off_count

            min_signi = 3
            min_exc = 0.002
            min_off_events = 10
            backg_syst = 0.01
            obs_time = 50.0 * u.h

            flux_factor, lima_signi = calc_flux_for_N_sigma(
                N_sigma = 5,
                cumul_excess = nexcess,
                cumul_off = off_count,
                min_signi = min_signi,
                min_exc = min_exc,
                min_off_events = min_off_events,
                alpha = 1,
                target_obs_time = obs_time,
                actual_obs_time = t_eff,
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
            # mask = np.ones(len(flux_factor), dtype=bool)
            # Mask where flux_minus is negative or -inf, and handle upper limits
            mask = np.isfinite(flux_minus) & np.isfinite(flux_plus) & np.isfinite(flux_factor)
            E = (E_bins[:-1] + E_bins[1:]) / 2

            if len(self.cuts) > 1 or optimized_on_crab:
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

        if optimized_on_crab:
            def plot_perf_peper_sensitivity_src_indep_percentage_errors_sys(ax):
                import importlib.resources as pkg_resources

                from .. import resources
                with pkg_resources.path(resources, "sensitivity_src_indep.txt") as text_path:
                    data = np.loadtxt(text_path)

                energy_med = data[:,0] 
                sensitivity = data[:,1]
                sensitivity_error_down = data[:,4]
                sensitivity_error_up = data[:,5]
                energy_init = 0.03
                energy_cut = 15

                
                # print(list(zip(energy_med[(sensitivity > 0) & (energy_med > energy_init) & (energy_med < energy_cut)],
                #             sensitivity[(sensitivity > 0) & (energy_med > energy_init) & (energy_med < energy_cut)])))

                
                band = ax.fill_between(energy_med[(sensitivity > 0) & (energy_med > energy_init) & (energy_med < energy_cut)], 
                        sensitivity_error_up[(sensitivity > 0) & (energy_med > energy_init) & (energy_med < energy_cut)], 
                        sensitivity_error_down[(sensitivity > 0) & (energy_med > energy_init) & (energy_med < energy_cut)],  
                        lw=0, color='C6', alpha=0.3)  
                line, = ax.loglog(energy_med[(sensitivity > 0) & (energy_med > energy_init) & (energy_med < energy_cut)], 
                        sensitivity[(sensitivity > 0) & (energy_med > energy_init) & (energy_med < energy_cut)], 
                        label = 'LST-1 (src-independent)', color = 'C6', ls='solid')  
                
                return band, line
            def plot_perf_paper_sensitivity_src_indep_percentage_without_5percentbg_errors_sys(ax):

                import importlib.resources as pkg_resources

                from .. import resources
                with pkg_resources.path(resources, "sensitivity_src_indep_without_5percentbg.txt") as text_path:
                    data = np.loadtxt(text_path)

                energy_med = data[:,0] 
                sensitivity = data[:,1]
                sensitivity_error_down = data[:,4]
                sensitivity_error_up = data[:,5]
                

                energy_init = 0.03
                energy_cut = 0.15
                    
                band = ax.fill_between(energy_med[(sensitivity > 0) & (energy_med > energy_init) & (energy_med < energy_cut)], 
                        sensitivity_error_up[(sensitivity > 0) & (energy_med > energy_init) & (energy_med < energy_cut)], 
                        sensitivity_error_down[(sensitivity > 0) & (energy_med > energy_init) & (energy_med < energy_cut)],  
                        label = 'LST-1 (src-independent) - without 5% background',  lw=0, facecolor='C6', alpha=0.3, hatch='o')  

                line, = ax.loglog(energy_med[(sensitivity > 0) & (energy_med > energy_init) & (energy_med < energy_cut)], 
                        sensitivity[(sensitivity > 0) & (energy_med > energy_init) & (energy_med < energy_cut)], 
                        color = 'C6', ls='dashed')  
                
                return band, line
            plot_perf_peper_sensitivity_src_indep_percentage_errors_sys(ax)
            plot_perf_paper_sensitivity_src_indep_percentage_without_5percentbg_errors_sys(ax)

        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Reco Energy [TeV]")
        ax.set_ylabel("Differential sensitivity [% Obs. Flux.]")
        # Add ticks 2, 3, 4 to the existing y-ticks
        # yticks = list(ax.get_yticks())
        # for tick in [2, 3, 4]:
        #     if tick not in yticks:
        #         yticks.append(tick)
        # yticks = sorted(yticks)
        # ax.set_yticks(yticks)
        # ax.set_yticklabels([str(int(tick)) if tick in [2, 3, 4] else str(tick) for tick in yticks])
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

    def plot_PSF(self, n_off=5, ax=None, label="CTLearn", output_file=None, plot_MC: list[str]=[], export_to_h5: str=None,
        import_from_h5: str = None,
        import_label: str = None, ylim=(0, 0.6)):
        import concurrent.futures

        import matplotlib.pyplot as plt

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
                # t_eff_temp, t_elapsed_temp = self.compute_eff_time(dl2)
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
                return h_on_file, h_off_file#, t_eff_temp, t_elapsed_temp

            file_args = list(zip(self.reco_directions, self.pointings, self.dl2s, self.cuts_masks_gammaness_only, Theta_cuts))
            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for result in tqdm(executor.map(process_file, file_args), total=len(file_args), desc=f"Computing PSF [{cut.get_label()}]"):
                    results.append(result)

            for r in results:
                h_on += r[0]
                h_off += r[1]
                # t_eff += r[2]
                # t_elapsed += r[3]

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
        self, n_off=5, axs=None, label="CTLearn", output_file=None
    ):
        gammaness_cuts = np.arange(0, 1.05, 0.05)
        import matplotlib.pyplot as plt

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
            print(I_g_off_counts_tot[i])
            print(I_g_on_counts_tot[i])

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

    def plot_excess_vs_background_rates(self, n_off=5, output_file=None):
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
            t_eff, _ = compute_eff_time(dl2, self.CTLearn, self.time_key)
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
        self, n_off=5, output_file=None, cuts_index=0
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
                t_eff_temp, _ = compute_eff_time(dl2, self.CTLearn, self.time_key)
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
