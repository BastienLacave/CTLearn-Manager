import glob

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from tqdm import tqdm

from ctlearn_manager.utils import Cuts, DefaultCuts, CutType
from ctlearn_manager.utils.DL2_processing import DL2DataProcessor
from ctlearn_manager.utils.utils import ClusterConfiguration, set_mpl_style

from .. import TriModelCollection
from ..tri_model import CTLearnTriModelManager


class RFCounterpart(DL2DataProcessor):
    def __init__(
        self,
        dl2_processed_dir,
        CTLearn_TriModel_Manager: CTLearnTriModelManager or TriModelCollection,
        DL2_files=None,
        runs=None,
        # cluster_configuration=ClusterConfiguration(),
        cuts: list[Cuts] = [DefaultCuts.GH_0_9.value],
        source_position=SkyCoord.from_name("Crab"),
        default_E_bins=np.logspace(
            np.log10(0.02), np.log10(20), int((np.log10(20) - np.log10(0.02)) * 5 + 1)
        )
        * u.TeV,
    ):
        if isinstance(CTLearn_TriModel_Manager, CTLearnTriModelManager):
            self.CTLearnTriModelCollection = TriModelCollection(
                [CTLearn_TriModel_Manager],
                cluster_configuration=CTLearn_TriModel_Manager.cluster_configuration,
            )
        else:
            self.CTLearnTriModelCollection = CTLearn_TriModel_Manager
        self.cluster_configuration = self.CTLearnTriModelCollection.cluster_configuration

        if (
            (self.cluster_configuration.cluster == "lst-cluster")
            and (runs is not None)
            and (DL2_files is None)
        ):
            self.DL2_files = []
            for run in tqdm(runs, desc="Fetching DL2 files from runs"):
                DL2_file_run = glob.glob(
                    f"/fefs/aswg/data/real/DL2/*/v0.*/tailcut*/nsb_tuning_*/dl2_LST-1.Run{run:05d}.h5"
                )[0]
                self.DL2_files.append(DL2_file_run)
                # self.DL2_files.append(f"/fefs/aswg/data/real/DL2/20220331/v0.10/tailcut84/nsb_tuning_0.14/dl2_LST-1.Run{run}.h5")
        elif DL2_files is not None:
            self.DL2_files = DL2_files
        else:
            raise ValueError(
                "Either DL2_files or runs must be provided, if runs are provided, be sure to run on the LST cluster."
            )
        self.DL2_files = np.sort(self.DL2_files)

        # self.CTLearnTriModelManager = CTLearnTriModelManager
        # self.pointing_table = pointing_table
        self.source_position = source_position
        self.dl2_processed_dir = dl2_processed_dir
        self.telscope_names = self.CTLearnTriModelCollection.tri_models[
            0
        ].telescope_names
        self.stereo = self.CTLearnTriModelCollection.tri_models[0].stereo
        # self.stereo = CTLearnTriModelManager.stereo
        self.cuts = cuts
        # self.gammaness_cut = gammaness_cut
        self.reconstruction_method = "RF"
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
        self.CTLearn = False
        E_bins_tot = np.empty(len(self.cuts), dtype=object)
        GH_cuts_tot = np.empty(len(self.cuts), dtype=object)
        Theta_cuts_tot = np.empty(len(self.cuts), dtype=object)
        for i, cut in enumerate(self.cuts):
            if (
                cut.cut_type == CutType.EFFICIENCY_OPTIMIZED
                or cut.cut_type == CutType.SENSITIVITY_OPTIMIZED
            ):
                # print(self.edep_cuts)
                # get E bins from IRFs cuts file
                cuts_file = self.CTLearnTriModelCollection.tri_models[
                    0
                ].direction_model.get_IRF_data()[
                    1
                ][
                    0
                ]  # index 1 is the cut file table, index 0 is the first line (the path to the file)
                with fits.open(cuts_file, mode="readonly") as hdul:
                    E_bins = hdul["GH_CUTS"].data["low"]
                    E_bins = np.append(E_bins, hdul["GH_CUTS"].data["high"][-1]) * u.TeV
                    E_bins_tot[i] = E_bins

                    GH_cuts = hdul["GH_CUTS"].data["cut"]
                    GH_cuts_tot[i] = GH_cuts

                    Theta_cuts = hdul["RAD_MAX"].data["cut"]
                    Theta_cuts_tot[i] = Theta_cuts
            else:
                E_bins = default_E_bins
                E_bins_tot[i] = E_bins
        self.E_bins = E_bins_tot
        self.GH_cuts = GH_cuts_tot
        self.Theta_cuts = Theta_cuts_tot
        self.set_keys()

        if any("LST" in name and "1" in name for name in self.telscope_names):
            # print("LST1 is in the telescope names")
            self.telescope_location = EarthLocation(
                lon=-17.89149701 * u.deg,
                lat=28.76152611 * u.deg,
                # height of central pin + distance from pin to elevation axis
                height=2184 * u.m + 15.883 * u.m,
            )
        self.process_DL2_data()
        self.load_processed_data()
        set_mpl_style()


class LazyRFCounterpart(DL2DataProcessor):
    def __init__(
        self,
        DL2DataProcessor: DL2DataProcessor,
        dl2_processed_dir,
        gammaness_cut=0.9,
        lstchain_version="0.10",
    ):
        self.cluster_configuration = (
            DL2DataProcessor.CTLearnTriModelManager.cluster_configuration
        )
        self.lstchain_version = lstchain_version

        runs = []

        for dl2 in DL2DataProcessor.dl2s:
            for obs_id in dl2["obs_id"]:
                run_temp = int(str(obs_id)[:4])
                if run_temp not in runs:
                    runs.append(run_temp)

        print(runs)

        if (self.cluster_configuration.cluster == "lst-cluster") and (runs is not None):
            self.DL2_files = []
            for run in runs:
                # try:
                DL2_file_run = glob.glob(
                    f"/fefs/aswg/data/real/DL2/*/v{self.lstchain_version}/tailcut*/nsb_tuning_*/dl2_LST-1.Run{run:05d}.h5"
                )[0]
                # except:
                #     DL2_file_run = glob.glob(f"/fefs/aswg/data/real/DL2/*/v0.9/tailcut*/nsb_tuning_*/dl2_LST-1.Run{run:05d}.h5")[0]
                self.DL2_files.append(DL2_file_run)
                # self.DL2_files.append(f"/fefs/aswg/data/real/DL2/20220331/v0.10/tailcut84/nsb_tuning_0.14/dl2_LST-1.Run{run}.h5")
        else:
            raise ValueError(
                "Either DL2_files or runs must be provided, if runs are provided, be sure to run on the LST cluster."
            )

        self.CTLearnTriModelManager = DL2DataProcessor.CTLearnTriModelManager
        self.source_position = DL2DataProcessor.source_position
        self.dl2_processed_dir = dl2_processed_dir
        self_telscope_names = DL2DataProcessor.CTLearnTriModelManager.telescope_names
        self.stereo = DL2DataProcessor.CTLearnTriModelManager.stereo
        self.gammaness_cut = gammaness_cut
        self.reconstruction_method = "RF"
        self.reco_field_suffix = (
            self.reconstruction_method
            if self.stereo
            else f"{self.reconstruction_method}_tel"
        )
        self.telescope_id = (
            DL2DataProcessor.CTLearnTriModelManager.telescope_ids
            if self.stereo
            else DL2DataProcessor.CTLearnTriModelManager.telescope_ids[0]
        )
        # self.irfs = CTLearnTriModelManager.irfs
        self.CTLearn = False
        self.set_keys()

        if any("LST" in name and "1" in name for name in self_telscope_names):
            # print("LST1 is in the telescope names")
            self.telescope_location = EarthLocation(
                lon=-17.89149701 * u.deg,
                lat=28.76152611 * u.deg,
                # height of central pin + distance from pin to elevation axis
                height=2184 * u.m + 15.883 * u.m,
            )

        self.process_DL2_data()
        self.load_processed_data()
        set_mpl_style()
