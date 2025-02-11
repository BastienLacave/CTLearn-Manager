from ctlearn_manager.utils.utils import ClusterConfiguration, set_mpl_style
from ctlearn_manager.utils.DL2_processing import DL2DataProcessor
from ..tri_model import CTLearnTriModelManager
import glob
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u


class RFCounterpart(DL2DataProcessor):

    def __init__(self, dl2_processed_dir, CTLearnTriModelManager: CTLearnTriModelManager, DL2_files=None, runs=None, cluster_configuration = ClusterConfiguration(), gammaness_cut=0.9, source_position=SkyCoord.from_name("Crab")):
        self.cluster_configuration = cluster_configuration


        if (self.cluster_configuration.cluster == 'lst-cluster') and (runs is not None) and (DL2_files is None):
            self.DL2_files = []
            for run in runs:
                DL2_file_run = glob.glob(f"/fefs/aswg/data/real/DL2/*/v0.*/tailcut*/nsb_tuning_*/dl2_LST-1.Run{run:05d}.h5")[0]
                self.DL2_files.append(DL2_file_run)
                # self.DL2_files.append(f"/fefs/aswg/data/real/DL2/20220331/v0.10/tailcut84/nsb_tuning_0.14/dl2_LST-1.Run{run}.h5")
        elif DL2_files is not None:
            self.DL2_files = DL2_files
        else:
            raise ValueError("Either DL2_files or runs must be provided, if runs are provided, be sure to run on the LST cluster.")

        self.CTLearnTriModelManager = CTLearnTriModelManager
        self.source_position = source_position
        self.dl2_processed_dir = dl2_processed_dir
        self_telscope_names = CTLearnTriModelManager.telescope_names
        self.stereo = CTLearnTriModelManager.stereo
        self.gammaness_cut = gammaness_cut
        self.reconstruction_method = "RF"
        self.reco_field_suffix = self.reconstruction_method if self.stereo else f"{self.reconstruction_method}_tel"
        self.telescope_id = CTLearnTriModelManager.telescope_ids if self.stereo else CTLearnTriModelManager.telescope_ids[0]
        # self.irfs = CTLearnTriModelManager.irfs
        self.CTLearn = False
        self.set_keys()



        if any("LST" in name and "1" in name for name in self_telscope_names):
            # print("LST1 is in the telescope names")
            self.telescope_location = EarthLocation(
            lon=-17.89149701 * u.deg,
            lat=28.76152611 * u.deg,
            # height of central pin + distance from pin to elevation axis
            height=2184 * u.m + 15.883 * u.m
            )
        
        self.process_DL2_data()
        self.load_processed_data()
        set_mpl_style()


class LazyRFCounterpart(DL2DataProcessor):

    def __init__(self, DL2DataProcessor: DL2DataProcessor, dl2_processed_dir, gammaness_cut=0.9):
        self.cluster_configuration = DL2DataProcessor.CTLearnTriModelManager.cluster_configuration



        runs = []

        for dl2 in DL2DataProcessor.dl2s:
            for obs_id in dl2["obs_id"]:
                run_temp = int(str(obs_id)[:4])
                if run_temp not in runs:
                    runs.append(run_temp)

        print(runs)


        if (self.cluster_configuration.cluster == 'lst-cluster') and (runs is not None):
            self.DL2_files = []
            for run in runs:
                DL2_file_run = glob.glob(f"/fefs/aswg/data/real/DL2/*/v0.*/tailcut*/nsb_tuning_*/dl2_LST-1.Run{run:05d}.h5")[0]
                self.DL2_files.append(DL2_file_run)
                # self.DL2_files.append(f"/fefs/aswg/data/real/DL2/20220331/v0.10/tailcut84/nsb_tuning_0.14/dl2_LST-1.Run{run}.h5")
        else:
            raise ValueError("Either DL2_files or runs must be provided, if runs are provided, be sure to run on the LST cluster.")

        self.CTLearnTriModelManager = DL2DataProcessor.CTLearnTriModelManager
        self.source_position = DL2DataProcessor.source_position
        self.dl2_processed_dir = dl2_processed_dir
        self_telscope_names = DL2DataProcessor.CTLearnTriModelManager.telescope_names
        self.stereo = DL2DataProcessor.CTLearnTriModelManager.stereo
        self.gammaness_cut = gammaness_cut
        self.reconstruction_method = "RF"
        self.reco_field_suffix = self.reconstruction_method if self.stereo else f"{self.reconstruction_method}_tel"
        self.telescope_id = DL2DataProcessor.CTLearnTriModelManager.telescope_ids if self.stereo else DL2DataProcessor.CTLearnTriModelManager.telescope_ids[0]
        # self.irfs = CTLearnTriModelManager.irfs
        self.CTLearn = False
        self.set_keys()



        if any("LST" in name and "1" in name for name in self_telscope_names):
            # print("LST1 is in the telescope names")
            self.telescope_location = EarthLocation(
            lon=-17.89149701 * u.deg,
            lat=28.76152611 * u.deg,
            # height of central pin + distance from pin to elevation axis
            height=2184 * u.m + 15.883 * u.m
            )
        
        self.process_DL2_data()
        self.load_processed_data()
        set_mpl_style()

        
            