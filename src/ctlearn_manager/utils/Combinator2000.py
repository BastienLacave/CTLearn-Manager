from ctlearn_manager.utils.utils import ClusterConfiguration, set_mpl_style
from ctlearn_manager.utils.DL2_processing import DL2DataProcessor
from astropy.table import (join, hstack, vstack)
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, concatenate
from tqdm import tqdm
import astropy.units as u

class Combinator2000(DL2DataProcessor):


    def __init__(self, dl2_processors, direction_index, energy_index, type_index, gammaness_cut=0.9):

        self.gammaness_cut = gammaness_cut
        self.direction_processor = dl2_processors[direction_index]
        self.energy_processor = dl2_processors[energy_index]
        self.type_processor = dl2_processors[type_index]
        # self.cluster_configuration = self.direction_processor.cluster_configuration
        self.telscope_names = self.direction_processor.telscope_names
        self.source_position = self.direction_processor.source_position
        if any("LST" in name and "1" in name for name in self.telscope_names):
            # print("LST1 is in the telescope names")
            self.telescope_location = EarthLocation(
            lon=-17.89149701 * u.deg,
            lat=28.76152611 * u.deg,
            # height of central pin + distance from pin to elevation axis
            height=2184 * u.m + 15.883 * u.m
            )
        self.direction_dl2s = self.direction_processor.dl2s
        self.energy_dl2s = self.energy_processor.dl2s
        self.type_dl2s = self.type_processor.dl2s
        if len(self.direction_dl2s) > 1:
            self.direction_dl2 = vstack(self.direction_dl2s)
        else:
            self.direction_dl2 = self.direction_dl2s[0]
        if len(self.energy_dl2s) > 1:
            self.energy_dl2 = vstack(self.energy_dl2s)
        else:
            self.energy_dl2 = self.energy_dl2s[0]
        if len(self.type_dl2s) > 1:
            self.type_dl2 = vstack(self.type_dl2s)
        else:
            self.type_dl2 = self.type_dl2s[0]

        self.ref_dl2 = self.direction_dl2.copy()
        
        self.dl2s = join(self.ref_dl2, 
                 self.energy_dl2[["obs_id", "event_id", self.energy_processor.energy_key]], 
                 keys=["obs_id", "event_id"], join_type='left')
        self.dl2s = [join(self.dl2s, 
                 self.type_dl2[["obs_id", "event_id", self.type_processor.gammaness_key]], 
                 keys=["obs_id", "event_id"], join_type='left')]
        for dl2 in self.dl2s:
            dl2.sort('time')


        set_mpl_style()
        self.set_keys()
        self.reco_directions = []
        self.pointings = []
        self.cuts_masks = []
        for DL2 in tqdm(self.dl2s, desc="Computing sky positions"):
            cut_mask = DL2[self.type_processor.gammaness_key] > self.gammaness_cut
            self.cuts_masks.append(cut_mask)
            times = DL2[self.time_key]

            frame = AltAz(obstime=times, location=self.telescope_location, pressure=100*u.hPa, temperature=20*u.deg_C, relative_humidity=0.1)
            reco_temp = SkyCoord(alt=DL2[self.reco_alt_key], az=DL2[self.reco_az_key], frame=frame)#, obstime=DL2["time"])
            pointing_temp = SkyCoord(alt=DL2[self.pointing_alt_key], az=DL2[self.pointing_az_key], frame=frame)#, obstime=dl2["time"])
            transformed_reco = reco_temp.transform_to(self.source_position)
            transformed_pointing = pointing_temp.transform_to(self.source_position)

            self.reco_directions.append(transformed_reco)
            self.pointings.append(transformed_pointing)
        # cut_mask = dl2[self.gammaness_key] > self.gammaness_cut

    def set_keys(self):
        self.gammaness_key = f"{self.type_processor.reco_field_suffix}_prediction" #if self.CTLearn else "gammaness"
        self.energy_key = f"{self.energy_processor.reco_field_suffix}_energy" #if self.CTLearn else "reco_energy"
        self.intensity_key = "hillas_intensity" #if self.CTLearn else "intensity"
        self.reco_alt_key = f"{self.direction_processor.reco_field_suffix}_alt" #if self.CTLearn else "reco_alt"
        self.reco_az_key = f"{self.direction_processor.reco_field_suffix}_az" #if self.CTLearn else "reco_az"
        self.pointing_alt_key = "altitude" #if self.CTLearn else "alt_tel"
        self.pointing_az_key = "azimuth" #if self.CTLearn else "az_tel"
        self.time_key = "time" #if self.CTLearn else "dragon_time"
