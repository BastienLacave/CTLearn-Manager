from ..tri_model import CTLearnTriModelManager
from ..io.io import load_DL2_data
from ..utils.utils import set_mpl_style
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, concatenate
import astropy.units as u
import numpy as np
from pyirf.statistics import li_ma_significance
from astropy.coordinates import Angle
import pickle
import os


class DL2DataProcessor():
    """
    A class to process DL2 data and perform various analyses such as plotting theta^2 distributions and computing on-off counts.
    Attributes:
    -----------
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
    Methods:
    --------
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
    
    def __init__(self, DL2_files, CTLearnTriModelManager: CTLearnTriModelManager, gammaness_cut=0.9, source_position=SkyCoord.from_name("Crab"), dl2_processed_dir=None):
        self.DL2_files = DL2_files
        self.CTLearnTriModelManager = CTLearnTriModelManager
        self.source_position = source_position
        self.dl2_processed_dir = dl2_processed_dir
        self_telscope_names = CTLearnTriModelManager.telescope_names
        self.stereo = CTLearnTriModelManager.stereo
        self.gammaness_cut = gammaness_cut
        self.reconstruction_method = "CTLearn"
        self.reco_field_suffix = self.reconstruction_method if self.stereo else f"{self.reconstruction_method}_tel"
        self.telescope_id = CTLearnTriModelManager.telescope_ids[0] if self.stereo else CTLearnTriModelManager.telescope_ids

        if any("LST" in name and "1" in name for name in self_telscope_names):
            print("LST1 is in the telescope names")
            self.telescope_location = EarthLocation(
            lon=-17.89149701 * u.deg,
            lat=28.76152611 * u.deg,
            # height of central pin + distance from pin to elevation axis
            height=2184 * u.m + 15.883 * u.m
            )
        
        self.process_DL2_data()

        

    def process_DL2_data(self):
        
        print(f"Preprocessing DL2 data...")
        self.reco_directions = []
        self.pointings = []
        self.dl2s = []
        self.dl2s_cuts = []
        for DL2_file in self.DL2_files:
            if self.dl2_processed_dir is None:
                output_file = DL2_file.replace('.h5', '_dl2_processed.pkl')
            else:
                output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_dl2_processed.pkl'))

            if not os.path.exists(output_file):
                print(f"Loading {DL2_file}")
                dl2 = load_DL2_data(DL2_file, self)
                with open(output_file, 'wb') as f:
                    pickle.dump(dl2, f)
                print(f"Saved processed DL2 data to {output_file}")
            else:
                print(f"Loading processed DL2 data from {output_file}")
                with open(output_file, 'rb') as f:
                    dl2 = pickle.load(f)
            dl2_cuts = dl2[dl2[f"{self.reco_field_suffix}_prediction"] > self.gammaness_cut]
            print(f"{len(dl2_cuts)} events after cuts")
            print("Computing sky positions...")
            times = dl2_cuts["time"]
            # times = Time(np.array(dl2["time"]), format='mjd', scale='tai')
            frame = AltAz(obstime=times, location=self.telescope_location, pressure=100*u.hPa, temperature=20*u.deg_C, relative_humidity=0.1)
            reco_temp = SkyCoord(alt=dl2_cuts[f"{self.reco_field_suffix}_alt"], az=dl2_cuts[f"{self.reco_field_suffix}_az"], frame=frame)#, obstime=dl2_cuts["time"])
            pointing_temp = SkyCoord(alt=dl2_cuts["altitude"], az=dl2_cuts["azimuth"], frame=frame)#, obstime=dl2_cuts["time"])
            self.reco_directions.append(reco_temp.transform_to(self.source_position))
            self.pointings.append(pointing_temp.transform_to(self.source_position))
            self.dl2s.append(dl2)
            self.dl2s_cuts.append(dl2_cuts)

            


    def plot_theta2_distribution(self, bins, n_off=3):
        import matplotlib.pyplot as plt
        set_mpl_style()
        on_count_tot = 0 #np.zeros(len(gammaness_cuts_CTL))
        off_count_tot = 0 #np.zeros(len(gammaness_cuts_CTL))
        angle2_bins = np.linspace(0, 0.4, bins)
        angle2_center = (angle2_bins[:-1] + angle2_bins[1:])/2
        h_on = np.zeros(bins-1)
        h_off = np.zeros(bins-1)
        t_eff = 0 * u.h
        t_elapsed = 0 * u.h
        print("Computing on-off counts...")
        for reco_direction, pointing_direction, dl2 in zip(self.reco_directions, self.pointings, self.dl2s_cuts):
            (
                on_count_temp,
                off_count_temp, 
                on_separation_temp, 
                all_off_separation_temp, 
                significance_lima_temp
                ) = self.compute_on_off_counts( 
                dl2, 
                reco_direction, 
                pointing_direction, 
                n_off=n_off, 
                theta2_cut=0.04*u.deg**2, 
                gcut=self.gammaness_cut, 
                E_min=0, 
                E_max=100, 
                I_min=None, 
                I_max=None
            )
            # print(on_separation_temp)
            # print(all_off_separation_temp)
            on_count_tot += on_count_temp
            off_count_tot += off_count_temp
            h_on_temp, _ = np.histogram(on_separation_temp.to(u.deg).value**2, bins=angle2_bins)
            h_off_temp, _ = np.histogram(all_off_separation_temp.to(u.deg).value**2, bins=angle2_bins)
            h_on += h_on_temp
            h_off += h_off_temp / n_off # To plot the average off source counts


            t_eff_temp, t_elapsed_temp = self.compute_eff_time(dl2)
            t_eff += t_eff_temp
            t_elapsed += t_elapsed_temp

        # t_eff = 2 * u.h
        lima_signi = li_ma_significance(np.float64(on_count_tot), 
                                            np.float64(off_count_tot), 
                                            alpha=1/n_off)
        fig, ax = plt.subplots()
        label = "$t_{eff}$ = "+f"{t_eff.to(u.h):.2f}"+"\n$N_{on}$ = "+f"{on_count_tot} "+"\n$N_{off}$ = "+f"{(off_count_tot/n_off):.2f}"+"\n$N_{excess}$ = "+f"{(on_count_tot - off_count_tot/n_off):.2f}"+" \n$\sigma_{Li&Ma}$ = "+f"{lima_signi:.2f}"
        props = dict(boxstyle='round', facecolor='none', alpha=0.95, edgecolor='k')
        txt = plt.text(0.12, 0.96, label, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props, color="k")
        plt.plot(angle2_center, h_on, label='on source')
        plt.plot(angle2_center, h_off, label='off source', zorder=0)
        plt.fill_between(angle2_center, 
                        h_on - np.sqrt(h_on), 
                        h_on + np.sqrt(h_on), 
                        alpha=0.3, zorder=1)
        plt.fill_between(angle2_center,
                            h_off - np.sqrt(h_off),
                            h_off + np.sqrt(h_off),
                            alpha=0.3, zorder=1)
        plt.xlim(0, 0.4)
        plt.axvline(0.04, color='black', linestyle='--')
        plt.text(0.1, 0.8, '0.2° radius', color='black', fontsize=14, rotation=90, transform=ax.transAxes, ha='right', va='center')
        # plt.text(0.045, on_count_CTL[np.where(angle2_center < 0.04)[0][-1]], 'on source', color=colors[0], fontsize=14, ha='left', va='bottom')
        # plt.text(0.045, off_count_CTL[np.where(angle2_center < 0.04)[0][-1]]/3 - 100, 'off source', color=colors[1], fontsize=14, ha='left', va='top')
        plt.legend()
        plt.xlabel(r'Separation [deg$^2$]')
        plt.ylabel('Counts')
        plt.title('LST-1 Crab Nebula with CTLearn')
        # plt.yscale('log')
        plt.show()



    def compute_off_regions(self, pointing, n_off):
        center = pointing # SkyCoord(ra=10*u.degree, dec=20*u.degree)
        # ra_axis = pointing.directional_offset_by(0, 0.5*u.deg)
        # source = self.source_position #SkyCoord(ra=11*u.degree, dec=20*u.degree)
        angle_source = pointing.position_angle(self.source_position)
        radius = center.separation(self.source_position)
        angles = np.linspace(0, 2*np.pi, n_off+1, endpoint=False)# + np.pi/(n_off)

        new_ra = []
        new_dec = []
        for angle in angles:    
            position_off = center.directional_offset_by(angle_source + Angle(angle, 'rad'), radius)
            new_ra.append(position_off.ra.degree)
            new_dec.append(position_off.dec.degree)
        off_regions =  SkyCoord(ra=new_ra[1:]*u.degree, dec=new_dec[1:]*u.degree)
        return off_regions
    
    def compute_eff_time(self, events): 
        timestamp = np.array(events["time"].to_value('unix'))
        delta_t = np.array(events["delta_t"])
        delta_t = [dt.to_value('sec') for dt in delta_t]

        if not isinstance(timestamp, u.Quantity):
            timestamp *= u.s
        if not isinstance(delta_t, u.Quantity):
            delta_t *= u.s

        # time differences between the events in the table (which in general are
        # NOT all triggered events):
        time_diff = np.diff(timestamp)

        # elapsed time: sum of those time differences, excluding large ones which
        # might indicate the DAQ was stopped (e.g. if the table contains more
        # than one run). We set 0.01 s as limit to decide a "break" occurred:
        t_elapsed = np.sum(time_diff[time_diff < 0.01 * u.s])

        # delta_t is the time elapsed since the previous triggered event.
        # We exclude the null values that might be set for the first even in a file.
        # Same as the elapsed time, we exclude events with delta_t larger than 0.01 s.
        delta_t = delta_t[
            (delta_t > 0.0 * u.s) & (delta_t < 0.01 * u.s)
        ]

        # dead time per event (minimum observed delta_t, ):
        dead_time = np.amin(delta_t)


        rate = 1 / (np.mean(delta_t) - dead_time)

        t_eff = t_elapsed / (1 + rate * dead_time)
        # print(t_eff.to(u.h))   
        # with open(f"/home/bastien.lacave/PhD/Analysis/DL2_Processing/times/{run}_t_eff.pkl", "wb") as f:
        #     pickle.dump(t_eff, f)
        return t_eff, t_elapsed

    def compute_on_off_counts(self, events, reco_coord, pointing_coord, n_off, theta2_cut=0.04*u.deg**2, gcut=0.5, E_min=0, E_max=100, I_min=None, I_max=None):
        if I_min == None or I_max == None:
            mask = (events[f"{self.reco_field_suffix}_energy"] > E_min) & (events[f"{self.reco_field_suffix}_energy"] < E_max) & (events[f"{self.reco_field_suffix}_prediction"] > gcut)
        # else:
        #     mask = (events['intensity'] > I_min) & (events['intensity'] < I_max) & (events[f"{self.reco_field_suffix}_prediction"] > gcut)


        # ON
        on_separation = reco_coord.separation(self.source_position)[mask]
        on_count = len(on_separation[on_separation < np.sqrt(theta2_cut)])
        # sum_norm_on = len(on_separation[(on_separation > norm_theta[0]) & (on_separation < norm_theta[1])])

        # OFF
        off_regions = self.compute_off_regions(pointing_coord, n_off)
        off_count = 0
        # sum_norm_off = 0
        all_off_separation = []
        for i in range(n_off):
            off_separation = reco_coord.separation(off_regions[i])[mask]
            all_off_separation.append(off_separation)
            off_count += len(off_separation[off_separation < np.sqrt(theta2_cut)])
            # sum_norm_off += len(off_separation[(off_separation > norm_theta[0]) & (off_separation < norm_theta[1])])
        all_off_separation = np.array(all_off_separation).flatten() * u.deg

        # alpha = sum_norm_on / sum_norm_off
        alpha = 1/n_off
        # stat = WStatCountsStatistic(n_on=on_count, n_off=off_count, alpha=alpha)
        # significance_lima = stat.sqrt_ts
        significance_lima = li_ma_significance(on_count, off_count, alpha)
        # print(f"Significance: {significance_lima:.2f}")
        # N_excess = on_count - alpha*off_count

        return on_count, off_count, on_separation, all_off_separation, significance_lima