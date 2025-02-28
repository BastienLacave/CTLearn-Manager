from ..tri_model import CTLearnTriModelManager
from ..io.io import load_DL2_data
from ..utils.utils import set_mpl_style, get_avg_pointing, calc_flux_for_N_sigma, find_68_percent_range
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, concatenate
import astropy.units as u
import numpy as np
from pyirf.statistics import li_ma_significance
from astropy.coordinates import Angle
import pickle
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm


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
    
    def __init__(self, DL2_files, CTLearnTriModelManager: CTLearnTriModelManager, gammaness_cut=0.9, source_position=SkyCoord.from_name("Crab"), dl2_processed_dir=None, pointing_table='dl1/monitoring/telescope/pointing'):
        
        self.DL2_files = DL2_files
        self.CTLearnTriModelManager = CTLearnTriModelManager
        self.source_position = source_position
        self.dl2_processed_dir = dl2_processed_dir
        self.telscope_names = CTLearnTriModelManager.telescope_names
        self.stereo = CTLearnTriModelManager.stereo
        self.gammaness_cut = gammaness_cut
        self.pointing_table = pointing_table
        self.reconstruction_method = "CTLearn"
        self.reco_field_suffix = self.reconstruction_method if self.stereo else f"{self.reconstruction_method}_tel"
        self.telescope_id = CTLearnTriModelManager.telescope_ids if self.stereo else CTLearnTriModelManager.telescope_ids[0]
        # self.irfs = CTLearnTriModelManager.irfs
        self.CTLearn = True
        self.set_keys()
        




        if any("LST" in name and "1" in name for name in self.telscope_names):
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

    def set_keys(self):
        self.gammaness_key = f"{self.reco_field_suffix}_prediction" #if self.CTLearn else "gammaness"
        self.energy_key = f"{self.reco_field_suffix}_energy" #if self.CTLearn else "reco_energy"
        self.intensity_key = "hillas_intensity" #if self.CTLearn else "intensity"
        self.reco_alt_key = f"{self.reco_field_suffix}_alt" #if self.CTLearn else "reco_alt"
        self.reco_az_key = f"{self.reco_field_suffix}_az" #if self.CTLearn else "reco_az"
        self.pointing_alt_key = "altitude" #if self.CTLearn else "alt_tel"
        self.pointing_az_key = "azimuth" #if self.CTLearn else "az_tel"
        self.time_key = "time" #if self.CTLearn else "dragon_time"
    

    def process_DL2_data(self):
        
        print(f"Preprocessing DL2 (~50min/run), only once")
        

        for DL2_file in self.DL2_files:
            if self.dl2_processed_dir is None:
                dl2_output_file = DL2_file.replace('.h5', '_dl2_processed.pkl')
                reco_output_file = DL2_file.replace('.h5', '_reco_directions.pkl')
                pointing_output_file = DL2_file.replace('.h5', '_pointings.pkl')
                I_g_on_counts_output_file = DL2_file.replace('.h5', '_I_g_on_counts.pkl')
                I_g_off_counts_output_file = DL2_file.replace('.h5', '_I_g_off_counts.pkl')
            else:
                dl2_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_dl2_processed.pkl'))
                reco_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_reco_directions.pkl'))
                pointing_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_pointings.pkl'))
                I_g_on_counts_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_I_g_on_counts.pkl'))
                I_g_off_counts_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_I_g_off_counts.pkl'))


            if (not os.path.exists(reco_output_file)) or (not os.path.exists(pointing_output_file)) or (not os.path.exists(dl2_output_file)) or (not os.path.exists(I_g_on_counts_output_file)) or (not os.path.exists(I_g_off_counts_output_file)):
                self.CTLearnTriModelManager.cluster_configuration.info()
                if self.CTLearnTriModelManager.cluster_configuration.use_cluster:
                    processor_file = f"{self.dl2_processed_dir}/{DL2_file.split('/')[-1]}_processor.pkl"
                    with open(processor_file, 'wb') as f:
                        pickle.dump(self, f)
                    self.CTLearnTriModelManager.cluster_configuration.write_sbatch_script(f"process_dl2_{DL2_file.split('/')[-1]}", f"process_dl2_file {DL2_file} {processor_file}", self.dl2_processed_dir)
                    os.system(f"sbatch {self.dl2_processed_dir}/process_dl2_{DL2_file.split('/')[-1]}.sh")
                else:
                    # print(f"[NOT USING SLURM] Processing {DL2_file}")

                    processor_file = f"{self.dl2_processed_dir}/{DL2_file.split('/')[-1]}_processor.pkl"
                    with open(processor_file, 'wb') as f:
                        pickle.dump(self, f)
                    # self.CTLearnTriModelManager.cluster_configuration.write_sbatch_script(f"process_dl2_{DL2_file.split('/')[-1]}", f"process_dl2_file {DL2_file} {processor_file}", self.dl2_processed_dir)
                    # os.system(f"sbatch {self.dl2_processed_dir}/process_dl2_{DL2_file.split('/')[-1]}.sh")
                    print(f"process_dl2_file {DL2_file} {processor_file}")
                    os.system(f"process_dl2_file {DL2_file} {processor_file}")



                    # dl2 = load_DL2_data(DL2_file, self)
                    # dl2 = dl2[dl2[self.gammaness_key] > 0] # Remove unpredicted events
                    # with open(dl2_output_file, 'wb') as f:
                    #     pickle.dump(dl2, f)
                    # print(f"Saved processed DL2 data to {dl2_output_file}")

                    # dl2 = dl2[dl2[self.gammaness_key] > 0] # Remove unpredicted events
                    # cut_mask = dl2[self.gammaness_key] > self.gammaness_cut
                    # dl2_cuts = dl2[cut_mask]
                    # print(f"{len(dl2_cuts)} events after cuts")

                    # print("Computing sky positions...")
                    # times = dl2["time"]
                    # # times = Time(np.array(dl2["time"]), format='mjd', scale='tai')

                    # frame = AltAz(obstime=times, location=self.telescope_location, pressure=100*u.hPa, temperature=20*u.deg_C, relative_humidity=0.1)
                    # reco_temp = SkyCoord(alt=dl2[self.alt_key], az=dl2[self.az_key], frame=frame)#, obstime=dl2["time"])
                    # pointing_temp = SkyCoord(alt=dl2["altitude"], az=dl2["azimuth"], frame=frame)#, obstime=dl2["time"])
                    # transformed_reco = reco_temp.transform_to(self.source_position)
                    # transformed_pointing = pointing_temp.transform_to(self.source_position)

                    # # Convert SkyCoord objects to dictionaries
                    # transformed_reco_dict = {'ra': transformed_reco.ra.deg, 'dec': transformed_reco.dec.deg}
                    # transformed_pointing_dict = {'ra': transformed_pointing.ra.deg, 'dec': transformed_pointing.dec.deg}

                    # with open(reco_output_file, 'wb') as f:
                    #     pickle.dump(transformed_reco_dict, f)
                    # with open(pointing_output_file, 'wb') as f:
                    #     pickle.dump(transformed_pointing_dict, f)

                    # print(f"Saved reco directions to {reco_output_file}")
                    # print(f"Saved pointings to {pointing_output_file}")

    def load_processed_data(self):
        from tqdm import tqdm

        self.reco_directions = []
        self.pointings = []
        self.dl2s = []
        # self.dl2s_cuts = []
        self.cuts_masks = []
        self.I_g_on_counts = []
        self.I_g_off_counts = []

        for DL2_file in tqdm(self.DL2_files, desc="Loading processed data"):
            if self.dl2_processed_dir is None:
                dl2_output_file = DL2_file.replace('.h5', '_dl2_processed.pkl')
                reco_output_file = DL2_file.replace('.h5', '_reco_directions.pkl')
                pointing_output_file = DL2_file.replace('.h5', '_pointings.pkl')
                I_g_on_counts_output_file = DL2_file.replace('.h5', '_I_g_on_counts.pkl')
                I_g_off_counts_output_file = DL2_file.replace('.h5', '_I_g_off_counts.pkl')
            else:
                dl2_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_dl2_processed.pkl'))
                reco_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_reco_directions.pkl'))
                pointing_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_pointings.pkl'))
                I_g_on_counts_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_I_g_on_counts.pkl'))
                I_g_off_counts_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_I_g_off_counts.pkl'))

            if (os.path.exists(reco_output_file)) and (os.path.exists(pointing_output_file)) and (os.path.exists(dl2_output_file)):

                with open(dl2_output_file, 'rb') as f:
                        dl2 = pickle.load(f)
                if self.gammaness_key in dl2.colnames:
                    dl2 = dl2[dl2[self.gammaness_key] > 0] # Remove unpredicted events
                    cut_mask = dl2[self.gammaness_key] > self.gammaness_cut
                else:
                    cut_mask = np.ones(len(dl2), dtype=bool)
                self.cuts_masks.append(cut_mask)
                self.dl2s.append(dl2)
            
            if (os.path.exists(reco_output_file)) and (os.path.exists(pointing_output_file)) and (os.path.exists(dl2_output_file)):
                with open(reco_output_file, 'rb') as f:
                    transformed_reco_dict = pickle.load(f)
                with open(pointing_output_file, 'rb') as f:
                    transformed_pointing_dict = pickle.load(f)

                
                # dl2_cuts = dl2[cut_mask]
                
                # Convert dictionaries back to SkyCoord objects
                transformed_reco = SkyCoord(ra=transformed_reco_dict['ra']*u.deg, dec=transformed_reco_dict['dec']*u.deg, frame=self.source_position)
                transformed_pointing = SkyCoord(ra=transformed_pointing_dict['ra']*u.deg, dec=transformed_pointing_dict['dec']*u.deg, frame=self.source_position)
        
                self.reco_directions.append(transformed_reco)
                self.pointings.append(transformed_pointing)
                

                
                # self.dl2s_cuts.append(dl2_cuts)
            if (os.path.exists(I_g_on_counts_output_file)) and (os.path.exists(I_g_off_counts_output_file)):

                with open(I_g_on_counts_output_file, 'rb') as f:
                    I_g_on_counts = pickle.load(f)
                with open(I_g_off_counts_output_file, 'rb') as f:
                    I_g_off_counts = pickle.load(f)
                
                self.I_g_on_counts.append(I_g_on_counts)
                self.I_g_off_counts.append(I_g_off_counts)



    def plot_theta2_distribution(self, bins, n_off=3):
        import matplotlib.pyplot as plt
        
        on_count_tot = 0 #np.zeros(len(gammaness_cuts))
        off_count_tot = 0 #np.zeros(len(gammaness_cuts))
        angle2_bins = np.linspace(0, 0.4, bins)
        angle2_center = (angle2_bins[:-1] + angle2_bins[1:])/2
        h_on = np.zeros(bins-1)
        h_off = np.zeros(bins-1)
        t_eff = 0 * u.h
        t_elapsed = 0 * u.h
        print("Computing on-off counts...")
        for reco_direction, pointing_direction, dl2, cuts_mask in zip(self.reco_directions, self.pointings, self.dl2s, self.cuts_masks):
            reco_direction = reco_direction[cuts_mask]
            pointing_direction = pointing_direction[cuts_mask]
            dl2 = dl2[cuts_mask]
            (
                on_count_temp,
                off_count_temp, 
                on_separation_temp, 
                all_off_separation_temp, 
                _
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
        label = "$t_{eff}$ = "+f"{t_eff.to(u.h):.2f}"+"\n$N_{on}$ = "+f"{on_count_tot} "+"\n$\overline{N}_{off}$ = "+f"{(off_count_tot/n_off):.1f}"+"\n$N_{excess}$ = "+f"{(on_count_tot - off_count_tot/n_off):.1f}"+" \n$\sigma_{Li&Ma}$ = "+f"{lima_signi:.2f}"
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
        # plt.text(0.045, on_count[np.where(angle2_center < 0.04)[0][-1]], 'on source', color=colors[0], fontsize=14, ha='left', va='bottom')
        # plt.text(0.045, off_count[np.where(angle2_center < 0.04)[0][-1]]/3 - 100, 'off source', color=colors[1], fontsize=14, ha='left', va='top')
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
        if self.CTLearn:
            timestamp = np.array(events[self.time_key].to_value('unix'))
        else:
            timestamp = np.array(events[self.time_key])

        delta_t = np.array(events["delta_t"])
        # delta_t = np.diff(timestamp)
        # delta_t = [dt.to_value('sec') for dt in delta_t]

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
            mask = (events[self.energy_key] > E_min) & (events[self.energy_key] < E_max) & (events[self.gammaness_key] > gcut)
        else:
            mask = (events['hillas_intensity'] > I_min) & (events['hillas_intensity'] < I_max) & (events[self.gammaness_key] > gcut)


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

    def plot_skymap(self):

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        if len(self.DL2_files) == 1:
            plt.title(f'Sky Map for {self.DL2_files[0].split("/")[-1]}')
        else:
            plt.title('Sky Map')
        
        ra_values = []
        dec_values = []
        pointings_ra = []
        pointings_dec = []
        rotation_angles = []
        cartesian_pointing_x = []
        cartesian_pointing_y = []
        cartesian_pointing_z = []

        cartesian_reco_x = []
        cartesian_reco_y = []
        cartesian_reco_z = []
        sky_offsets = []

        LST_EPOCH_pointings = []
        new_recos = []



        LST_EPOCH = Time("2018-10-01T00:00:00", scale="utc")

        for reco, cuts_mask, dl2, pointing in zip(self.reco_directions, self.cuts_masks, self.dl2s, self.pointings):
            # offsets = reco.spherical_offset_to(pointing)[cuts_mask]
            dl2 = dl2[cuts_mask]

            ra_values.extend(reco[cuts_mask].ra.deg)
            dec_values.extend(reco[cuts_mask].dec.deg)
            pointings_ra.extend(pointing[cuts_mask].ra.deg)
            pointings_dec.extend(pointing[cuts_mask].dec.deg)

    

            ################################
            # frame = AltAz(obstime=LST_EPOCH, location=self.telescope_location, pressure=100*u.hPa, temperature=20*u.deg_C, relative_humidity=0.1)
            # reco_temp = SkyCoord(alt=dl2[self.reco_alt_key], az=dl2[self.reco_az_key], frame=frame)#, obstime=dl2["time"])
            # pointing_temp = SkyCoord(alt=dl2[self.pointing_alt_key], az=dl2[self.pointing_az_key], frame=frame)#, obstime=dl2["time"])
            # transformed_reco = reco_temp.transform_to(self.source_position)
            # transformed_pointing = pointing_temp.transform_to(self.source_position)

            # offsets = transformed_reco.spherical_offsets_to(transformed_pointing)
            # new_recos.extend(pointing[cuts_mask].spherical_offsets_by(offsets[0], offsets[1]).transform_to(self.source_position))
            ################################
            # offsets = reco[cuts_mask].spherical_offsets_to(pointing[cuts_mask])
            # angles = (dl2[self.reco_az_key] - 100.893 * u.deg).to(u.rad)
            # print(angles)

            # offsets_x = np.cos(angles) * offsets[0] - np.sin(angles) * offsets[1]
            # offsets_y = np.sin(angles) * offsets[0] + np.cos(angles) * offsets[1]

            # new_recos.extend(pointing[cuts_mask].spherical_offsets_by(offsets_x, offsets_y).transform_to(self.source_position))
            ################################
            # old_pointing = SkyCoord(
            #     u.Quantity(dl2[self.pointing_az_key], unit=u.deg),
            #     u.Quantity(dl2[self.pointing_alt_key], unit=u.deg),
            #     frame="altaz",
            #     location=self.telescope_location,
            #     obstime=LST_EPOCH,
            # )

            # reco_direction = SkyCoord(
            #     u.Quantity(dl2[self.reco_az_key], unit=u.deg),
            #     u.Quantity(dl2[self.reco_alt_key], unit=u.deg),
            #     frame="altaz",
            #     location=self.telescope_location,
            #     obstime=LST_EPOCH,
            # )

            # new_pointing = SkyCoord(
            #     u.Quantity(dl2[self.pointing_az_key], unit=u.deg),
            #     u.Quantity(dl2[self.pointing_alt_key], unit=u.deg),
            #     frame="altaz",
            #     location=self.telescope_location,
            #     obstime=dl2[self.time_key],
            # )
            # sky_offset = old_pointing.spherical_offsets_to(reco_direction)
            # # better to make new object
            # reco_spherical_offset_az = u.Quantity(sky_offset[0], unit=u.deg)
            # reco_spherical_offset_alt = u.Quantity(sky_offset[1], unit=u.deg)

            # # angles = (dl2[self.reco_az_key] - 100.893 * u.deg).to(u.rad)

            # # rotated_reco_spherical_offset_az = np.cos(angles) * reco_spherical_offset_az - np.sin(angles) * reco_spherical_offset_alt
            # # rotated_reco_spherical_offset_alt = np.sin(angles) * reco_spherical_offset_az + np.cos(angles) * reco_spherical_offset_alt
            # rotated_reco_spherical_offset_az = reco_spherical_offset_az
            # rotated_reco_spherical_offset_alt = reco_spherical_offset_alt

            # # new_pointing = SkyCoord(
            # #     u.Quantity(dl2[self.pointing_az_key], unit=u.deg),
            # #     u.Quantity(dl2[self.pointing_alt_key], unit=u.deg),
            # #     frame="altaz",
            # #     location=self.telescope_location,
            # #     obstime=dl2[self.time_key],
            # # )
            # new_reco_direction = new_pointing.spherical_offsets_by(
            #     rotated_reco_spherical_offset_az, rotated_reco_spherical_offset_alt
            # ).transform_to(self.source_position)
            # new_recos.extend(new_reco_direction)
            ################################






            # az_values.extend(dl2[self.reco_az_key])  # Assuming 'az' is the azimuth key in dl2
            

            
            
            # sky_offset = pointing[cuts_mask].spherical_offsets_to(reco[cuts_mask])
            # print(sky_offset)
            # sky_offsets.extend([sky_offset[0].degree, sky_offset[1].degree])
            # # angular_separation = pointing.separation(reco)
            # # table.add_column(sky_offset[0], name="spherical_offset_az")
            # # table.add_column(sky_offset[1], name="spherical_offset_alt")
            # # table.add_column(angular_separation, name="angular_separation")
            # rotation_angles.extend((sky_offset[0].radian - (100.893 * u.deg).to(u.rad).value))
        # print(rotation_angles.shape)
        # print(sky_offsets.shape)
        # sky_offsets = np.array(sky_offsets)
        # new_sky_offset = [[np.cos(rotation_angles), -np.sin(rotation_angles)], [np.sin(rotation_angles), np.cos(rotation_angles)]] @ sky_offsets


        # new_sky_offset_x = []
        # new_sky_offset_y = []
        # for rot_angle, sky_offset in zip(rotation_angles, sky_offsets):
        #     new_sky_offset_x.append(np.cos(rot_angle) * sky_offset[0] - np.sin(rot_angle) * sky_offset[1])
        #     new_sky_offset_y.append(np.sin(rot_angle) * sky_offset[0] + np.cos(rot_angle) * sky_offset[1])

        #     # table.remove_columns(
        #     #     [
        #     #         "telescope_pointing_azimuth",
        #     #         "telescope_pointing_altitude",
        #     #     ]
        #     # )

        # new_pos = SkyCoord(ra=pointings_ra * u.deg, dec=pointings_dec * u.deg, frame='icrs').spherical_offsets_by(new_sky_offset_x * u.deg, new_sky_offset_y * u.deg)
        # new_recos_ra = [coord.ra.deg for coord in new_recos]
        # new_recos_dec = [coord.dec.deg for coord in new_recos]

        # plt.scatter(ra_values, new_recos_ra, s=1, label='Reconstructed', color='b')
        # plt.show()

        # plt.hist2d(new_recos_ra, new_recos_dec, bins=100, cmap='viridis', zorder=0)
        plt.hist2d(ra_values, dec_values, bins=100, cmap='viridis', zorder=0)
        # ax = plt.gca()
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(label='Counts')

        # plt.scatter(self.source_position.ra.deg, self.source_position.dec.deg, s=50, label='Source', marker='x', color='w', linewidths=2)
        plt.scatter(pointings_ra, pointings_dec, s=10, label='pointing', color='r')


        for pointing, cuts_mask in zip(self.pointings, self.cuts_masks):
            pointing = pointing[cuts_mask]
            off_regions = self.compute_off_regions(pointing[0], n_off=3)
            for off_region in off_regions:
                # print(off_region)
                off_circle = plt.Circle((off_region.ra.deg, off_region.dec.deg), radius=0.2, color='w', fill=False, lw=1, ls='--')
                plt.scatter(off_region.ra.deg, off_region.dec.deg, s=50, label='Off', marker='x', color='r', linewidths=2)
                plt.gca().add_artist(off_circle)

        on_circle = plt.Circle((self.source_position.ra.deg, self.source_position.dec.deg), radius=0.2, color='w', fill=False, lw=1)
        plt.scatter(self.source_position.ra.deg, self.source_position.dec.deg, s=50, label='Source', marker='o', color='g', linewidths=2)
        plt.gca().add_artist(on_circle)

        plt.gca().set_aspect('equal', adjustable='box')

        plt.legend()
        plt.show()

    def plot_sensitivity(self, n_off=3, ax=None, label="CTLearn"):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        E_bins = np.logspace(np.log10(0.03), np.log10(2), 10) * u.TeV
        on_count = np.zeros(len(E_bins) - 1)
        off_count = np.zeros(len(E_bins) - 1)
        t_eff = 0 * u.h
        t_elapsed = 0 * u.h
        # on_count_RF = np.zeros(len(gammaness_cuts_RF))
        # off_count_RF = np.zeros(len(gammaness_cuts_RF))
        for reco_direction, pointing_direction, dl2, cuts_mask in zip(self.reco_directions, self.pointings, self.dl2s, self.cuts_masks):
            reco_direction = reco_direction[cuts_mask]
            pointing_direction = pointing_direction[cuts_mask]
            dl2 = dl2[cuts_mask]

            for i, E_min, E_max in zip(range(len(E_bins) - 1), E_bins[:-1], E_bins[1:]):
                (
                    on_count_temp,
                    off_count_temp, 
                    _, _, _
                    ) = self.compute_on_off_counts( 
                    dl2, 
                    reco_direction, 
                    pointing_direction, 
                    n_off=n_off, 
                    theta2_cut=0.04*u.deg**2, 
                    gcut=self.gammaness_cut, 
                    E_min=E_min, 
                    E_max=E_max, 
                    I_min=None, 
                    I_max=None
                )
                on_count[i] += on_count_temp
                off_count[i] += off_count_temp / n_off
            t_eff_temp, t_elapsed_temp = self.compute_eff_time(dl2)
            t_eff += t_eff_temp
            t_elapsed += t_elapsed_temp
            # on_count_RF += df['on_count_RF'].to_numpy()
            # off_count_RF += df['off_count_RF'].to_numpy()
        
        nexcess = on_count-off_count

        min_signi = 3   # below this value (significance of the test source, Crab, for the *actual* observation 
                # time of the sample and obtained with 1 off region) we ignore the corresponding cut 
                # combination

        min_exc = 0.002 # in fraction of off. Below this we ignore the corresponding cut combination. 
        min_off_events = 10 # minimum number of off events in the actual observation used. Below this we 
                            # ignore the corresponding cut combination.

        backg_syst = 0.01
        t_eff = 0.33 * u.h
        obs_time = 50. * u.h 

        flux_factor, lima_signi = calc_flux_for_N_sigma(5, nexcess, off_count, min_signi, min_exc, min_off_events, 1, obs_time, t_eff, cond=False)  
        flux_minus, lima_signi_minus = calc_flux_for_N_sigma(5, nexcess + backg_syst * off_count + (nexcess + 2*off_count)**0.5, off_count, min_signi, min_exc, min_off_events, 1, obs_time, t_eff, cond=False)  
        flux_plus, lima_signi_plus = calc_flux_for_N_sigma(5, nexcess - backg_syst * off_count - (nexcess + 2*off_count)**0.5, off_count, min_signi, min_exc, min_off_events, 1, obs_time, t_eff, cond=False)


        # Create a figure with subplots
        # fig = plt.figure(figsize=(10, 8))
        # gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        mask = np.where(flux_factor >=0)


        E = (E_bins[:-1] + E_bins[1:])/2

        # Create figure and GridSpec layout
        # fig = plt.figure(figsize=(10, 8))
        # gs = GridSpec(1, 1, height_ratios=[1], hspace=0)  # Adjust hspace to remove space between plots

        # Top subplot
        # ax1 = fig.add_subplot(gs[0])
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(E[mask], flux_factor[mask] * 100, marker='o', label=label, zorder=10, ls='--')
        ax.fill_between(E[mask].value, flux_minus[mask]*100, flux_plus[mask]*100, alpha=0.2, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Reco Energy [TeV]")
        ax.set_ylabel("Differential sensitivity [% Obs. Flux.]")
        ax.set_xlim(0.03, 2)
        # ax.set_ylim(2, 60)
        ax.set_yticks([2, 5, 10, 20, 50])
        ax.set_yticklabels(['2', '5', '10', '20', '50'])
        ax.set_title('Differential sensitivity')
        ax.legend()

        plt.tight_layout()
        if ax is None:
            plt.show()

    def plot_PSF(self, n_off=3, ax=None, label="CTLearn"):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec



        E_bins = np.logspace(np.log10(0.03), np.log10(2), 10) * u.TeV
        on_count = np.zeros(len(E_bins) - 1)
        off_count = np.zeros(len(E_bins) - 1)
        t_eff = 0 * u.h
        t_elapsed = 0 * u.h
        angle_bins = np.linspace(0, 0.4, 25)
        h_on = np.zeros((len(E_bins) - 1, len(angle_bins) - 1))
        h_off = np.zeros((len(E_bins) - 1, len(angle_bins) - 1))
        # on_count_RF = np.zeros(len(gammaness_cuts_RF))
        # off_count_RF = np.zeros(len(gammaness_cuts_RF))
        for reco_direction, pointing_direction, dl2, cuts_mask in zip(self.reco_directions, self.pointings, self.dl2s, self.cuts_masks):
            reco_direction = reco_direction[cuts_mask]
            pointing_direction = pointing_direction[cuts_mask]
            dl2 = dl2[cuts_mask]

            for i, E_min, E_max in zip(range(len(E_bins) - 1), E_bins[:-1], E_bins[1:]):
                (
                    on_count_temp,
                    off_count_temp, 
                    on_separation_temp, 
                    all_off_separation_temp, 
                    _
                    ) = self.compute_on_off_counts( 
                    dl2, 
                    reco_direction, 
                    pointing_direction, 
                    n_off=n_off, 
                    theta2_cut=0.04*u.deg**2, 
                    gcut=self.gammaness_cut, 
                    E_min=E_min, 
                    E_max=E_max, 
                    I_min=None, 
                    I_max=None
                )
                on_count[i] += on_count_temp
                off_count[i] += off_count_temp
                h_on_temp, _ = np.histogram(on_separation_temp.to(u.deg).value**2, bins=angle_bins)
                h_off_temp, _ = np.histogram(all_off_separation_temp.to(u.deg).value**2, bins=angle_bins)
                h_on[i] += h_on_temp
                h_off[i] += h_off_temp / n_off # To plot the average off source counts
            t_eff_temp, t_elapsed_temp = self.compute_eff_time(dl2)
            t_eff += t_eff_temp
            t_elapsed += t_elapsed_temp

        nexcess = h_on-h_off

        psf = np.zeros(len(E_bins)-1)
        psf_min = np.zeros(len(E_bins)-1)
        psf_max = np.zeros(len(E_bins)-1)

        # bkg_condition = [False,  False,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False]
        # bkg_condition_RF = [False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True , True, False, False, False, False, False]
        for i, E_min, E_max in zip(range(len(E_bins)-1), E_bins[:-1], E_bins[1:]):
            # print(nexcess[i])
            # print(angle_bins)
            # print( find_68_percent_range(nexcess[i], angle_bins))
            psf[i] = find_68_percent_range(nexcess[i], angle_bins)**0.5
            psf_max[i] = find_68_percent_range(nexcess[i] + 0.01*h_off[i] + np.sqrt(nexcess[i] + 2*h_off[i]), angle_bins)**0.5
            psf_min[i] = find_68_percent_range(nexcess[i] - 0.01*h_off[i] - np.sqrt(nexcess[i] + 2*h_off[i]), angle_bins)**0.5

        E = (E_bins[:-1] + E_bins[1:])/2

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(E.value, psf, marker='o', label=label, zorder=10, ls='--')
        ax.fill_between(E.value, 
                        psf - 1/np.sqrt(np.sum(h_on, axis=1)), 
                        psf + 1/np.sqrt(np.sum(h_on, axis=1)), 
                        alpha=0.3, zorder=0,)
        ax.legend()
        ax.set_ylabel('68% cont. [deg]')
        ax.set_xlabel('Reco Energy [TeV]')
        ax.set_xscale('log')
        # ax.yscale('log')
        # ax.legend()
        ax.set_xlim(0.03, 2)
        # ax.ylim(bottom=0.1, top=0.5)
        ax.set_title('Point Spread Function')

        if ax is None:
            plt.show()
        # plt.show()

    def get_gammaness_cuts_for_efficiencies(self, MC_dl2, efficiencies, E_min=None, E_max=None, I_min=None, I_max=None):
        gammaness_cuts = []
        for efficiency in efficiencies:
            if E_min is not None and E_max is not None:
                mask = (MC_dl2[self.energy_key] > E_min) & (MC_dl2[self.energy_key] < E_max)
            elif I_min is not None and I_max is not None:
                mask = (MC_dl2['hillas_intensity'] > I_min) & (MC_dl2['hillas_intensity'] < I_max)
            else:
                mask = np.ones(len(MC_dl2), dtype=bool)
            
            sorted_gammaness = np.sort(MC_dl2[self.gammaness_key][mask])
            cut_index = int((1 - efficiency) * len(sorted_gammaness))
            gammaness_cut = sorted_gammaness[cut_index]
            gammaness_cuts.append(gammaness_cut)
        return gammaness_cuts

    def get_efficiency_for_gamaness_cuts(self, MC_dl2, gammaness_cuts, E_min=None, E_max=None, I_min=None, I_max=None):
        efficiencies = []
        for gammaness_cut in gammaness_cuts:
            if E_min is not None and E_max is not None:
                mask = (MC_dl2[self.energy_key] > E_min) & (MC_dl2[self.energy_key] < E_max)
            elif I_min is not None and I_max is not None:
                mask = (MC_dl2['hillas_intensity'] > I_min) & (MC_dl2['hillas_intensity'] < I_max)
            else:
                mask = np.ones(len(MC_dl2), dtype=bool)
            
            mask &= MC_dl2[self.gammaness_key] > gammaness_cut
            efficiency = len(MC_dl2[mask]) / len(MC_dl2)
            efficiencies.append(efficiency)
        return efficiencies

    def plot_bkg_discrimination_capability(self, n_off=3, axs=None, label="CTLearn"):
        gammaness_cuts = np.arange(0, 1.05, 0.05)
        import matplotlib.pyplot as plt

        if axs is None:
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))#, sharey=True)
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

        for i, ax, (I_min, I_max) in zip(range(len(intensity_ranges)), axs, intensity_ranges):
            ax.plot(I_g_off_counts_tot[i], I_g_on_counts_tot[i], marker='o', linestyle='-', label=label)
            ax.set_xlabel('Background Counts')
            ax.set_title(f'[{I_min} - {I_max}] p.e.')
            # ax.set_xscale('log')
            # ax.set_xlim(left=0.1)
           
        
        axs[0].set_ylabel('Excess Counts')
        axs[0].legend()
        plt.suptitle('Excess Counts vs Background Counts for Different Intensity Ranges')
        if axs is None:
            plt.show()

    def plot_excess_vs_background_rates(self, n_off=3):
        gammaness_cuts = np.arange(0, 1.05, 0.05)
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))#, sharey=True)
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

        for i, ax, (I_min, I_max) in zip(range(len(intensity_ranges)), axs, intensity_ranges):
            ax.plot(I_g_off_counts_tot[i] / total_t_eff, I_g_on_counts_tot[i] / total_t_eff, marker='o', linestyle='-',)
            ax.set_xlabel('Background Counts')
            ax.set_title(f'[{I_min} - {I_max}] p.e.')
            ax.set_xscale('log')
        
        axs[0].set_ylabel('Excess Rate [Hz]')
        plt.suptitle('Excess Rate vs Background Rate for Different Intensity Ranges')
        plt.show()

    def plot_excess_and_background_rates_vs_energy(self, n_off=3):
        import matplotlib.pyplot as plt

        E_bins = np.logspace(np.log10(0.03), np.log10(2), 10) * u.TeV
        excess_rates = np.zeros(len(E_bins) - 1)
        background_rates = np.zeros(len(E_bins) - 1)
        t_eff = 0 * u.h

        for reco_direction, pointing_direction, dl2 in zip(self.reco_directions, self.pointings, self.dl2s):
            for i, E_min, E_max in zip(range(len(E_bins) - 1), E_bins[:-1], E_bins[1:]):
                on_count, off_count, _, _, _ = self.compute_on_off_counts(
                    dl2, 
                    reco_direction, 
                    pointing_direction, 
                    n_off=n_off, 
                    theta2_cut=0.04 * u.deg ** 2, 
                    gcut=self.gammaness_cut, 
                    E_min=E_min, 
                    E_max=E_max
                )
                t_eff_temp, _ = self.compute_eff_time(dl2)
                excess_rates[i] += ((on_count - off_count / n_off) / t_eff_temp.to(u.s)).value
                background_rates[i] += (off_count / n_off / t_eff_temp.to(u.s)).value
                t_eff += t_eff_temp

        E = (E_bins[:-1] + E_bins[1:]) / 2

        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        axs[0].plot(E.value, excess_rates, marker='o', linestyle='-')
        axs[0].set_ylabel('Excess Rate [Hz]')
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        # axs[0].set_title('Excess Rate vs Energy')

        axs[1].plot(E.value, background_rates, marker='o', linestyle='-')
        axs[1].set_xlabel('Reco Energy [TeV]')
        axs[1].set_ylabel('Background Rate [Hz]')
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        # axs[1].set_title('Background Rate vs Energy')

        plt.setp(axs[0].get_xticklabels(), visible=False)

        plt.tight_layout()
        plt.show()





        
