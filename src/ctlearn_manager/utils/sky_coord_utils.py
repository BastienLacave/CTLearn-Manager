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
        self_telscope_names = CTLearnTriModelManager.telescope_names
        self.stereo = CTLearnTriModelManager.stereo
        self.gammaness_cut = gammaness_cut
        self.pointing_table = pointing_table
        self.reconstruction_method = "CTLearn"
        self.reco_field_suffix = self.reconstruction_method if self.stereo else f"{self.reconstruction_method}_tel"
        self.telescope_id = CTLearnTriModelManager.telescope_ids if self.stereo else CTLearnTriModelManager.telescope_ids[0]
        # self.irfs = CTLearnTriModelManager.irfs



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

        

    def process_DL2_data(self):
        
        # print(f"Preprocessing DL2 data...")
        

        for DL2_file in self.DL2_files:
            if self.dl2_processed_dir is None:
                dl2_output_file = DL2_file.replace('.h5', '_dl2_processed.pkl')
                reco_output_file = DL2_file.replace('.h5', '_reco_directions.pkl')
                pointing_output_file = DL2_file.replace('.h5', '_pointings.pkl')
            else:
                dl2_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_dl2_processed.pkl'))
                reco_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_reco_directions.pkl'))
                pointing_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_pointings.pkl'))


            if (not os.path.exists(reco_output_file)) or (not os.path.exists(pointing_output_file)) or (not os.path.exists(dl2_output_file)):
                if self.CTLearnTriModelManager.cluster_configuration.use_cluster:
                    processor_file = f"{self.dl2_processed_dir}/{DL2_file.split('/')[-1]}_processor.pkl"
                    with open(processor_file, 'wb') as f:
                        pickle.dump(self, f)
                    self.CTLearnTriModelManager.cluster_configuration.write_sbatch_script(f"process_dl2_{DL2_file.split('/')[-1]}", f"process_dl2_file {DL2_file} {processor_file}", self.dl2_processed_dir)
                    os.system(f"sbatch {self.dl2_processed_dir}/process_dl2_{DL2_file.split('/')[-1]}.sh")
                else:
                    print(f"[NOT USING SLURM] Processing {DL2_file}")

                    processor_file = f"{self.dl2_processed_dir}/{DL2_file.split('/')[-1]}_processor.pkl"
                    with open(processor_file, 'wb') as f:
                        pickle.dump(self, f)
                    # self.CTLearnTriModelManager.cluster_configuration.write_sbatch_script(f"process_dl2_{DL2_file.split('/')[-1]}", f"process_dl2_file {DL2_file} {processor_file}", self.dl2_processed_dir)
                    # os.system(f"sbatch {self.dl2_processed_dir}/process_dl2_{DL2_file.split('/')[-1]}.sh")
                    os.system(f"process_dl2_file {DL2_file} {processor_file}")



                    # dl2 = load_DL2_data(DL2_file, self)
                    # dl2 = dl2[dl2[f"{self.reco_field_suffix}_prediction"] > 0] # Remove unpredicted events
                    # with open(dl2_output_file, 'wb') as f:
                    #     pickle.dump(dl2, f)
                    # print(f"Saved processed DL2 data to {dl2_output_file}")

                    # dl2 = dl2[dl2[f"{self.reco_field_suffix}_prediction"] > 0] # Remove unpredicted events
                    # cut_mask = dl2[f"{self.reco_field_suffix}_prediction"] > self.gammaness_cut
                    # dl2_cuts = dl2[cut_mask]
                    # print(f"{len(dl2_cuts)} events after cuts")

                    # print("Computing sky positions...")
                    # times = dl2["time"]
                    # # times = Time(np.array(dl2["time"]), format='mjd', scale='tai')

                    # frame = AltAz(obstime=times, location=self.telescope_location, pressure=100*u.hPa, temperature=20*u.deg_C, relative_humidity=0.1)
                    # reco_temp = SkyCoord(alt=dl2[f"{self.reco_field_suffix}_alt"], az=dl2[f"{self.reco_field_suffix}_az"], frame=frame)#, obstime=dl2["time"])
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
        self.reco_directions = []
        self.pointings = []
        self.dl2s = []
        self.dl2s_cuts = []

        for DL2_file in self.DL2_files:
            if self.dl2_processed_dir is None:
                dl2_output_file = DL2_file.replace('.h5', '_dl2_processed.pkl')
                reco_output_file = DL2_file.replace('.h5', '_reco_directions.pkl')
                pointing_output_file = DL2_file.replace('.h5', '_pointings.pkl')
            else:
                dl2_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_dl2_processed.pkl'))
                reco_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_reco_directions.pkl'))
                pointing_output_file = os.path.join(self.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_pointings.pkl'))

            if (os.path.exists(reco_output_file)) and (os.path.exists(pointing_output_file)) and (os.path.exists(dl2_output_file)):

                with open(dl2_output_file, 'rb') as f:
                        dl2 = pickle.load(f)
                

                print(f"Loading reco directions from {reco_output_file}")
                with open(reco_output_file, 'rb') as f:
                    transformed_reco_dict = pickle.load(f)
                print(f"Loading pointings from {pointing_output_file}")
                with open(pointing_output_file, 'rb') as f:
                    transformed_pointing_dict = pickle.load(f)

                dl2 = dl2[dl2[f"{self.reco_field_suffix}_prediction"] > 0] # Remove unpredicted events
                # avg_data_ze, avg_data_az = get_avg_pointing(DL2_file, pointing_table=f"{self.pointing_table}/tel_{self.telescope_id:03d}")
                # cuts_file = self.CTLearnTriModelManager.direction_model.get_closest_IRF_data(avg_data_ze, avg_data_az)[1]
                # TODO use energy dependent cuts
                cut_mask = dl2[f"{self.reco_field_suffix}_prediction"] > self.gammaness_cut
                dl2_cuts = dl2[cut_mask]
                print(f"{len(dl2_cuts)} events after cuts")
                
                # Convert dictionaries back to SkyCoord objects
                transformed_reco = SkyCoord(ra=transformed_reco_dict['ra']*u.deg, dec=transformed_reco_dict['dec']*u.deg, frame=self.source_position)
                transformed_pointing = SkyCoord(ra=transformed_pointing_dict['ra']*u.deg, dec=transformed_pointing_dict['dec']*u.deg, frame=self.source_position)
        
                self.reco_directions.append(transformed_reco[cut_mask])
                self.pointings.append(transformed_pointing[cut_mask])
                self.dl2s.append(dl2)
                self.dl2s_cuts.append(dl2_cuts)

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
        timestamp = np.array(events["time"].to_value('unix'))

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

    def plot_skymap(self):

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        plt.title('Sky Map')
        
        ra_values = []
        dec_values = []
        pointings_ra = []
        pointings_dec = []

        for reco in self.reco_directions:
            ra_values.extend(reco.ra.deg)
            dec_values.extend(reco.dec.deg)
        
        for pointing in self.pointings:
            pointings_ra.extend(pointing.ra.deg)
            pointings_dec.extend(pointing.dec.deg)

        plt.hist2d(ra_values, dec_values, bins=100, cmap='viridis')
        # ax = plt.gca()
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(label='Counts')

        plt.scatter(self.source_position.ra.deg, self.source_position.dec.deg, s=50, label='Source', marker='x', color='w', linewidths=2)
        plt.scatter(pointings_ra, pointings_dec, s=10, label='pointing', color='r')


        for pointing in self.pointings:
            off_regions = self.compute_off_regions(pointing[0], n_off=3)
            for off_region in off_regions:
                # print(off_region)
                off_circle = plt.Circle((off_region.ra.deg, off_region.dec.deg), radius=0.2, color='w', fill=False, lw=1, ls='--')
                plt.gca().add_artist(off_circle)

        on_circle = plt.Circle((self.source_position.ra.deg, self.source_position.dec.deg), radius=0.2, color='w', fill=False, lw=1)
        plt.gca().add_artist(on_circle)

        plt.gca().set_aspect('equal', adjustable='box')

        plt.legend()
        plt.show()

    


    def plot_sensitivity(self, n_off=3):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        E_bins = np.logspace(np.log10(0.03), np.log10(2), 10) * u.TeV
        on_count = np.zeros(len(E_bins) - 1)
        off_count = np.zeros(len(E_bins) - 1)
        t_eff = 0 * u.h
        t_elapsed = 0 * u.h
        # on_count_RF = np.zeros(len(gammaness_cuts_RF))
        # off_count_RF = np.zeros(len(gammaness_cuts_RF))
        for reco_direction, pointing_direction, dl2 in zip(self.reco_directions, self.pointings, self.dl2s_cuts):

            for i, E_min, E_max in zip(range(len(E_bins) - 1), E_bins[:-1], E_bins[1:]):
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
                    E_min=E_min, 
                    E_max=E_max, 
                    I_min=None, 
                    I_max=None
                )
                on_count[i] += on_count_temp
                off_count[i] += off_count_temp
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
        plt.plot(E[mask], flux_factor[mask] * 100, marker='o', label=r"CTLearn", zorder=10, ls='--')
        plt.fill_between(E[mask].value, flux_minus[mask]*100, flux_plus[mask]*100, alpha=0.2, zorder=0)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Reco Energy [TeV]")
        plt.ylabel("Differential sensitivity [% Obs. Flux.]")
        plt.xlim(0.03, 2)
        plt.ylim(2, 60)
        plt.yticks([2, 5, 10, 20, 50])
        plt.gca().set_yticklabels(['2', '5', '10', '20', '50'])
        plt.title('Differential sensitivity')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_PSF(self, n_off=3):
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
        for reco_direction, pointing_direction, dl2 in zip(self.reco_directions, self.pointings, self.dl2s_cuts):

            for i, E_min, E_max in zip(range(len(E_bins) - 1), E_bins[:-1], E_bins[1:]):
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

        plt.plot(E.value, psf, marker='o', label=r"CTLearn", zorder=10, ls='--')
        plt.fill_between(E.value, 
                        psf - 1/np.sqrt(np.sum(h_on, axis=1)), 
                        psf + 1/np.sqrt(np.sum(h_on, axis=1)), 
                        alpha=0.3, zorder=0,)
        plt.legend()
        plt.ylabel('68% cont. [deg]')
        plt.xlabel('Reco Energy [TeV]')
        plt.xscale('log')
        # plt.yscale('log')
        plt.legend()
        plt.xlim(0.03, 2)
        # plt.ylim(bottom=0.1, top=0.5)
        plt.title('Point Spread Function')

        plt.show()




        
