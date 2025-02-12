import os
import pickle
from astropy.coordinates import SkyCoord, AltAz
from astropy import units as u
from ..utils.sky_coord_utils import DL2DataProcessor
from ..io.io import load_DL2_data, load_DL2_data_RF
from astropy.time import Time
import numpy as np
import tqdm

import argparse

parser = argparse.ArgumentParser(description="Process DL2 file with DL2DataProcessor")
parser.add_argument("dl2_file", type=str, help="Path to the DL2 file to process")
parser.add_argument("processor", type=str, help="Path to the DL2DataProcessor instance")




def process_dl2_file():

    args = parser.parse_args()

    # Assuming DL2DataProcessor instance is saved as a pickle file
    with open(args.processor, 'rb') as f:
        processor = pickle.load(f)

    DL2_file = args.dl2_file


    if processor.dl2_processed_dir is None:
        dl2_output_file = DL2_file.replace('.h5', '_dl2_processed.pkl')
        reco_output_file = DL2_file.replace('.h5', '_reco_directions.pkl')
        pointing_output_file = DL2_file.replace('.h5', '_pointings.pkl')
        I_g_on_counts_output_file = DL2_file.replace('.h5', '_I_g_on_counts.pkl')
        I_g_off_counts_output_file = DL2_file.replace('.h5', '_I_g_off_counts.pkl')
    else:
        dl2_output_file = os.path.join(processor.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_dl2_processed.pkl'))
        reco_output_file = os.path.join(processor.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_reco_directions.pkl'))
        pointing_output_file = os.path.join(processor.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_pointings.pkl'))
        I_g_on_counts_output_file = os.path.join(processor.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_I_g_on_counts.pkl'))
        I_g_off_counts_output_file = os.path.join(processor.dl2_processed_dir, os.path.basename(DL2_file).replace('.h5', '_I_g_off_counts.pkl'))

    if not os.path.exists(dl2_output_file):
        print(f"Loading {DL2_file}", flush=True)
        if processor.CTLearn:
            dl2 = load_DL2_data(DL2_file, processor)
        else:
            dl2 = load_DL2_data_RF(DL2_file, processor)
        dl2 = dl2[dl2[processor.gammaness_key] > 0] # Remove unpredicted events
        with open(dl2_output_file, 'wb') as f:
            pickle.dump(dl2, f)
        print(f"Saved processed DL2 data to {dl2_output_file}", flush=True)
    else:
        print(f"Loading processed DL2 data from {dl2_output_file}", flush=True)
        with open(dl2_output_file, 'rb') as f:
            dl2 = pickle.load(f)
    dl2 = dl2[dl2[processor.gammaness_key] > 0] # Remove unpredicted events
    print(f"Loaded {len(dl2)} events", flush=True)
    # cut_mask = dl2[processor.gammaness_key] > processor.gammaness_cut
    # dl2_cuts = dl2[cut_mask]
    # print(f"{len(dl2_cuts)} events after cuts", flush=True)

    if (not os.path.exists(reco_output_file)) or (not os.path.exists(pointing_output_file)): 
        print("Computing sky positions...", flush=True)
        if processor.CTLearn:
            times = dl2[processor.time_key]
        else:
            times = Time(np.array(dl2[processor.time_key]), format='unix', scale='tai')
        # times = Time(np.array(dl2["time"]), format='mjd', scale='tai')

        frame = AltAz(obstime=times, location=processor.telescope_location, pressure=100*u.hPa, temperature=20*u.deg_C, relative_humidity=0.1)
        reco_temp = SkyCoord(alt=dl2[processor.reco_alt_key], az=dl2[processor.reco_az_key], frame=frame)#, obstime=dl2["time"])
        pointing_temp = SkyCoord(alt=dl2[processor.pointing_alt_key], az=dl2[processor.pointing_az_key], frame=frame)#, obstime=dl2["time"])
        transformed_reco = reco_temp.transform_to(processor.source_position)
        transformed_pointing = pointing_temp.transform_to(processor.source_position)

        # Convert SkyCoord objects to dictionaries
        transformed_reco_dict = {'ra': transformed_reco.ra.deg, 'dec': transformed_reco.dec.deg}
        transformed_pointing_dict = {'ra': transformed_pointing.ra.deg, 'dec': transformed_pointing.dec.deg}

        with open(reco_output_file, 'wb') as f:
            pickle.dump(transformed_reco_dict, f)
        with open(pointing_output_file, 'wb') as f:
            pickle.dump(transformed_pointing_dict, f)

        print(f"Saved reco directions to {reco_output_file}", flush=True)
        print(f"Saved pointings to {pointing_output_file}", flush=True)
        
    else:
        print(f"Loading reco directions from {reco_output_file}", flush=True)
        with open(reco_output_file, 'rb') as f:
            transformed_reco_dict = pickle.load(f)
        print(f"Loading pointings from {pointing_output_file}", flush=True)
        with open(pointing_output_file, 'rb') as f:
            transformed_pointing_dict = pickle.load(f)
        transformed_reco = SkyCoord(ra=transformed_reco_dict['ra']*u.deg, dec=transformed_reco_dict['dec']*u.deg, frame=processor.source_position)
        transformed_pointing = SkyCoord(ra=transformed_pointing_dict['ra']*u.deg, dec=transformed_pointing_dict['dec']*u.deg, frame=processor.source_position)

    if (not os.path.exists(I_g_on_counts_output_file)) or (not os.path.exists(I_g_off_counts_output_file)):
        gammaness_cuts = np.arange(0, 1.05, 0.05)
        n_off=3

        intensity_ranges = [(50, 200), (200, 800), (800, 3200), (3200, np.inf)]
        I_g_on_counts = []
        I_g_off_counts = []
        for (I_min, I_max) in intensity_ranges:
            excess_counts = []
            off_counts = []
            print(f"Computing excesses for [{I_min} - {I_max}] p.e.", flush=True)
            for gcut in gammaness_cuts:
                print(f"Computing excesses for gammaness cut {gcut}", flush=True)
                total_excess = 0
                total_off = 0
                # for reco_direction, pointing_direction, dl2 in zip(processor.reco_directions, processor.pointings, processor.dl2s):
                on_count, off_count, _, _, _ = processor.compute_on_off_counts(
                    dl2, 
                    transformed_reco, 
                    transformed_pointing, 
                    n_off=n_off, 
                    theta2_cut=0.04 * u.deg ** 2, 
                    gcut=gcut, 
                    E_min=None, 
                    E_max=None, 
                    I_min=I_min, 
                    I_max=I_max
                )
                total_excess += on_count - off_count / n_off
                total_off += off_count / n_off

                excess_counts.append(total_excess)
                off_counts.append(total_off)

            I_g_on_counts.append(excess_counts)
            I_g_off_counts.append(off_counts)
        
        with open(I_g_on_counts_output_file, 'wb') as f:
            pickle.dump(I_g_on_counts, f)
        with open(I_g_off_counts_output_file, 'wb') as f:
            pickle.dump(I_g_off_counts, f)

    os.remove(args.processor)
    print(f"Removed processor pickle file {args.processor}", flush=True)


if __name__ == "__main__":
    process_dl2_file()