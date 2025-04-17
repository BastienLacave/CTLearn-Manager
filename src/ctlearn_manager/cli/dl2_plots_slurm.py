import argparse
import os
import pickle

from ..utils.DL2_processing import DL2DataProcessor

parser = argparse.ArgumentParser(description="Process DL2 file with DL2DataProcessor")
parser.add_argument("stereo_tri_model", type=str, help="Path to the stereo tri model")
parser.add_argument("output_directory", type=str, help="Path to the output directory")
parser.add_argument("dl2_processed_dir")
parser.add_argument("gammaness_cut", type=float, help="Gammaness cut for the data processing", default=0.9)



def plot_dl2():
    """Main function to process DL2 files and plot the results."""
    args = parser.parse_args()

    with open(args.stereo_tri_model, 'rb') as f:
        stereo_tri_model = pickle.load(f)

    dl2_processor = DL2DataProcessor(stereo_tri_model.dl2_data_files, stereo_tri_model, gammaness_cut=args.gammaness_cut, dl2_processed_dir="/capstor/scratch/cscs/blacave/data/CRAB_DL2_Multimodels/new/dl2_preprocessed/")
    dl2_processor.plot_everything(args.output_directory)

    os.remove(args.stereo_tri_model)
        


if __name__ == "__main__":
    plot_dl2()