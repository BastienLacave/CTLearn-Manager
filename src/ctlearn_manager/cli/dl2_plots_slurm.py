import argparse
import os
import pickle

from ..utils.DL2_processing import DL2DataProcessor


def plot_dl2():
    """Main function to process DL2 files and plot the results."""
    parser = argparse.ArgumentParser(
        description="Process DL2 file with DL2DataProcessor"
    )
    parser.add_argument(
        "--stereo_tri_model", type=str, help="Path to the stereo tri model"
    )
    parser.add_argument(
        "--output_directory", type=str, help="Path to the output directory"
    )
    parser.add_argument(
        "--gammaness_cut",
        type=float,
        help="Gammaness cut for the data processing",
        default=0.9,
    )
    parser.add_argument("--edep_cuts", help="Apply energy cuts", default=False)
    args = parser.parse_args()

    with open(args.stereo_tri_model, "rb") as f:
        stereo_tri_model = pickle.load(f)
    print(args.edep_cuts)
    print(type(args.edep_cuts))
    print(args.gammaness_cut)

    dl2_processor = DL2DataProcessor(
        stereo_tri_model.dl2_data_files,
        stereo_tri_model,
        # gammaness_cut=args.gammaness_cut,
    )  # , edep_cuts=args.edep_cuts)
    dl2_processor.plot_everything(
        args.output_directory, suffix=stereo_tri_model.direction_model.model_nickname
    )

    os.remove(args.stereo_tri_model)


if __name__ == "__main__":
    plot_dl2()
