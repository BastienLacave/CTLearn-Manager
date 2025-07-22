import matplotlib.pyplot as plt
from ctlearn_manager.utils.DL2_processing import DL2DataProcessor

__all__ = ["WhoIsBetter"]


class WhoIsBetter:
    def __init__(self, dl2_processors: list[DL2DataProcessor], labels: list[str]):
        self.dl2_processors = dl2_processors
        cuts = [dl2_processor.cuts for dl2_processor in dl2_processors]
        if not all(cut == cuts[0] for cut in cuts):
            raise ValueError("Cuts from each dl2_processor are not identical.")
        self.cuts = cuts[0]
        self.labels = labels
        assert len(self.dl2_processors) == len(self.labels), "Number of dl2_processors and labels must match."

    def plot_theta2_distribution(self, output_file=None):
        for dl2_processor, label in zip(self.dl2_processors, self.labels):
            dl2_processor.plot_theta2_distribution(output_file=output_file.replace(".png", f"_{label}.png") if output_file else None)

    def plot_skymap(self, output_file=None):
        for dl2_processor, label in zip(self.dl2_processors, self.labels):
            dl2_processor.plot_skymap(output_file=output_file.replace(".png", f"_{label}.png") if output_file else None)

    def plot_sensitivity(self, output_file=None):
        fig, ax = plt.subplots()
        if len(self.cuts) == 1:
            self.cuts[0].plot_cuts_info_plt(ax)
        for dl2_processor, label in zip(self.dl2_processors, self.labels):
            dl2_processor.plot_sensitivity(ax=ax, label=label)
        if output_file:
            plt.savefig(output_file)
        plt.show()

    def plot_PSF(self, output_file=None):
        fig, ax = plt.subplots()
        if len(self.cuts) == 1:
            stored_efficiency_theta = self.cuts[0].efficiency_theta
            self.cuts[0].efficiency_theta = None
            self.cuts[0].plot_cuts_info_plt(ax)
            self.cuts[0].efficiency_theta = stored_efficiency_theta
        for dl2_processor, label in zip(self.dl2_processors, self.labels):
            dl2_processor.plot_PSF(ax=ax, label=label)
        if output_file:
            plt.savefig(output_file)
        plt.show()

    def plot_bkg_discrimination_capability(self, output_file=None):
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        for dl2_processor, label in zip(self.dl2_processors, self.labels):
            dl2_processor.plot_bkg_discrimination_capability(axs=axs, label=label)
        if output_file:
            plt.savefig(output_file)
        plt.show()
