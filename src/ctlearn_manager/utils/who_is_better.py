import matplotlib.pyplot as plt

class WhoIsBetter():
    def __init__(self, dl2_processors, labels):
        self.dl2_processors = dl2_processors
        self.labels = labels


    def plot_theta2_distribution(self):
        for dl2_processor in self.dl2_processors:
            dl2_processor.plot_theta2_distribution()
    
    def plot_skymap(self):
        for dl2_processor in self.dl2_processors:
            dl2_processor.plot_skymap()

    def plot_sensitivity(self):
        
        fig, ax = plt.subplots()
        for dl2_processor, label in zip(self.dl2_processors, self.labels):
            dl2_processor.plot_sensitivity(ax=ax, label=label)
        plt.show()

    def plot_PSF(self):
        fig, ax = plt.subplots()
        for dl2_processor, label in zip(self.dl2_processors, self.labels):
            dl2_processor.plot_PSF(ax=ax, label=label)

        plt.show()

    def plot_bkg_discrimination_capability(self):
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        for dl2_processor, label in zip(self.dl2_processors, self.labels):
            dl2_processor.plot_bkg_discrimination_capability(axs=axs, label=label)

        plt.show()

    
