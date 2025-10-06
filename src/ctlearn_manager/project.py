import os
from pathlib import Path

from . import CTLearnModelManager, CTLearnTriModelManager, DataSample
from .utils import CTLMDirectories, get_user_confirmation, ClusterConfiguration, set_mpl_style
from ctlearn_manager.utils.utils import set_global_theme, ColorTheme
# from.io import load_model_from_index

__all__ = [
    'CTLearnManagerProject',
]


class CTLearnManagerProject:


    def __init__(self, project_directory: str, theme:ColorTheme = ColorTheme.light_theme):

        if not Path(project_directory).resolve().is_absolute():
            raise ValueError("The project directory must be an absolute path.")

        if not Path(project_directory).exists():
            os.makedirs(project_directory, exist_ok=True)

        self.project_directory = project_directory
        set_global_theme(theme)

    def create_tri_model(
        self,
        tri_model_parameters: dict,
        overwrite: bool = False,
    ):
        """
        Create a CTLearn model manager instance with the given parameters.
        
        Parameters:
            model_nickname (str): Nickname for the model.
            model_parameters (dict): Parameters for the model.
            cluster_configuration (dict, optional): Configuration for the cluster.
        
        Returns:
            CTLearnModelManager: Instance of the CTLearnModelManager.
        """
        tri_model_nickname = tri_model_parameters.get("tri_model_nickname")
        tri_models_directory = f"{self.project_directory}/models/{tri_model_nickname}"
        if Path(tri_models_directory).exists():
            if not overwrite:
                raise FileExistsError(
                    f"TriModel {tri_model_nickname} already exists in {self.project_directory}. Use 'overwrite=True' to overwrite."
                )
            else:
                get_user_confirmation(prompt=f"TriModel {tri_model_nickname} already exists. Do you want to overwrite it?\n This will delete the existing model and all its data.")
                os.system(f"rm -rf {tri_models_directory}")
        project_directories = CTLMDirectories(self.project_directory, tri_model_nickname)
        

        direction_reco = tri_model_parameters.get("direction_reco", "cameradirection")
        assert direction_reco in ["cameradirection", "skydirection"], (
            f"direction_reco must be one of ['cameradirection', 'skydirection']: {direction_reco}"
        )
        recos = ['type', 'energy', direction_reco]
        if isinstance(tri_model_parameters.get("training_samples"), dict):
            training_samples = [
                tri_model_parameters.get("training_samples")['type'],
                tri_model_parameters.get("training_samples")['energy'],
                tri_model_parameters.get("training_samples")[direction_reco],
            ]
        else: # isinstance(tri_model_parameters.get("training_samples"), list[DataSample]):
            training_samples = [tri_model_parameters.get("training_samples")]*3
        # else:
        #     raise ValueError("training_samples must be a dict or a list of DataSample instances.")

        for reco, training_sample in zip(recos, training_samples):
            match reco:
                case "type":
                    os.makedirs(project_directories.type_model_directory, exist_ok=True)
                    model_dir = project_directories.type_model_directory
                    reco_siffix = 'type'
                case "energy":
                    os.makedirs(project_directories.energy_model_directory, exist_ok=True)
                    model_dir = project_directories.energy_model_directory
                    reco_siffix = 'energy'
                case "cameradirection" | "skydirection":
                    os.makedirs(project_directories.direction_model_directory, exist_ok=True)
                    model_dir = project_directories.direction_model_directory
                    reco_siffix = 'direction'
                case _:
                    raise ValueError(f"Unknown reco type: {reco}")

            model_parameters = {
                "model_nickname": f"{tri_model_nickname}_{reco_siffix}",
                "model_dir": model_dir,  # Main directory, will contain a nw directory with you model, named after the model_nickname, will be created for you.

                "reco": reco,  # ['energy', 'type', 'cameradirection', 'skydirection']
                "telescope_names": tri_model_parameters.get("telescope_names"),  # List of telescope names
                "telescope_ids": tri_model_parameters.get("telescope_ids"),  # List of telescope ids
                "max_training_epochs": tri_model_parameters.get("max_training_epochs"),  # Maximum number of training epochs
                "training_samples": training_sample, #tri_model_parameters.get("training_samples"),  # Training data
                "stereo": tri_model_parameters.get("stereo"),  # If True, model will be trained on stereo events
                #### OPTIONAL PARAMETERS
                'channels' : tri_model_parameters.get('channels', ['cleaned_image', 'cleaned_relative_peak_time']), # Order matters. # Default is ['cleaned_image', 'cleaned_relative_peak_time']
                'min_telescopes' : tri_model_parameters.get('min_telescopes', 1), # Minimum number of triggered telescopes for each events to be used in the model, if >=2, model will be stereo.
                'notes' : tri_model_parameters.get('notes', ''),  # Notes about the model
            }

            model = CTLearnModelManager(
                model_parameters,
                project_directories
            )

    def open_tri_model(
        self,
        tri_model_nickname: str,
        cluster_configuration:ClusterConfiguration=ClusterConfiguration(),
    ):
        """
        Open an existing CTLearn model manager instance with the given nickname.
        
        Parameters:
            tri_model_nickname (str): Nickname for the model.
            cluster_configuration (dict, optional): Configuration for the cluster.
        
        Returns:
            CTLearnModelManager: Instance of the CTLearnModelManager.
        """
        project_directories = CTLMDirectories(self.project_directory, tri_model_nickname)
        if not Path(project_directories.tri_models_directory).exists():
            raise FileNotFoundError(
                f"TriModel {tri_model_nickname} does not exist in {self.project_directory}."
            )

        type_model = project_directories.load_model_from_index(
            f"{tri_model_nickname}_type",
            project_directories.model_index_file,
        )

        energy_model = project_directories.load_model_from_index(
            f"{tri_model_nickname}_energy",
            project_directories.model_index_file,
        )

        direction_model = project_directories.load_model_from_index(
            f"{tri_model_nickname}_direction",
            project_directories.model_index_file,
        )

        tri_model = CTLearnTriModelManager(
            direction_model=direction_model,
            energy_model=energy_model,
            type_model=type_model,
            cluster_configuration=cluster_configuration,
            project_directories=project_directories,
        )
        return tri_model


