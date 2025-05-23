from astropy.table import QTable
from .utils import ParticleType
from ..model_manager import CTLearnModelManager

__all__ = [
    "IndexTables",]

class IndexTables():

    def __call__(self, model_manager: CTLearnModelManager, particle_type: ParticleType=None):
        self.model_manager = model_manager
        self.particle_type = particle_type

        if self.particle_type is not None:
            self.DL2_MC = self.IndexTable(
                QTable(
                        names=[
                            f"testing_DL2_{self.particle_type.value}_files",
                            f"testing_DL2_{self.particle_type.value}_zenith_distances",
                            f"testing_DL2_{self.particle_type.value}_azimuths",
                        ],
                        dtype=["S256", float, float],
                        units=[None, "deg", "deg"],
                    ),
                f"{self.model_manager.model_nickname}/DL2/MC/{particle_type.value}"
                )


            self.TRAINING = self.IndexTable(
                QTable(
                        names=[
                            f"training_{particle_type.value}_dir",
                            f"training_{particle_type.value}_patterns",
                            f"training_{particle_type.value}_zenith_distances",
                            f"training_{particle_type.value}_azimuths",
                            f"training_{particle_type.value}_energy_min",
                            f"training_{particle_type.value}_energy_max",
                            f"training_{particle_type.value}_nsb_min",
                            f"training_{particle_type.value}_nsb_max",
                        ],
                        dtype=[
                            "S256",
                            "S256",
                            float,
                            float,
                            float,
                            float,
                            float,
                            float,
                        ],
                        units=[None, None, "deg", "deg", "TeV", "TeV", "Hz", "Hz"],
                    ),
                f"{self.model_manager.model_nickname}/training/{particle_type.value}")
            
            self.TESTING = self.IndexTable(
                QTable(
                    names=[
                        f"testing_{particle_type.value}_dirs",
                        f"testing_{particle_type.value}_zenith_distances",
                        f"testing_{particle_type.value}_azimuths",
                        f"testing_{particle_type.value}_patterns",
                    ],
                    dtype=["S256", float, float, "S256"],
                    units=[None, "deg", "deg", None],
            ),
                f"{self.model_manager.model_nickname}/testing/{particle_type.value}",
            )


        self.PARAMETERS = self.IndexTable(
            QTable(
                names=[
                    "model_nickname",
                    "model_dir",
                    "reco",
                    "channels",
                    "telescope_names",
                    "telescope_ids",
                    "notes",
                    "max_training_epochs",
                    "min_telescopes",
                    "stereo",
                ],
                dtype=[
                    "S256",
                    "S256",
                    "S256",
                    "S256",
                    "S256",
                    "S256",
                    "S256",
                    int,
                    int,
                    bool,
                ],
            ),
            f"{self.model_manager.model_nickname}/parameters"
        )


        self.IRF = self.IndexTable(
            QTable(
                names=[
                    "config",
                    "cuts_file",
                    "irf_file",
                    "benckmark_file",
                    "zenith",
                    "azimuth",
                ],
                dtype=["S256", "S256", "S256", "S256", float, float],
                units=[None, None, None, None, "deg", "deg"],
            ),
            f"{self.model_manager.model_nickname}/IRF"
        )

        
        self.DL2_DATA = self.IndexTable(
            QTable(
                names=["DL2_files", "DL2_zenith_distances", "DL2_azimuths"],
                dtype=["S256", float, float],
            ),
            f"{self.model_manager.model_nickname}/DL2/Data"
        )


    class IndexTable:
        def __init__(self, default_table: QTable, table_path: str):
            self.default_table = default_table
            self.table_path = table_path
