"""
Create HDU index files for HDU tables and Obs tables,
from a given path of DL3 files and a glob pattern to select DL3 files
The index filenames are the standard as per
http://gamma-astro-data-formats.readthedocs.io/en/latest/

The Index files can be stored in a different path, but by default
they are stored at the same place as the DL3 files.
"""
from ctapipe.core import (
    Provenance,
    Tool,
    ToolConfigurationError,
    traits,
)
import os
from astropy.io import fits
from astropy.table import QTable, Table
from astropy.time import Time
import logging
from lstchain.__init__ import __version__
from lstchain.high_level import (
    # create_hdu_index_hdu,
    create_obs_index_hdu,
)


__all__ = ["FITSIndexWriter"]

log = logging.getLogger(__name__)

DEFAULT_HEADER = fits.Header()
DEFAULT_HEADER["CREATOR"] = f"lstchain v{__version__}"
DEFAULT_HEADER["HDUDOC"] = (
    "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
)
DEFAULT_HEADER["HDUVERS"] = "0.3"
DEFAULT_HEADER["HDUCLASS"] = "GADF"
DEFAULT_HEADER["ORIGIN"] = "CTA"
DEFAULT_HEADER["TELESCOP"] = "CTA-N"

def create_hdu_index_hdu(file_list, hdu_index_file, overwrite=False):
    """
    Create the hdu index table and write it to the given file.

    Parameters
    ----------
    file_list : list
        list of the fits files
    hdu_index_file : Path
        Path for HDU index file
    overwrite : Bool
        Boolean to overwrite existing file
    """

    hdu_index_tables = []

    base_dir = os.path.commonpath(
        [hdu_index_file.parent.resolve(), file_list[0].resolve()]
    )
    # loop through the files
    for file in file_list:
        if file.is_file():
            try:
                hdu_list = fits.open(file)
                # check that the HDUs are present
                evt_hdr = hdu_list["EVENTS"].header
                hdu_list["GTI"].header
                hdu_list["POINTING"].header
            except Exception:
                log.error(f"fits corrupted for file {file}")
                continue
        else:
            log.error(f"fits {file} doesn't exist")
            continue

        # Event list
        t_events = {
            "OBS_ID": evt_hdr["OBS_ID"],
            "HDU_TYPE": "events",
            "HDU_CLASS": "events",
            "FILE_DIR": os.path.relpath(file.parent, hdu_index_file.parent),
            "FILE_NAME": file.name,
            "HDU_NAME": "EVENTS",
            "SIZE": file.stat().st_size,
        }
        hdu_index_tables.append(t_events)

        # GTI
        t_gti = t_events.copy()

        t_gti["HDU_TYPE"] = "gti"
        t_gti["HDU_CLASS"] = "gti"
        t_gti["HDU_NAME"] = "GTI"

        hdu_index_tables.append(t_gti)

        # POINTING
        t_pnt = t_events.copy()

        t_pnt["HDU_TYPE"] = "pointing"
        t_pnt["HDU_CLASS"] = "pointing"
        t_pnt["HDU_NAME"] = "POINTING"

        hdu_index_tables.append(t_pnt)

        # 0:PRIMARY, 1:EVENTS, 2:GTI, 3:POINTING, 4-:IRF 
        for hdu in hdu_list[4:]:
            # print(hdu.name)
            # GH_CUTS and AL_CUTS don't have HDUCLAS4 header
            # if hdu.header["EXTNAME"] in ['GH_CUTS', 'AL_CUTS', 'QUALITY_CUTS_EXPR']:
            #     continue

            if hdu.header["EXTNAME"] in ["EFFECTIVE AREA", "ENERGY DISPERSION", "PSF"]:

                irf_hdu = hdu.header["HDUCLAS4"]
                
                t_irf = t_events.copy()
                t_irf["HDU_CLASS"] = irf_hdu.lower()
                t_irf["HDU_TYPE"] = irf_hdu.lower().strip(
                    "_" + irf_hdu.lower().split("_")[-1]
                )
                t_irf["HDU_NAME"] = hdu.name
                hdu_index_tables.append(t_irf)
            else:
                continue
            
    hdu_index_table = Table(hdu_index_tables)

    hdu_index_header = DEFAULT_HEADER.copy()
    hdu_index_header["CREATED"] = Time.now().utc.iso
    hdu_index_header["HDUCLAS1"] = "INDEX"
    hdu_index_header["HDUCLAS2"] = "HDU"
    hdu_index_header["INSTRUME"] = evt_hdr["INSTRUME"]
    hdu_index_header["BASE_DIR"] = base_dir

    hdu_index = fits.BinTableHDU(
        hdu_index_table, header=hdu_index_header, name="HDU INDEX"
    )
    hdu_index_list = fits.HDUList([fits.PrimaryHDU(), hdu_index])
    hdu_index_list.writeto(hdu_index_file, overwrite=overwrite)

class FITSIndexWriter(Tool):
    name = "FITSIndexWriter"
    description = __doc__
    example = """
    To create DL3 index files with default values:
    > lstchain_create_dl3_index_files
        -d /path/to/DL3/files/

    Or specify some more configurations:
    > lstchain_create_dl3_index_files
        -d /path/to/DL3/files/
        -o /path/to/DL3/index/files
        -p "dl3*[run_1-run_n]*.fits"
        --overwrite

    Or if the DL3 files are stored in sub-directories:
    > lstchain_create_dl3_index_files
       -d /path/to/DL3/files/
       -o /path/to/DL3/index/files
       -p "/sub-directory*/dl3*[run_1-run_n]*.fits"
       --overwrite

    Or if the DL3 files are stored in the current directory:
    > lstchain_create_dl3_index_files
       -d ./
       -o ./
       -p "dl3*[run_1-run_n]*.fits"
       --overwrite
    """

    input_dl3_dir = traits.Path(
        help="Input path of DL3 files",
        exists=True,
        directory_ok=True,
        file_ok=False
    ).tag(config=True)

    file_pattern = traits.Unicode(
        help="File pattern to search in the given Path",
        default_value="dl3*.fits"
    ).tag(config=True)

    output_index_path = traits.Path(
        help="Output path for the Index files",
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=False,
        default_value=None
    ).tag(config=True)

    overwrite = traits.Bool(
        help="If True, overwrites existing output file without asking",
        default_value=False,
    ).tag(config=True)

    aliases = {
        ("d", "input-dl3-dir"): "FITSIndexWriter.input_dl3_dir",
        ("o", "output-index-path"): "FITSIndexWriter.output_index_path",
        ("p", "file-pattern"): "FITSIndexWriter.file_pattern",
    }

    flags = {
        "overwrite": (
            {"FITSIndexWriter": {"overwrite": True}},
            "overwrite output files if True",
        )
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hdu_index_filename = "hdu-index.fits.gz"
        self.obs_index_filename = "obs-index.fits.gz"

    def setup(self):

        self.list_files = sorted(self.input_dl3_dir.glob(self.file_pattern))
        if len(self.list_files) == 0:
            raise ToolConfigurationError(
                f"No files found with pattern {self.file_pattern} in {self.input_dl3_dir}"
            )

        for f in self.list_files:
            Provenance().add_input_file(f)

        if not self.output_index_path:
            self.output_index_path = self.input_dl3_dir

        self.hdu_index_file = self.output_index_path / self.hdu_index_filename
        self.obs_index_file = self.output_index_path / self.obs_index_filename

        self.provenance_log = self.output_index_path / (self.name + ".provenance.log")

        if self.hdu_index_file.exists():
            if self.overwrite:
                self.log.warning(f"Overwriting {self.hdu_index_file}")
                self.hdu_index_file.unlink()
            else:
                raise ToolConfigurationError(
                    f"Output file {self.hdu_index_file} already exists,"
                    "use --overwrite to overwrite"
                )

        if self.obs_index_file.exists():
            if self.overwrite:
                self.log.warning(f"Overwriting {self.obs_index_file}")
                self.obs_index_file.unlink()
            else:
                raise ToolConfigurationError(
                    f"Output file {self.obs_index_file} already exists,"
                    " use --overwrite to overwrite"
                )

        self.log.debug("HDU Index file: %s", self.hdu_index_file)
        self.log.debug("OBS Index file: %s", self.obs_index_file)

    def start(self):

        create_hdu_index_hdu(
            self.list_files,
            self.hdu_index_file,
            self.overwrite,
        )
        create_obs_index_hdu(
            self.list_files,
            self.obs_index_file,
            self.overwrite
        )
        self.log.debug("HDULists created for the index files")

    def finish(self):

        Provenance().add_output_file(self.hdu_index_file)
        Provenance().add_output_file(self.obs_index_file)


def main():
    tool = FITSIndexWriter()
    tool.run()


if __name__ == "__main__":
    main()
