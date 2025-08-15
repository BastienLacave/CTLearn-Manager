"""
Create DL3 FITS file from given data DL2 file,
selection cuts and IRF FITS files.

Change the selection parameters as need be using the aliases.
The default values are written in the EventSelector and DL3Cuts Component
and also given in some example configs in docs/examples/

For using IRF interpolation methods, to get IRF with sky pointing the same or
closer (in the interpolation parameter space) to that of the data provided,
one has to provide,

- the path to the IRFs, and
- glob search pattern for selecting the IRFs to be used

If instead of using IRF interpolation, one needs to add only the nearest IRF
node to the given data, in the interpolation space, then one needs to pass the
use-nearest-irf-node flag.

For the cuts on gammaness, the Tool looks at the IRF provided or the final
interpolated/selected IRF, to either use global cuts, based on the header
value of the global gammaness cut, GH_CUT, present in each HDU, or
energy-dependent cuts, based on the GH_CUTS HDU.

To use a separate config file for providing the selection parameters,
copy and append the relevant example config files, into a custom config file.

For source-dependent analysis, a source-dep flag should be passed.
Similarly to the cuts on gammaness, the global alpha cut values are provided
from AL_CUT stored in the HDU header. The alpha cut is already applied on this
step, and all survived events with each assumed source position (on and off)
are saved after the gammaness and alpha cut.
To adapt to the high-level analysis used by gammapy, assumed source position
(on and off) is set as a reco source position just as a trick to obtain
survived events easily.
"""

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import vstack, QTable
import astropy.units as u
from ctapipe.core import (
    Provenance,
    Tool,
    ToolConfigurationError,
    traits,
)
import pandas as pd

from lstchain.io import (
    EventSelector,
    DL3Cuts,
    get_srcdep_assumed_positions,
    remove_duplicated_events,
    get_dataset_keys,
)
from lstchain.high_level import (
    check_in_delaunay_triangle,
    compare_irfs,
    create_event_list,
    fill_reco_altaz_w_expected_pos,
    interpolate_irf,
)
from lstchain.paths import (
    dl2_to_dl3_filename,
    run_info_from_filename,
)
# from lstchain.reco.utils import get_effective_time
from lstchain.reco.utils import get_geomagnetic_delta
import numpy as np
from numba import njit


__all__ = ["DataReductionFITSWriter"]

dl2_params_src_dep_lstcam_key = "/dl2/event/telescope/parameters_src_dependent/LST_LSTCam"
dl2_params_lstcam_key = "/dl2/event/telescope/parameters/LST_LSTCam"

def get_effective_time(events, CTLearn=True, time_key='dragon_time'):
    # Extract timestamps and delta_t as numpy arrays
    if CTLearn:
        # print(events[time_key])
        timestamp = np.asarray(events[time_key].to_value("unix"))
    else:
        timestamp = np.asarray(events[time_key])

    delta_t = np.asarray(events["delta_t"])

    # Ensure units are attached only once
    if not isinstance(timestamp, u.Quantity):
        timestamp = timestamp * u.s
    if not isinstance(delta_t, u.Quantity):
        delta_t = delta_t * u.s

    # Fast diff and mask for elapsed time (DAQ breaks >0.01s)
    time_diff = np.diff(timestamp)
    t_elapsed = np.sum(time_diff[time_diff < 0.01 * u.s])

    # Mask for valid delta_t (exclude first event and DAQ breaks)
    valid_delta = (delta_t > 0.0 * u.s) & (delta_t < 0.01 * u.s)
    delta_t_valid = delta_t[valid_delta]

    if delta_t_valid.size == 0:
        # Avoid division by zero if no valid delta_t
        return 0 * u.h, 0 * u.h

    dead_time = np.min(delta_t_valid)
    mean_delta = np.mean(delta_t_valid)
    # Avoid division by zero in rate calculation
    denom = mean_delta - dead_time
    if denom <= 0:
        rate = 0
    else:
        rate = 1 / denom

    t_eff = t_elapsed / (1 + rate * dead_time)
    return t_eff.to(u.h), t_elapsed.to(u.h)

def get_effective_time_old(events):
    """
    Calculate the effective observation time of a set of real data events
    from a sky observation. delta_t (s) must be the time elapsed from the
    previous *triggered* event, regardless of whether the list of events
    contains all triggered events or not. It can be a list only of events
    which e.g. have valid image parameters. Besides delta_t, each event must
    have dragon_time, a timestamp (s)

    Parameters
    ----------
    events: pandas DataFrame or astropy.table.QTable
    If a dataframe, units are assumed to be seconds

    Returns
    -------
    t_eff: astropy Quantity (in seconds, if input has no units)
    t_elapsed: astropy Quantity (ditto)
    """

    # For consistency with the event selection applied in the DL2 to DL3 stage
    # we require the events to be tagged as "physics triggers". In this way we
    # count as livetime only periods in which the telescope is recording showers
    # and properly tagging them. Without this filter the effective time was in 
    # *rare* occasions (example: run 7199) a few % larger than it should be - it was 
    # including periods in which showers were tagged "UNKNOWN", which were not 
    # present in the DL3 file. For most runs the effect is zero.
    
    # typemask = events["event_type"] == EventType.SUBARRAY.value
    
    # timestamp = np.array(events["dragon_time"][typemask])
    # delta_t = np.array(events["delta_t"][typemask])
    timestamp = np.array(events["dragon_time"])
    delta_t = np.array(events["delta_t"])

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

    # Estimate the "true external rate", i.e. what we would see in absence of
    # dead time. For a Poisson process with fixed dead time per event,
    # it can be shown that the expected value of delta_t is
    # <delta_t> = dead_time + 1/rate
    # Note that the formula is not strictly correct if we have interleaved
    # events (pedestal and flatfield) at regular intervals, because the
    # delta_t will never be larger than the time between interleaved
    # events. But this truncation would hardly be noticeable for the typical
    # cosmics rates, and 200 Hz of interleaved events.

    rate = 1 / (np.mean(delta_t) - dead_time)

    t_eff = t_elapsed / (1 + rate * dead_time)

    return t_eff, t_elapsed

@njit
def compute_diff(arr):
    n = len(arr)
    diff = np.empty(n, dtype=arr.dtype)
    diff[0] = 0  # Assuming the first difference is 0
    for i in range(1, n):
        diff[i] = arr[i] - arr[i - 1]
    return diff

def load_DL2_data(input_file):
    tel_id =  1
    reco_method = "CTLearn"
    path = "telescope"
    tel = f"tel_{tel_id:03d}"
    from astropy.table import hstack, join
    from ctapipe.io import read_table

    pointing = read_table(input_file, f"dl1/monitoring/{path}/pointing/{tel}")
    # pointing = read_table_hdf5(input_file, path=f"dl1/monitoring/{path}/pointing/{tel}")
    pointing.sort("time")

    dl2 = None


    dl2_classification = read_table(
        input_file, f"dl2/event/{path}/classification/{reco_method}/{tel}"
    )
    dl2 = dl2_classification


    dl2_energy = read_table(
        input_file, f"dl2/event/{path}/energy/{reco_method}/{tel}"
    )
    dl2 = (
        join(dl2, dl2_energy, keys=["obs_id", "event_id"])
        if dl2 is not None
        else dl2_energy
    )



    dl2_geometry = read_table(
        input_file, f"dl2/event/{path}/geometry/{reco_method}/{tel}"
    )
    dl2 = (
        join(dl2, dl2_geometry, keys=["obs_id", "event_id"])
        if dl2 is not None
        else dl2_geometry
    )


    # dl1 = read_table(input_file, f"dl1/event/{path}/parameters/{tel}")[
    #     ["obs_id", "event_id", "hillas_intensity"]
    # ]
    # dl2 = join(dl2, dl1, keys=["obs_id", "event_id"]) if dl2 is not None else dl1

    dl2.sort("event_id")
    dl2 = hstack([dl2, pointing])
    dl2.sort("time")

    # print("Computing time differences...")
    t_diff = compute_diff(dl2["time"].to_value("unix"))
    dl2["delta_t"] = t_diff

    # print(f"Loaded {len(dl2)} events")
    return dl2

def read_data_dl2_to_QTable(filename, srcdep_pos=None):
    """
    Read data DL2 files from lstchain and return QTable format, along with
    a dict of target parameters for IRF interpolation

    Parameters
    ----------
    filename: path to the lstchain DL2 file
    srcdep_pos: assumed source position for source-dependent analysis

    Returns
    -------
    data: `astropy.table.QTable`
    data_params: 'Dict' of target interpolation parameters
    """

    # Mapping
    name_mapping = {
        "CTLearn_tel_prediction": "gh_score",
        "altitude": "pointing_alt",
        "azimuth": "pointing_az",
        "CTLearn_tel_energy": "reco_energy",
        "CTLearn_tel_alt": "reco_alt",
        "CTLearn_tel_az": "reco_az",
        "time": "dragon_time",
    }
    unit_mapping = {
        # "reco_energy": u.TeV,
        # "pointing_alt": u.rad,
        # "pointing_az": u.rad,
        # "reco_alt": u.rad,
        # "reco_az": u.rad,
        # "dragon_time": u.s,
    }

    # add alpha for source-dependent analysis
    # srcdep_flag = dl2_params_src_dep_lstcam_key in get_dataset_keys(filename)

    # if srcdep_flag:
    #     unit_mapping['alpha'] = u.deg

    # data = pd.read_hdf(filename, key=dl2_params_lstcam_key)
    data = load_DL2_data(filename)

    # if srcdep_flag:
    #     data_srcdep = get_srcdep_params(filename, srcdep_pos)
    #     data = pd.concat([data, data_srcdep], axis=1)
    for old_name, new_name in name_mapping.items():
        if old_name in data.colnames:
            data.rename_column(old_name, new_name)

    # data = data.rename(columns=name_mapping)
    # data = QTable.from_pandas(data)
    data = QTable(data)
    # print(data.colnames)

    # Make the columns as Quantity
    for k, v in unit_mapping.items():
        data[k] *= v

    # Create dict of target parameters for IRF interpolation
    data_params = {}

    zen = np.pi / 2 * u.rad - data["pointing_alt"].mean().to(u.rad)
    az = data["pointing_az"].mean().to(u.rad)
    if az < 0:
        az += 2*np.pi * u.rad
    b_delta = u.Quantity(get_geomagnetic_delta(zen=zen, az=az))

    data_params["ZEN_PNT"] = round(zen.to_value(u.deg), 5) * u.deg
    data_params["AZ_PNT"] = round(az.to_value(u.deg), 5) * u.deg
    data_params["B_DELTA"] = round(b_delta.to_value(u.deg), 5) * u.deg
    # print(data_params)
    return data, data_params


class DataReductionFITSWriter(Tool):
    name = "DataReductionFITSWriter"
    description = __doc__
    example = """
    To generate DL3 file from an observed data DL2 file, using default cuts:
    > lstchain_create_dl3_file
        -d /path/to/DL2_data_file.h5
        -o /path/to/DL3/file/
        -i /path/to/irf/
        --source-name Crab
        --source-ra 83.633deg
        --source-dec 22.01deg

    Or use a config file for the cuts:
    > lstchain_create_dl3_file
        -d /path/to/DL2_data_file.h5
        -o /path/to/DL3/file/
        -i /path/to/irf/
        --source-name Crab
        --source-ra 83.633deg
        --source-dec 22.01deg
        --overwrite
        --config /path/to/config.json

    Or pass the selection cuts from command-line:
    > lstchain_create_dl3_file
        -d /path/to/DL2_data_file.h5
        -o /path/to/DL3/file/
        --input-irf-path /path/to/irf/
        --irf-file-pattern "irf*.fits.gz"
        --source-name Crab
        --source-ra 83.633deg
        --source-dec 22.01deg
        --overwrite

    Or generate source-dependent DL3 files
    > lstchain_create_dl3_file
        -d /path/to/DL2_data_file.h5
        -o /path/to/DL3/file/
        --input-irf-path /path/to/irf
        --irf-file-pattern "irf.fits.gz"
        --source-name Crab
        --source-dep
        --overwrite

    Or use a list of IRFs for including interpolated IRF:
    > lstchain_create_dl3_file
        -d /path/to/DL2_data_file.h5
        -o /path/to/DL3/file/
        -i /path/to/irf/
        -p "irf*.fits.gz"
        --interp-method linear
        --source-name Crab
        --source-ra 83.633deg
        --source-dec 22.01deg
        --overwrite

    Or use a list of IRFs for including only the nearest IRF:
    > lstchain_create_dl3_file
        -d /path/to/DL2_data_file.h5
        -o /path/to/DL3/file/
        -i /path/to/irf/
        -p "irf*.fits.gz"
        --use-nearest-irf-node
        --source-name Crab
        --source-ra 83.633deg
        --source-dec 22.01deg
        --overwrite

    """

    input_dl2 = traits.Path(
        help="Input data DL2 file",
        exists=True,
        directory_ok=False,
        file_ok=True,
    ).tag(config=True)

    output_dl3_path = traits.Path(
        help="DL3 output filedir",
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    input_irf_path = traits.Path(
        help="Path for compressed FITS file of IRFs",
        exists=True,
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    irf_file_pattern = traits.Unicode(
        help="IRF file pattern to search in the given IRF files path",
        default_value="*irf*.fits.gz",
    ).tag(config=True)

    # cuts_file_pattern = traits.Unicode(
    #     help="IRF file pattern to search in the given IRF files path",
    #     default_value="*irf*.fits.gz",
    # ).tag(config=True)

    source_name = traits.Unicode(
        help="Name of Source",
    ).tag(config=True)

    source_ra = traits.Unicode(
        help="RA position of the source",
    ).tag(config=True)

    source_dec = traits.Unicode(
        help="DEC position of the source",
    ).tag(config=True)

    interp_method = traits.Unicode(
        help="Interpolation method to be used, when required",
        default_value="linear",
    ).tag(config=True)

    overwrite = traits.Bool(
        help="If True, overwrites existing output file without asking",
        default_value=False,
    ).tag(config=True)

    # source_dep = traits.Bool(
    #     help="If True, source-dependent analysis will be performed.",
    #     default_value=False,
    # ).tag(config=True)

    keep_duplicated_events = traits.Bool(
        help="If True, duplicated events after alpha and gammaness cut are not removed.",
        default_value=False,
    ).tag(config=True)

    use_nearest_irf_node = traits.Bool(
        help="If True, only look for the nearest IRF node to the data. No interpolation",
        default_value=False,
    ).tag(config=True)

    gzip = traits.Bool(
        help="If True, the DL3 file will be gzipped",
        default_value=False,
    ).tag(config=True)

    classes = [EventSelector, DL3Cuts]

    aliases = {
        ("d", "input-dl2"): "DataReductionFITSWriter.input_dl2",
        ("o", "output-dl3-path"): "DataReductionFITSWriter.output_dl3_path",
        ("i", "input-irf-path"): "DataReductionFITSWriter.input_irf_path",
        ("p", "irf-file-pattern"): "DataReductionFITSWriter.irf_file_pattern",
        # ("cuts", "cuts-file-pattern"): "DataReductionFITSWriter.cuts_file_pattern",
        "interp-method": "DataReductionFITSWriter.interp_method",
        "source-name": "DataReductionFITSWriter.source_name",
        "source-ra": "DataReductionFITSWriter.source_ra",
        "source-dec": "DataReductionFITSWriter.source_dec",
    }

    flags = {
        "overwrite": (
            {"DataReductionFITSWriter": {"overwrite": True}},
            "overwrite output file if True",
        ),
        "source-dep": (
            {"DataReductionFITSWriter": {"source_dep": True}},
            "source-dependent analysis if True",
        ),
        "keep-duplicated-events": (
            {"DataReductionFITSWriter": {"keep_duplicated_events": True}},
            "duplicated events are not removed if True",
        ),
        "use-nearest-irf-node": (
            {"DataReductionFITSWriter": {"use_nearest_irf_node": True}},
            "Only use the closest IRF, if True",
        ),
        "gzip": (
            {"DataReductionFITSWriter": {"gzip": True}},
            "gzip the DL3 files if True",
        ),
    }

    

    def setup(self):

        self.filename_dl3 = dl2_to_dl3_filename(self.input_dl2, compress=self.gzip)
        self.provenance_log = self.output_dl3_path / (self.name + ".provenance.log")

        Provenance().add_input_file(self.input_dl2)

        self.event_sel = EventSelector(parent=self, filters = {})
        self.cuts = DL3Cuts(parent=self)

        self.output_file = self.output_dl3_path.absolute() / self.filename_dl3
        if self.output_file.exists():
            if self.overwrite:
                self.log.warning(f"Overwriting {self.output_file}")
                self.output_file.unlink()
            else:
                raise ToolConfigurationError(
                    f"Output file {self.output_file} already exists,"
                    " use --overwrite to overwrite"
                )

        if self.input_irf_path:
            self.irf_list = sorted(
                self.input_irf_path.glob(self.irf_file_pattern)
            )
            # self.cuts_file = sorted(self.input_irf_path.glob(self.cuts_file_pattern))[0]
            # print(self.cuts_file)

            if len(self.irf_list) > 1:
                self.use_irf_interpolation = True
                # Compare the IRFs for its metadata and cuts
                if not compare_irfs(self.irf_list):
                    raise ToolConfigurationError(
                        f"IRF files in {self.input_irf_path} with pattern, "
                        f"{self.irf_file_pattern} are not similar and cannot"
                        " be used to interpolate. Use different list of IRFs."
                    )

            elif len(self.irf_list) == 1:
                self.use_irf_interpolation = False
                self.log.info(
                    f"Only single IRF {self.irf_list[0]} provided."
                    " No interpolation possible"
                )
                self.irf_final_hdu = fits.open(self.irf_list[0])
                self.irf_final = self.irf_list[0]
            else:
                raise ToolConfigurationError(
                    f"No IRF files in {self.input_irf_path} with pattern, "
                    f"{self.irf_file_pattern} found. Use different parameters"
                )

        if not (self.source_ra or self.source_dec):
            self.source_pos = SkyCoord.from_name(self.source_name)
        elif bool(self.source_ra) != bool(self.source_dec):
            raise ToolConfigurationError(
                "Either provide both RA and DEC values for the source or none"
            )
        else:
            self.source_pos = SkyCoord(ra=self.source_ra, dec=self.source_dec)

        self.log.debug(f"Output DL3 file: {self.output_file}")

    def interp_irfs(self):
        """
        Get the optimal number of IRFs necessary for interpolation
        IF the target parameter is not inside a simplex formed by
        the given list of IRFs, use the nearest grid point.
        """

        self.irf_list = check_in_delaunay_triangle(
            self.irf_list, self.data_params
        )

        if len(self.irf_list) > 1:
            self.log.info(
                f"Paths of IRFs used for interpolation: {self.irf_list}"
            )
            self.irf_final_hdu = interpolate_irf(
                self.irf_list, self.data_params, self.interp_method
            )
        else:
            self.irf_final_hdu = fits.open(self.irf_list[0])
            self.log.info(
                f"Nearest IRF {self.irf_list[0]} is used without interpolation"
            )

    def check_energy_dependent_cuts(self):
        """
        Check if the final IRF has energy-dependent gammaness cuts or not.
        """
        try:
            self.use_energy_dependent_gh_cuts = (
                "GH_CUT" not in self.irf_final_hdu["EFFECTIVE AREA"].header
            )
        except KeyError:
            raise ToolConfigurationError(
                f"{self.irf_final_hdu} does not have EFFECTIVE AREA HDU, "
                " to check for global cut information in the Header value"
            )

        if self.source_dep:
            self.use_energy_dependent_alpha_cuts = (
                "AL_CUT" not in self.irf_final_hdu["EFFECTIVE AREA"].header
            )

    def apply_srcindep_gh_cut(self):
        """
        Apply gammaness cut.
        """
        self.data = self.event_sel.filter_cut(self.data)

        if self.use_energy_dependent_gh_cuts:
            self.energy_dependent_gh_cuts = QTable.read(
                self.irf_final_hdu["GH_CUTS"]
            )

            self.data = self.cuts.apply_energy_dependent_gh_cuts(
                self.data, self.energy_dependent_gh_cuts
            )
            # print(self.energy_dependent_gh_cuts.meta['GH_EFF'])
            self.log.info(
                "Using gamma efficiency of "
                f"{self.energy_dependent_gh_cuts.meta['GH_EFF']}"
            )
        else:
            self.cuts.global_gh_cut = self.irf_final_hdu[1].header["GH_CUT"]
            self.data = self.cuts.apply_global_gh_cut(self.data)
            self.log.info(f"Using global G/H cut of {self.cuts.global_gh_cut}")

    def apply_srcdep_gh_alpha_cut(self):
        """
        Apply gammaness and alpha cut for source-dependent analysis.
        """
        srcdep_assumed_positions = get_srcdep_assumed_positions(self.input_dl2)

        for i, srcdep_pos in enumerate(srcdep_assumed_positions):
            data_temp, _ = read_data_dl2_to_QTable(
                self.input_dl2, srcdep_pos=srcdep_pos
            )

            data_temp = self.event_sel.filter_cut(data_temp)

            if self.use_energy_dependent_gh_cuts:
                self.energy_dependent_gh_cuts = QTable.read(
                    self.irf_final_hdu["GH_CUTS"]
                )

                data_temp = self.cuts.apply_energy_dependent_gh_cuts(
                    data_temp, self.energy_dependent_gh_cuts
                )
                self.log.info(
                    "Using gamma efficiency of "
                    f"{self.energy_dependent_gh_cuts.meta['GH_EFF']}"
                )
            else:
                self.cuts.global_gh_cut = self.irf_final_hdu[1].header["GH_CUT"]
                data_temp = self.cuts.apply_global_gh_cut(data_temp)

            if self.use_energy_dependent_alpha_cuts:
                self.energy_dependent_alpha_cuts = QTable.read(
                    self.irf_final_hdu["AL_CUTS"]
                )
                data_temp = self.cuts.apply_energy_dependent_alpha_cuts(
                    data_temp, self.energy_dependent_alpha_cuts
                )
                self.log.info(
                    "Using alpha containment region of "
                    f'{self.energy_dependent_alpha_cuts.meta["AL_CONT"]}'
                )
            else:
                self.cuts.global_alpha_cut = self.irf_final_hdu[1].header["AL_CUT"]
                data_temp = self.cuts.apply_global_alpha_cut(data_temp)

            # Fill the reco alt/az positions with expected source positions
            data_temp = fill_reco_altaz_w_expected_pos(data_temp)

            if i == 0:
                self.data = data_temp
            else:
                self.data = vstack([self.data, data_temp])

        if not self.keep_duplicated_events:
            if len(srcdep_assumed_positions) > 2:
                self.log.warning(
                    "If multiple off positions are assumed, the process to "
                    "remove duplicated events can introduce a bias"
                )
            n_events_before = len(self.data)

            remove_duplicated_events(self.data)
            n_events_after = len(self.data)

            duplicated_events_ratio = (n_events_before - n_events_after)/n_events_after
            self.log.info(
                "Remove duplicated events: a ratio of duplicated events is "
                f"{duplicated_events_ratio}"
            )

        # Sort the data frame based on event_id
        self.data.sort('event_id')

    def start(self):
        self.source_dep = False

        if not self.source_dep:
            self.data, self.data_params = read_data_dl2_to_QTable(self.input_dl2)
        else:
            self.data, self.data_params = read_data_dl2_to_QTable(self.input_dl2, "on")

        if self.use_irf_interpolation:
            if not self.use_nearest_irf_node:
                self.interp_irfs()
            else:
                ## Check
                self.irf_final_hdu = fits.open(
                    check_in_delaunay_triangle(
                        self.irf_list, self.data_params, self.use_nearest_irf_node
                    )[0]
                )
        self.check_energy_dependent_cuts()

        self.effective_time, self.elapsed_time = get_effective_time(self.data)
        self.run_number = run_info_from_filename(self.input_dl2)[1]

        if not self.source_dep:
            self.apply_srcindep_gh_cut()
        else:
            self.apply_srcdep_gh_alpha_cut()

        self.log.info("Generating event list")
        self.events, self.gti, self.pointing = create_event_list(
            data=self.data,
            run_number=self.run_number,
            source_name=self.source_name,
            source_pos=self.source_pos,
            effective_time=self.effective_time.value,
            elapsed_time=self.elapsed_time.value,
            data_pars=self.data_params,
        )
        self.log.info(f"Target parameters for interpolation: {self.data_params}")

        self.hdulist = fits.HDUList(
            [fits.PrimaryHDU(), self.events, self.gti, self.pointing]
        )

        self.log.info("Adding IRF HDUs")
        self.mc_params = dict()

        h = self.irf_final_hdu[1].header
        # print(self.data_params)
        # print(self.irf_final_hdu[1].header)
        # for p in self.data_params.keys():
        #     self.mc_params[p] = u.Quantity(h[p], "deg")

        # self.log.info(
        #     f"Zenith pointing of MC at {self.mc_params['ZEN_PNT']:.3f}"
        # )
        # self.log.info(
        #     f"Azimuth pointing of MC at {self.mc_params['AZ_PNT']:.3f}"
        # )
        # self.log.info(
        #     f"Geomagnetic delta for the MC is {self.mc_params['B_DELTA']:.3f}"
        # )
        # print(self.irf_final_hdu.info())
        for irf_hdu in self.irf_final_hdu[1:]:
            self.hdulist.append(irf_hdu)

    def finish(self):

        self.hdulist.writeto(self.output_file, overwrite=self.overwrite)

        Provenance().add_output_file(self.output_file)


def main():
    tool = DataReductionFITSWriter()
    tool.run()


if __name__ == "__main__":
    main()
