"""
Saving Find All Bragg Disks results: the BraggVectors themselves, and a text
file of the parameters that produced them, for reuse outside the GUI.
"""

import os
import pprint
from datetime import datetime
from typing import Optional

import h5py
import numpy as np
import py4DSTEM

from py4D_browser.menu_actions import get_ND

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py4D_browser import DataViewer

# The extensions py4D_browser.menu_actions.load_file reads as HDF5.
H5_EXTENSIONS = (".h5", ".hdf5", ".py4dstem", ".emd", ".mat")


def loaded_filepath(parent: "DataViewer") -> Optional[str]:
    """
    The file the current dataset was loaded from, if known. py4D_browser
    doesn't record this anywhere other than the window title, which
    load_file sets to the path (and set_datacube, used by other plugins, to
    something else).
    """
    title = parent.windowTitle()
    return os.path.abspath(title) if os.path.isfile(title) else None


def find_source_emd_group(parent: "DataViewer") -> Optional[tuple[str, str]]:
    """
    Find the (filepath, group) the current dataset was natively loaded from,
    if it came from a py4DSTEM/EMD HDF5 file -- the only kind py4DSTEM.save
    can append to. Locates the group the same way load_file does (the first
    4D dataset in the file), and checks it's the same shape as the loaded
    data in case the window title is stale.
    """
    filepath = loaded_filepath(parent)
    if filepath is None or os.path.splitext(filepath)[-1].lower() not in H5_EXTENSIONS:
        return None

    try:
        with h5py.File(filepath, "r") as file:
            datacubes = get_ND(file)
            if len(datacubes) == 0 or datacubes[0].shape != parent.datacube.data.shape:
                return None
            group = "/".join(datacubes[0].name.split("/")[:-1])
            if len(group) > 1 and "emd_group_type" in file[group].attrs:
                return filepath, group
    except OSError:
        pass
    return None


def has_node(filepath: str, datapath: str) -> bool:
    with h5py.File(filepath, "r") as file:
        return datapath in file


def append_braggvectors(braggvectors, filepath: str, group: str, replace: bool):
    """
    Save braggvectors into an existing py4DSTEM/EMD file, under `group`
    (i.e. as a child of the datacube it was detected from), using py4DSTEM's
    own append mode.
    """
    datapath = f"{group}/{braggvectors.name}"
    root_name = braggvectors.root.name
    with h5py.File(filepath, "r") as file:
        file_root_names = [
            k for k in file if file[k].attrs.get("emd_group_type") == "root"
        ]

    if replace:
        # As of emdfile 0.0.16, py4DSTEM.save can't overwrite an existing
        # node in append mode without also overwriting the file's root
        # metadata (appendover mode), so remove the old one first.
        with h5py.File(filepath, "a") as file:
            del file[datapath]

    if root_name in file_root_names:
        # The datacube's tree mirrors the file's (e.g. a file py4DSTEM wrote
        # itself, read back natively): py4DSTEM.save appends by matching
        # the in-memory tree against the file's -- which only lands in the
        # right place if they really do line up.
        if f"/{root_name}{braggvectors._treepath}" != datapath:
            raise ValueError(
                f"The dataset's py4DSTEM tree ({root_name}) doesn't match "
                f"the file's layout at {group}"
            )
        py4DSTEM.save(filepath, braggvectors, mode="a")
    else:
        # Graft onto the file's tree at the datacube's group. (emdfile
        # requires the in-memory root's name to be absent from the file for
        # this -- with emdpath given and a matching root name, it silently
        # writes nothing.)
        py4DSTEM.save(filepath, braggvectors, mode="a", emdpath=group)

    # py4DSTEM.save has silent no-op paths, so never report success on its
    # word alone.
    if not has_node(filepath, datapath):
        raise RuntimeError(f"py4DSTEM.save didn't write anything at {datapath}")


def find_braggvectors_datapath(filepath: str) -> Optional[str]:
    """Find where py4DSTEM.save put a BraggVectors node in a file."""
    found = []

    def visit(name, obj):
        python_class = obj.attrs.get("python_class")
        if python_class in ("BraggVectors", b"BraggVectors") and not found:
            found.append("/" + name)

    with h5py.File(filepath, "r") as file:
        file.visititems(visit)
    return found[0] if found else None


def _to_python(value):
    # numpy scalars repr as e.g. np.float64(4.39), which ast.literal_eval
    # can't read back, so convert everything to plain Python types.
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_python(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _probe_snippet(probe_params) -> list[str]:
    source = probe_params["source"]
    if source == "selection":
        rx_min, rx_max, ry_min, ry_max = probe_params["ROI"]
        if (rx_max - rx_min) * (ry_max - ry_min) == 1:
            # Mirrors probe_kernel_panel._vacuum_probe_from_single_pixel.
            return [
                "# ROI is a single scan position, which get_vacuum_probe can't",
                "# handle (as of py4DSTEM 0.14.18), so apply the same masking",
                "# it would to that one pattern directly:",
                "import numpy as np",
                "from scipy.ndimage import (",
                "    binary_dilation, binary_opening, distance_transform_edt",
                ")",
                'rx, _, ry, _ = params["probe"]["ROI"]',
                'mask_params = params["probe"]["get_vacuum_probe"]',
                "probe = datacube.data[rx, ry].astype(np.float64)",
                'mask = probe > probe.max() * mask_params["threshold"]',
                'mask = binary_opening(mask, iterations=mask_params["opening"])',
                "mask = binary_dilation(mask, iterations=1)",
                "mask = np.cos((np.pi / 2) * np.minimum(",
                '    distance_transform_edt(~mask) / mask_params["expansion"], 1',
                ")) ** 2",
                "probe = py4DSTEM.Probe(probe * mask)",
            ]
        return [
            "import numpy as np",
            'rx_min, rx_max, ry_min, ry_max = params["probe"]["ROI"]',
            "ROI = np.zeros(datacube.Rshape, dtype=bool)",
            "ROI[rx_min:rx_max, ry_min:ry_max] = True",
            "probe = datacube.get_vacuum_probe(",
            '    ROI=ROI, **params["probe"]["get_vacuum_probe"]',
            ")",
        ]
    if source == "vacuum_file":
        return [
            "# py4DSTEM.read instead, for py4DSTEM/EMD files",
            'vacuum = py4DSTEM.import_file(params["probe"]["filepath"])',
            'probe = vacuum.get_vacuum_probe(**params["probe"]["get_vacuum_probe"])',
        ]
    return [
        "probe = py4DSTEM.Probe.generate_synthetic_probe(",
        '    **params["probe"]["generate_synthetic_probe"]',
        ")",
    ]


def write_params_file(
    h5_filepath: str,
    data: dict,
    probe_params: dict,
    kernel_params: dict,
    detect_params: dict,
    braggvectors_datapath: str,
) -> str:
    """
    Write the parameters behind a Find All Bragg Disks run next to the HDF5
    file its results were saved in, as a Python dict literal (so it can be
    pasted into Python as-is, or loaded with ast.literal_eval), with a
    header explaining how to reuse it with py4DSTEM.
    """
    params_filepath = os.path.splitext(h5_filepath)[0] + "_disk_detection_params.txt"

    params = _to_python(
        {
            "data": data,
            "probe": probe_params,
            "kernel": kernel_params,
            "find_Bragg_disks": detect_params,
            "braggvectors": {
                "filepath": os.path.abspath(h5_filepath),
                "datapath": braggvectors_datapath,
            },
        }
    )

    if data.get("datapath") is not None:
        read_datacube = [
            "datacube = py4DSTEM.read(",
            '    params["data"]["filepath"], datapath=params["data"]["datapath"]',
            ")",
        ]
    else:
        read_datacube = ["datacube = ...  # the same dataset, as a py4DSTEM DataCube"]

    usage = [
        "import ast",
        "import py4DSTEM",
        "",
        f"with open({os.path.basename(params_filepath)!r}) as f:",
        "    params = ast.literal_eval(f.read())",
        "",
        "# the saved results",
        "braggvectors = py4DSTEM.read(",
        '    params["braggvectors"]["filepath"],',
        '    datapath=params["braggvectors"]["datapath"],',
        ")",
        "",
        "# or, to rerun the same detection",
        *read_datacube,
        *_probe_snippet(params["probe"]),
        'probe.get_kernel(**params["kernel"])',
        "braggvectors = datacube.find_Bragg_disks(",
        '    template=probe.kernel, **params["find_Bragg_disks"]',
        ")",
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        "Bragg disk detection parameters, saved by the py4D-browser Disk",
        f"Detection plugin on {timestamp}.",
        "",
        "To reuse in Python (e.g. a Jupyter notebook):",
        "",
        *(f"    {line}" if line else "" for line in usage),
    ]

    with open(params_filepath, "w") as f:
        f.write("\n".join(f"# {line}".rstrip() for line in header))
        f.write("\n\n")
        f.write(pprint.pformat(params, sort_dicts=False, width=88))
        f.write("\n")

    return params_filepath
