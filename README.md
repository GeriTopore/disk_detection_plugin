# disk_detection_plugin
This repository contains a plugin for py4d_browser to enable interactive probe/kernel generation and disk detection of 4D-STEM data.

## Installation
This is an external plugin for [py4D-browser](https://github.com/py4dstem/py4D-browser) -- install py4D-browser first, then install this on top of it:
```bash
pip install git+https://github.com/GeriTopore/disk_detection_plugin
```
The plugin is discovered automatically the next time you launch `py4DGUI`, and appears as "Disk Detection..." under the Plugins menu.

## Usage
1. Load a 4D-STEM dataset in the main browser window.
2. Open Plugins > Disk Detection...
3. Choose a probe source (a rectangular or point selection on the current dataset, a separate vacuum file, or a synthetic probe) and generate a probe; its kernel is generated along with it and the Bragg disk previews update straight away. Change the kernel mode/parameters and regenerate the kernel as needed.
4. Tune detection parameters against one or more preview positions ("Update Previews" applies them), then use "Find All Bragg Disks" to run detection across the whole dataset.

### Saved results
"Find All Bragg Disks" saves:
- **The Bragg vectors.** If the dataset was loaded from a py4DSTEM/EMD HDF5 file, they're added to that same file (via py4DSTEM's own append mode), as a `braggvectors` node under the datacube -- you'll be asked before any previous results there are replaced. Other formats can't be appended to, so you'll instead be asked where to export the dataset and its results as a new py4DSTEM HDF5 file.
- **The parameters used**, as `<file>_disk_detection_params.txt` next to that HDF5 file: the probe, kernel and `find_Bragg_disks` settings as a Python dict, with a header showing how to load them (`ast.literal_eval`) and reuse them with py4DSTEM -- e.g. in a Jupyter notebook, to read back the saved Bragg vectors or rerun the same detection.

Loading a new dataset in the main window resets the plugin window, since its probe, kernel and previews belong to the previous dataset. A Find All Bragg Disks run in progress is stopped without saving.

Tick **Live Update** to have any probe, kernel or detection parameter change regenerate everything downstream of it immediately, and to update the previews continuously while dragging their position selectors.
