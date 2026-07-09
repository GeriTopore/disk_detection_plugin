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
3. On the **Probe Kernel** tab, choose a probe source (a rectangular or point selection on the current dataset, a separate vacuum file, or a synthetic probe), generate a probe, choose a kernel mode and generate the kernel, then Accept.
4. On the **Bragg Disk Detection** tab, tune detection parameters against one or more live preview positions, then use "Find All Bragg Disks" to run detection across the whole dataset. Results are attached to the dataset and are saved along with it via py4D-browser's File > Export Datacube > py4DSTEM HDF5.
