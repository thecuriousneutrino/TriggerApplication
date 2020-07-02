# Configuration files

***********************
# Description
**********************

Configure files are simple text files for passing variables to the Tools.

Text files are read by the Store class (src/Store) and automatically asigned to an internal map for the relavent Tool to use.


************************
# Usage
************************

Any line starting with a "#" will be ignored by the Store, as will blank lines.

Variables should be stored one per line as follows:


`Name Value # Comments`


Note: Only one value is permitted per name and they are stored in a string stream and templated cast back to the type given.

*******************
# List of toolchains
******************

Useful examples of triggering

| Toolchain       | Description |
| ----            | -----       |
| GPUtest         | Read WCSim ASCII file; Run nhits trigger on GPU; Write out text-file |
| WCSimReaderTest | Read WCSim root file; Run nhits trigger; Write out WCSim-like file |
| WCSimBONSAI     | Read WCSim root file; Run nhits trigger; Write out WCSim-like file; Run BONSAI vertex/direction reconstruction; Run FLOWER energy reconstruction; Write out reconstructed tree; Reset RecoInfo objects |
| SNTriggering    | Reads in BONSAI reconstruction from `reconstructed.root`; Create fake events in ReconRandomiser; Filter events; Run dimfit; Write out reconstructed tree; Reset RecoInfo objects |

Simple examples

| Toolchain       | Description |
| ----            | -----       |
| Dummy           | Run 2 versions of the Dummy print-out tool |
| template        | Run 2 versions of the Dummy print-out tool; used by `Create_run_config.sh`
| test            | Does nothing |
