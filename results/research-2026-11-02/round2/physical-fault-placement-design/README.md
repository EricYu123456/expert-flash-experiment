# QCFlash: Bulk Bitwise Processing in 3D NAND with Dynamic Sensing and Quad-level Encoding

Measurement-driven functional simulator and example datasets for **QCFlash**, a framework for bulk bitwise processing in commercial 3D QLC NAND flash using dynamically shifted read-reference voltages.

The simulator reconstructs NAND page-read behavior from experimentally measured threshold-voltage (`Vth`) data and evaluates the following four-operand bitwise operations:

- AND
- OR
- NAND
- NOR
- XOR
- XNOR

AND, OR, NAND, and NOR are implemented using shifted page reads. XOR and XNOR are implemented using the soft-bit-read (SBR) model.

---

## Dataset

The included workbooks contain experimentally collected threshold-voltage values corresponding to programmed shared-page data from individual wordlines of a Micron 96-layer QLC NAND flash memory:

```text
MT29F2T08GELBEJ4
```

Included datasets:

```text
data/vth_states_wl10.xlsx
data/vth_states_wl11.xlsx
data/vth_states_wl12.xlsx
```

Each workbook represents one measured wordline. Additional workbooks with the same sheet and column structure can be evaluated without modifying the simulator.

---

## Repository Structure

```text
QCFlash/
├── data/
│   ├── vth_states_wl10.xlsx
│   ├── vth_states_wl11.xlsx
│   └── vth_states_wl12.xlsx
├── results/
│   ├── AND_off8_RBER_sweep.png
│   ├── OR_off4_RBER_sweep.png
│   ├── vth_distribution_wl10.png
│   ├── vth_distribution_wl11.png
│   ├── result_samples.txt
│   └── README.md
├── qcflash.py
├── run.py
├── qcflash_simulator.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
├── DATA_LICENSE
└── README.md
```

---

## Tested Environments

The simulator was tested using both Windows and Linux command-line environments.

### Windows Environment

- Processor: Intel Core i7-14700
- Memory: 32 GB RAM DDR5
- Operating system: Windows 11
- Storage: 1 TB NVMe SSD
- Windows build: 22631
- Python: 3.12.10

### Linux Environment

- Processor: Intel Core i7-14700
- Memory: 32 GB DDR5
- Storage: 1 TB NVMe SSD
- Operating system: Ubuntu 22.04.5 LTS
- Python: 3.12.10

---

## Installation

### Windows Command Prompt

Open Command Prompt and move to the extracted repository:

```bat
cd /d "C:\Users\QCFlash"
```

Create a virtual environment:

```bat
py -m venv .venv
```

Activate it:

```bat
.venv\Scripts\activate
```

Install the required packages:

```bat
py -m pip install -r requirements.txt
```

If the `python` command is available instead of `py`, the following can also be used:

```bat
python -m pip install -r requirements.txt
```

### Linux or macOS

Move to the repository:

```bash
cd QCFlash
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

---

## Run One Operation

Example using fixed offsets:

```bash
python run.py --excel data/vth_states_wl10.xlsx --operation AND --offset-mode fixed
```

Supported operations:

```text
AND
OR
NAND
NOR
XOR
XNOR
ALL
```

For every operation, the command-line interface displays:

- The expected result bytes in hexadecimal
- The QCFlash-computed bytes in hexadecimal
- The resulting raw bit error rate (RBER)

Example output:

```text
Expected Result: 0x00 0x10 0x02 0x00 0x40 ...

QCFlash Result:  0x00 0x10 0x02 0x00 0x40 ...

RBER(%): 0.0
```

---

## Use Optimal Read Offsets

Optimal offset mode is available for AND, OR, NAND, and NOR:

```bash
python run.py --excel data/vth_states_wl10.xlsx --operation OR --offset-mode optimal
```

The simulator sweeps read-reference offsets in `0.01 V` steps using the optimization method implemented in the notebook.

---

## Run All Operations

To run all six operations:

```bash
python run.py --excel data/vth_states_wl10.xlsx --operation ALL --offset-mode fixed
```

Windows Command Prompt:

```bat
py run.py --excel data\vth_states_wl10.xlsx --operation ALL --offset-mode fixed
```

The `ALL` mode executes the operations in the following order:

```text
AND
OR
NAND
NOR
XOR
XNOR
```

After execution, a summary of the RBER for all operations is displayed.

Example:

```text
========================================================================
QCFlash RBER summary
========================================================================
AND : 0.000000%
OR  : 0.003000%
NAND: 0.050000%
NOR : 0.100000%
XOR : 0.200000%
XNOR: 0.200000%
```

The numerical results depend on the selected measured wordline dataset and offset configuration.

---

## Jupyter Notebook

The notebook provides a complete interactive demonstration:

```text
qcflash_simulator.ipynb
```

It includes:

- Loading measured QLC threshold-voltage data
- Reading native LSB, CSB, MSB, and TSB pages
- Running all six QCFlash operations
- Displaying expected and computed bytes in hexadecimal
- Calculating RBER
- Sweeping read-reference offsets
- Plotting threshold-voltage distributions
- Generating representative result files and figures

Start Jupyter from the repository root so that relative dataset paths work correctly:

### Windows

```bat
cd /d "C:\Users\QCFlash"
jupyter notebook
```

### Linux or macOS

```bash
cd QCFlash
jupyter notebook
```

Then open:

```text
qcflash_simulator.ipynb
```

The repository must remain structured so that the notebook can locate:

```text
data/vth_states_wl10.xlsx
data/vth_states_wl11.xlsx
data/vth_states_wl12.xlsx
```

---

## Main Files

### `qcflash.py`

Contains the QCFlash simulator functions extracted from the submitted notebook without changing the original computation logic.

### `run.py`

Command-line entry point for running one operation or all supported operations.

### `qcflash_simulator.ipynb`

Interactive notebook containing the full example workflow, plots, offset sweeps, and operation results.

### `data/`

Contains the publicly released measured QLC datasets.

### `results/`

Contains representative result samples and generated figures.

### `requirements.txt`

Lists the required Python packages.


---

## Implemented Functions

The simulator provides the following functions:

```python
read_qlc_page()
find_optimal_offsets()
qcflash_compute()
qlc_compute_sbr()
plot_offset_sweep()
plot_vth_distribution()
```

### `read_qlc_page()`

Reconstructs native QLC logical-page reads using measured cell threshold voltages and read-reference positions.

### `find_optimal_offsets()`

Sweeps read-reference offsets and identifies offset values that minimize operation-level RBER.

### `qcflash_compute()`

Performs shifted-read QCFlash operations:

```text
AND
OR
NAND
NOR
```

### `qlc_compute_sbr()`

Performs SBR-based QCFlash operations:

```text
XOR
XNOR
```

### `plot_offset_sweep()`

Plots operation RBER as a function of read-offset voltage.

### `plot_vth_distribution()`

Plots measured state-wise QLC threshold-voltage distributions and read-reference positions.

---

## Workbook Structure

Each compatible workbook contains measured cell states, threshold-voltage values, read-reference positions, programmed shared-page bytes, and state-wise threshold-voltage data.

The simulator directly uses the following worksheets:

```text
Index_wise Vth
Reference Positions
shared page data
state_wise_vth
```

The main fields include:

- Cell index
- Programmed QLC state
- Measured threshold voltage
- Read-reference position
- LSB programmed data
- CSB programmed data
- MSB programmed data
- TSB programmed data

The four logical pages correspond to shared-page operands programmed into a single physical QLC wordline.

---

## Using Dataset

A new workbook can be evaluated: 

Example:

```bash
python run.py --excel data/new_wordline_dataset.xlsx --operation ALL --offset-mode fixed
```
---

## License

The source code in this artifact is licensed under the MIT License.
See the `LICENSE` file.

The datasets in the `data/` directory are licensed under the Creative
Commons Attribution 4.0 International License (CC BY 4.0).
See the `DATA_LICENSE` file.