# Proof of Concept: R0V to DL0 Converter for LST Data

## Overview

This project aims to convert LST-1 data from R0V format to the standardized DL0 format for long-term storage and future analysis within the CTA Data Processing Pipeline System (DPPS). This PoC outlines the necessary changes in the `cta-lstchain` repository, focusing on waveform calibration, muon event identification, and adhering to the DL0 naming scheme.

---

## Goals

1. **R0V to DL0 Converter:**
   - Convert R0V data to DL0 format, including waveform calibration and pixel retention.
   - Ensure compatibility with future reanalysis by DPPS, using "level-A" calibration.

2. **Muon Event Handling:**
   - Detect and label muon candidates in the DL0 file with the `event_type` set to "muon."
   - Save all pixels for muon events in line with the LST Data Volume Reduction (DVR) scheme.

3. **DL0 Naming Scheme:**
   - Implement the file naming scheme outlined in the ACADA-to-DPPS ICD (page 17 of "ACADA-DPPS-I-160 DL0 naming scheme").

---

## Key Features

### 1. Waveform Calibration

- **Low-Level Calibration**: Apply DRS4 corrections to raw waveforms.
- **High-Level Calibration**: Convert calibrated waveforms to photoelectrons (similar to producing R1 waveforms).
- **Existing Resource**: Modify and adapt the example notebook [`get_LST_R1.ipynb`](https://github.com/cta-observatory/cta-lstchain/blob/master/lstchain/scripts/lstchain_dvr_pixselector.py#L227-L233) to output DL0 waveforms.

### 2. Muon Event Identification

**Two Methods:**
- **Method 1 (Faster)**:
  - Use the existing muon `.fits` file (e.g., `/fefs/aswg/data/real/DL1/20240914/v0.10/muons/muons_LST-1.Run19072.0000.fits`).
  - Extract `event_id`s of muon candidates and check that all pixels are saved in the R0V file.
  - Set the `event_type` to "muon" in the DL0 file and retain all pixels for these events.
  
- **Method 2 (Recalculate Image Parameters)**:
  - During waveform calibration, recalculate image parameters and apply the same selection used in [`lstchain_dvr_pixselector.py`](https://github.com/cta-observatory/cta-lstchain/blob/master/lstchain/scripts/lstchain_dvr_pixselector.py#L227-L233) or an improved version.
  - Mark events that meet the criteria as muon candidates and set the `event_type` to "muon."

**Considerations**:
- **Method 1** is faster as it avoids recalculating parameters, but **Method 2** may be more robust for improving the selection process.

### 3. DL0 File Naming

- Follow the DL0 naming scheme specified in the "ACADA-DPPS-I-160 DL0 naming scheme" on page 17 of the [ACADA-to-DPPS ICD](https://redmine.cta-observatory.org/dmsf/files/17757/view).
- Example components of the naming scheme:
  - Telescope identifier (e.g., LST-1)
  - Run number and subrun identifier
  - File version
  - Event types (e.g., muon events, interleaved events)

---

## Implementation Plan

### Step 1: Implement Waveform Calibration

- Adapt the existing R1 waveform calibration code to output DL0 files.
- Ensure that both low-level (DRS4 corrections) and high-level (conversion to photoelectrons) calibrations are applied.
  - Example notebook for R1 waveform calibration: [`get_LST_R1.ipynb`](https://github.com/cta-observatory/cta-lstchain/blob/master/lstchain/scripts/lstchain_dvr_pixselector.py#L227-L233).

### Step 2: Add Muon Event Detection

- **Method 1**: Implement logic to read muon `.fits` files, cross-check `event_id`s, and label muon candidates in the DL0 file.
  - Example muon `.fits` file path: `/fefs/aswg/data/real/DL1/20240914/v0.10/muons/muons_LST-1.Run19072.0000.fits`.
- **Method 2**: Optionally, implement recalculation of image parameters and selection criteria for muon candidates, referring to the relevant [lstchain script](https://github.com/cta-observatory/cta-lstchain/blob/master/lstchain/scripts/lstchain_dvr_pixselector.py#L227-L233).
- Save all pixels for muon events in accordance with lstchain's DVR scheme.

### Step 3: Apply DL0 Naming Scheme

- Implement the DL0 naming scheme in the file-writing logic.
- Ensure that the file names follow the standards set in the [ACADA-to-DPPS ICD](https://redmine.cta-observatory.org/dmsf/files/17757/view) for long-term compatibility.

### Step 4: Test and Validate

- Run the converter on a subset of R0V data.
- Validate the correctness of the DL0 files:
  - Verify that waveforms are correctly calibrated.
  - Confirm that muon candidates are accurately identified and stored in DL0 with the correct pixel retention and `event_type`.
  - Check that file names adhere to the DL0 naming scheme.

### Step 5: Optimization and Refinement

- Iterate on the selection criteria for muon candidates (if using Method 2).
- Refine data storage strategies for interleaved events and muon candidates.

---

## Conclusion

This PoC outlines the major steps and features required to convert LST-1 data from R0V to DL0 format. By following these steps, the project will be ready for integration with the CTA DPPS for long-term data analysis and storage.

