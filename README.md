# Decentralized Control of a TITO System

This project compares two decentralized controllers for a linear
two-input, two-output plant:

1. Direct input-output pairing
2. Cross input-output pairing

## Control architecture

The controller is connected in series with the plant under negative
unity feedback. The analysis and control is made using the frequency 
domain framework Individual Channel Anlysis and Design (ICAD).

## Signal convention

- `r1`, `r2`: reference inputs
- `e1`, `e2`: control errors
- `u1`, `u2`: manipulated plant inputs
- `y1`, `y2`: plant outputs

## Requirements

This project requires Python 3 and the following libraries:

- `control`: modeling, analysis, and simulation of control systems
- `matplotlib`: visualization of system responses
- `numpy`: numerical operations and array handling
- `scipy`: scientific and numerical computation
- `slycot`: numerical routines used by the Python Control Systems Library

The complete list of dependencies is available in `requirements.txt`.

## Creating a virtual environment

Using a virtual environment is recommended to keep the project dependencies
isolated from the main Python installation.

### Windows

Create the virtual environment:

```bash
python -m venv .venv

python -m pip install -r requirements.txt

## Important note about Slycot 

`Slycot` contains compiled numerical routines, so its installation can occasionally require additional compilers or system dependencies. Try the normal installation first: 

```bash python -m pip install -r requirements.txt