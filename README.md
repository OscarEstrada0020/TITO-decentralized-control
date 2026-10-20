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

## Installation

```bash
python -m venv .venv