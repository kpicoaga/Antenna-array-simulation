# Antenna Array Simulations for Multi-Band MIMO Systems

This repository contains Python simulations of tightly coupled hexagonal antenna arrays, developed to deepen my understanding of concepts from the paper *"Tightly Coupled Hexagonal Antenna Arrays for Multi-band Massive MIMO Systems"* co-authored by Dr. Mezghani. The simulation model dual-ban arrays and compute system capacity

## Purpose
The code explores mutual impedance and channel characteristics of compact antenna arrays, replicating key aspects of the referenced paper.

## Repository Structure
- `main.py`: Orchestrates the simulation, computing system capacity for a 7x7 hexagonal array across dual bands (3.3–3.7 GHz and 17.3–17.7 GHz) and plotting capacity vs. antenna spacing.
- `Z_Heq_Water.py`: Contains core simulation functions:
  - `build_Z_matrix`: Computes the mutual impedance matrix for the array, including self and mutual impedances.
  - `build_Z_RT_matrix`: Calculates mutual impedances between a receiver and transmit antennas.
  - `compute_Heq_and_Rn`: Derives the equivalent channel matrix (`H_eq`) and noise covariance matrix (`Rn`).
  - `water_filling`: Implements a baseline power allocation algorithm.
- `utils_hex_array.py`: Provides utility functions:
  - `get_hex_positions`: Generates hexagonal array positions.
  - `get_self_impedance`: Computes self-impedance for the TM₁ mode.
  - `compute_mutual_impedance`: Calculates mutual impedance based on distance and angle.
- `eigenvalue_plot.py`: Computes and plot the dominant SNR eigenmode across a wide frequency sweep (2–20 GHz) for a fixed antenna spacing.

 #REFERENCES 
- [1] N. Balasuriya, A. Mezghani, and E. Hossain, “Tightly Coupled Hexagonal Antenna Arrays for Multi-band Massive MIMO Systems,” in Proc. IEEE Wireless Communications Letters, June 2025.


Purpose for studying this paper: The purpose of studying the paper "Tightly Coupled Hexagonal Antenna Arrays for Multi-band Massive MIMO Systems" is to establish a theoretical and simulation foundation for my proposed engineering project, which focuses on energy optimization of multi-band antenna arrays using Deep Reinforcement Learning (DRL). This paper introduces a circuit-theoretic approach to modeling self and mutual impedance in tightly coupled hexagonal arrays, providing a physically realistic framework for capacity evaluation in both near-field and far-field conditions.
