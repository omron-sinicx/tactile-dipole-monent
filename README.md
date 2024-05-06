# Tactile Dipole Monent
This repository contains 3D design files and code used for the ICRA 2024 paper "An Electromagnetism-Inspired Method for Estimating In-Grasp Torque from Visuotactile Sensors".

Link: https://arxiv.org/abs/2404.15626

## Description
- `3D_cad_file` contains mesh and CAD files for the 3D printed jigs used in experiments.
- `code` contains code written for the project.
- The code included here is not meant to be stand-alone, as it requires ROS and other packages to run properly. Instead, it is meant to document the implementation of calculations used in the paper.
- The core calculations presented in the paper are contained in `touch.py``, within `current_contact_state()`.