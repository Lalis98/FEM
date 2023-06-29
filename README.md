# FEM

Welcome to the FEM Python Scripts Repository! This repository contains a collection of Python scripts for Finite Element Method (FEM) simulations.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [License](#license)
- [Contact](#contact)

## Introduction
This repository aims to provide a set of Python scripts for conducting Finite Element Method (FEM) simulations. The scripts are designed to work in conjunction with a main running script, which is responsible for creating the mesh and executing the simulation.

The main running script serves as the entry point for the FEM analysis, orchestrating the various components such as mesh generation, solver setup, and post-processing. 

Whether you're a beginner or an experienced FEM practitioner, you'll find these scripts useful for running FEM simulations with Python. The clear separation of responsibilities between the main running script and the supporting scripts allows for modularity and flexibility in customizing your FEM workflow.

Feel free to explore the repository and adapt the scripts to suit your specific FEM analysis requirements.

## Installation
To use the scripts in this repository, follow these steps:

1. Clone the repository to your local machine:
```
git clone https://github.com/Lalis98/FEM.git
```
2. Install the required dependencies. Please refer to the `requirements.txt` file for the specific packages and versions needed. You can install them using pip:
```
pip install -r requirements.txt
```

## Usage
To use the scripts in this repository, follow these instructions:

Ensure that you have Python installed on your system. You can download the latest version of Python from the official Python website (https://www.python.org) and follow the installation instructions specific to your operating system.

Install the required dependencies mentioned in the requirements.txt file. Refer to the previous instructions in this README file on how to install the dependencies using pip.

Open a terminal or command prompt and navigate to the directory where the scripts are located.

To run a specific script, use the following command:

```
python script_name.py
```
Replace script_name.py with the name of the script file you want to execute. For example, if you have a script named 2D_Plane_Stress_Q4_FEM.py, the command would be:

```
python 2D_Plane_Stress_Q4_FEM.py
```
The script will start running, and you will see the output or results based on the specific functionality of the script.

## Folder Structure
The repository follows a simple folder structure:

- `scripts/`: This directory contains the individual scripts for different simulations. Each script represents a specific simulation scenario or case. You can run these scripts independently to perform the desired simulation.

  - `requirements.txt`: This file, located within the `scripts/` directory, lists the required Python packages and their versions. You can install these dependencies using `pip` as mentioned earlier.

It is recommended to organize your scripts in a way that makes sense for your project. You can create subdirectories within the `scripts/` directory to group related simulations or categorize them based on specific criteria.

Feel free to adapt the folder structure based on your requirements and add any additional directories or files as needed for your project.


## License
This project is not licensed. All rights are reserved.

## Contact
If you have any questions, suggestions, or feedback, please feel free to reach out to the repository owner, Michalis Lefkiou, at:
- LinkedIn: [Michalis Lefkiou](https://www.linkedin.com/in/michalis-lefkiou/)
- Email: michalis.leukioug1@gmail.com

You can also open an issue in the repository for any specific concerns related to the FEM Python scripts.

