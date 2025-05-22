# Simple Bayesian Optimization of Reaction Conditions

This project provides a simple pipeline for Bayesian optimization (BO) of chemical reactions, combining molecular and tabular data. Molecules are represented using ECFP4 (Morgan) fingerprints, while other experimental conditions are handled as tabular features. The optimization is discrete and can suggest the next best experiment to run, based on your existing data and settings.

---

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- rdkit
- botorch
- gpytorch

You can install the requirements with pip or conda.

---

## How to Use

1. **Prepare Your Data:**
   - Place your experimental data as a CSV file (see `data/example_dataset.csv` for an example).
   - Each row should represent an experiment, with columns for molecular SMILES, conditions, and measured properties.

2. **Configure Settings:**
   - Edit `data/settings.json` to specify:
     - The CSV file to use (`csvFileName`)
     - Which columns correspond to molecules, conditions, and properties
     - Optionally, add or remove new molecules or conditions in the combinatorial search space with the `_add` or `_remove` keys.
   - Example for `_add`:
     ```json
     "_add": {
         "cu_source": ["[Cu]I", "[Cu]Br", "[Cu]Cl"],
         "temperature": [25, 50, 75]
     }
     ```
   - Example for `_remove`:
     ```json
     "_remove": {
         "cu_source": ["C1=CC(=C(C=C1)C(=O)Cl)C(=O)O"],
         "temperature": [300]
     }
     ```


3. **Run the Optimization:**
   - From the project root, simply run:
     ```bash
     python main.py
     ```
   - The script will read your data, build the search space, fit a Gaussian Process model, and print the next recommended experiment (SMILES and conditions).

---
