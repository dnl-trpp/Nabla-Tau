## Nabla-Tau
Official Code and experiments for paper "Gradient-based and Task-Agnostic Machine Unlearning"


### Instructions

#### Setup

All experiments were run with `Python v3.10.12`

To execute the experiments jupyter notebook is needed. If not installed, you can install it with

~~~bash
pip install notebook
~~~

#### Run

To run the experiments, from the current folder (that also contains this README) , open the notebook environment with:

~~~bash
jupyter notebook
~~~

Open the experiment you want to run. First and second cell contain parameters the hyperparameters that can be adjusted (For example `SEED` and `SPLIT`)
When ready, execute with `Run -> Run All Cells`

> Note: Run the three TRAIN procedures first to produce the initial model checkpoints

#### Results

Training checkpoints will be stored in the `Machine_Unlearning` Folder.
All results will be stored as json files in the corresponding folder inside the `Machine_Unlearning_Drive` Folder.
Result files contain a readout of all metrics used in the paper, for each baseline.

#### File Organization

Folder follows the following file structure:

* TRAIN_{Dataset}.ipynb: Where {Dataset} is one of the three used dataset. This file contains the initial training procedure
* RETRAIN_{Dataset}.ipynb: Where {Dataset} is one of the three used dataset. This file contains the golden baseline of retraining from scratch procedure
* Unlearning_main_{EXP_NAME}.ipynb: Where {EXP_NAME} is one of the four experiments. This file contains the unlearning procedure with all corresponding baselines.
* SSD and SCRUB (Folder): This folder contain the original implementation for said baselines
* Machine_Unlearning_Drive (Folder): Contains all results for all experiments organized in subfolders.
* Machine_Unlearning (Folder): Contains additional files and checkpoints
