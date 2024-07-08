# Labrador: Exploring the Limits of Masked Language Modeling for Laboratory Data

[**Get Started**](#get-started)
| [**arXiv paper**]()
| [**HuggingFace Models**](https://huggingface.co/Drbellamy/labrador)
| [**Data**](#data)
| [**License**](#license)

![Visual Abstract](img/Lab_Transformer_Visual_Abstract.png#gh-light-mode-only)
![Visual Abstract](img/Lab_Transformer_Visual_Abstract_dark.png#gh-dark-mode-only)

Labrador is a pre-trained continuous Transformer model for masked ***lab*** modeling.

Laboratory data are a rich source of information about a patient's health. They are often used to diagnose and monitor disease, and to guide treatment. However, lab values are continuous, often missing and therefore difficult to model with the Transformer architecture.

Labrador solves this problem by jointly embedding lab values with a token for the lab test identifier so that the quantitative and qualitative information from each test is combined into a single representation. 

Labrador is pre-trained on a large corpus of 100 million lab tests from over 260,000 patients. We rigorously evaluate Labrador on intrinsic and extrinsic tasks, including four real-world problems: cancer diagnosis, COVID-19 diagnosis, predicting elevated alcohol consumption and ICU mortality due to sepsis. We find that Labrador is superior to BERT across all evaluations but both are outperformed by XGBoost indicating that transfer learning from continuous EHR data is still an open problem. 

We discuss the limitations of our approach and suggest future directions for research in the corresponding paper, [Labrador: Exploring the Limits of Masked Language Modeling for Laboratory Data](). 

Pre-trained Labrador and BERT models are available for download from the [Hugging Face model hub](https://huggingface.co/Drbellamy/labrador).

# Get Started

**System packages**

This repository requires `git`, `make` and `Python3.10` or later. You can install these using several methods. We recommend `brew` ([install guide](https://brew.sh/)), which can be installed directly for MacOS and Linux users, whereas Windows users should first install and activate [WSL-Ubuntu](https://learn.microsoft.com/en-us/windows/wsl/install).

We recommend using [pyenv](https://github.com/pyenv/pyenv) to manage Python versions. 

```bash
brew update
brew install git make pyenv gzip
pyenv init
```

Follow pyenv's instructions to add contents to your shell configuration files. Restart your shell and install Python 3.10 or later.
    
```bash
pyenv install 3.10.0
```

**Clone this GitHub repository**
```bash
git clone https://github.com/DavidBellamy/labrador
```

**Virtual environment**

The following setup commands and all files in `scripts/` should be run from the root directory of this repository. 

Ensure that you have Python3.10 or later selected in your active shell (`pyenv shell 3.10` if using `pyenv`), then create a virtual environment and activate it:

```bash
python -m venv venv && source venv/bin/activate
```

**Install requirements, download model weights and run tests**
```bash
make setup
```

`make setup` installs this repository as a Python package in editable mode along with all its requirements ([requirements.txt](requirements.txt)), downloads the model weights for Labrador and BERT from Hugging Face to the `model_weights/` directory and runs the project's tests found in `tests/`. 

🚨 Make sure that all tests pass before proceeding.

**The Makefile**

The Makefile contains a number of commands for running the files in `scripts/`. Using these, you can determine the order in which the scripts should be run as well as what command-line arguments each script expects.

# Labrador model architecture

Labrador's architecture is based on the [BERT](https://arxiv.org/abs/1810.04805) model with some modifications. It has a continuous embedding layer and a continuous prediction head, as shown in the figure below. 

We define a bag of labs as the collection of lab tests ordered at the same point in time for a patient. Labrador is pre-trained as a masked language model (MLM) on bags of labs with the objective of predicting the masked lab code and value given the other labs in the bag.

![Figure 1](img/fig_1.png#gh-light-mode-only)
![Figure 1](img/fig_1_dark.png#gh-dark-mode-only)

# Data

The data used to pre-train Labrador are a subset of the [MIMIC-IV](https://mimic-iv.mit.edu/) dataset. This dataset is freely available but requires a data use agreement to access so we cannot share it directly. We use the following tables:
* `labevents.csv`
* `admissions.csv`
* `patients.csv`
* `d_labitems.csv`

## Directly from PhysioNet

These files could be downloaded by singing the usage agreement on [MIMIC-IV page on PhysioNet](https://physionet.org/content/mimiciv/1.0/) and executing following command
```bash
make mimic_data physionet_username=*insert_your_username*
```
The download speed from the PhysioNet is usually quite low, so it may take a few hours to download.

## From Google Cloud

Alternatively, the data could be downloaded few hundred times faster from Google cloud, but this requires some additional setup.
1. Connect your PhysioNet to your Google account in [Cloud Settings](https://physionet.org/settings/cloud/)
2. [Request](https://physionet.org/projects/mimiciv/1.0/request_access/3) a Google Cloud access to the dataset
3. Setup Google Cloud Project and [billing](https://cloud.google.com/billing/docs/how-to/manage-billing-account) account
4. Install [`google-cloud-sdk`](https://cloud.google.com/sdk/docs/install)
5. Authenticate with `gcloud auth login` in the terminal
6. Run `make mimic_data google_project_id=*your_project_id* tool=gsutil`


## Downstream tasks data

The data used in the sepsis mortality prediction evaluation are also derived from the MIMIC-IV dataset. You can download sepsis cohort data via Google BigQuery:
```sql
SELECT `subject_id`, `stay_id`, `sofa_time`, `sofa_score` FROM `physionet-data.mimiciv_derived.sepsis3`;
```
The resulting table should be saved as `data/raw/mimic4_sepsis_cohort.csv`

The data we used for the cancer diagnosis, COVID-19 diagnosis and elevated alcohol consumption prediction evaluations are open source so we share it directly in `data/`.

## Making the pre-training data

This is done in 3 steps. First, raw EHR data are converted to JSON lines for each patient. Then, the JSON lines are converted to bags of labs. Finally, the bags of labs are converted to TFRecords for faster training.

Note: these steps require a lot of memory and disk space. 

* Raw -> JSON lines: output file pattern is `{model}_{split}_patients.jsonl`, e.g. bert_train_patients.jsonl
    * Scripts: pretraining_raw_data_to_{model}.py, where {model} in [bert, labrador].
* JSON lines -> Bags of labs: output file pattern is `{model}_{split}_bags.jsonl`, e.g. bert_train_bags.jsonl
    * Scripts: pretraining_jsonl_to_{model}_bags.py, where {model} in [bert, labrador].
* Bags of labs -> TFRecords: output directory pattern is `{model}_tfrecords_{split}/`, e.g. bert_tfrecords_train
    * Scripts: pretraining_bags_to_{model}_tfrecords.py, where {model} in [bert, labrador].


# Citing Labrador

If you use Labrador in your research, please cite:


# License

This work is licensed under the MIT License. You can find the full text of the license in the [LICENSE](LICENSE) file.
