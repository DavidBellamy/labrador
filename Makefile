#################################################################################
# Setup				                                                            #
#################################################################################
setup: install download_model_weights unittests

install: requirements.txt setup.py
	pip install -e .

download_model_weights:
	mkdir -p model_weights
	python download_weights.py

unittests:
	python -m unittest discover -s tests

# Username from Physionet, which grants permission to access the MIMIC-IV database
physionet_username ?=

mimic_files = \
	hosp/labevents.csv.gz \
	hosp/d_labitems.csv.gz \
	core/admissions.csv.gz \
	core/patients.csv.gz

# Tool to use for downloading (default is wget, can be overridden by setting the tool variable)
tool ?= wget

# Optional: download the raw MIMIC-IV data using the specified tool (warning: large files)
mimic_data:
	mkdir -p $(raw_data_dir)
ifeq ($(tool), wget)
	@if [ -z "$(physionet_username)" ]; then \
		echo "Error: physionet_username is not set. Please provide a username to access the MIMIC-IV dataset"; \
		exit 1; \
	fi
	for file in $(mimic_files); do \
		wget -r -N -c -np --user $(physionet_username) --ask-password --directory-prefix=$(raw_data_dir) https://physionet.org/files/mimiciv/1.0/$$file; \
		gunzip -v $(raw_data_dir)/$$(basename $$file); \
	done
else ifeq ($(tool), gsutil)
	@if [ -z "$(google_project_id)" ]; then \
		echo "Error: google_project_id is not set. Please provide a project id to access the MIMIC-IV dataset"; \
		exit 1; \
	fi
	for file in $(mimic_files); do \
		gsutil -u $(google_project_id) cp gs://mimiciv-1.0.physionet.org/$$file $(raw_data_dir); \
		gunzip -v $(raw_data_dir)/$$(basename $$file); \
	done
else
	@echo "Error: Unsupported tool. Use 'wget' or 'gsutil'."
	@exit 1
endif

#################################################################################
# Project Settings                                                              #
#################################################################################

data_dir := data/ # toggle this to data_full/ when you want to run on the full dataset
raw_data_dir := $(addsuffix raw, $(data_dir))
processed_data_dir := $(addsuffix processed, $(data_dir))
evaluation_data_dir := $(addsuffix evaluations, $(data_dir))
results_dir := $(addsuffix results, $(data_dir))
model_weights_dir := model_weights/

random_seed = 3141592

#################################################################################
# Pre-training & Evaluation Parameters	                                        #
#################################################################################

# Minimum frequency of a lab test in MIMIC-IV to be included in pre-training
min_freq = 500

# Train/validation/test split proportions
train_pct = 0.7
val_pct = 0.1
test_pct = 0.2

# Number of quantiles to bin lab values into for BERT
num_bins = 10

# The time-window for each bag of labs (in hours)
max_time_delta = 0

# The minimum required number of labs per bag for inclusion in the pre-training dataset
min_bag_length = 3

# The maximum proportion of <NULL> strings (missing lab values) allowed in each bag of labs
null_threshold = 0.5

# d_model for Labrador and BERT
embed_dim = 1024

# Nested K-fold cross-validation for fine-tuning evaluations
k_inner = 3
k_outer = 5

# Make tensorflow quiet
export TF_CPP_MIN_LOG_LEVEL=3

#################################################################################
# Pre-processing the Pre-training Data                                          #
#################################################################################

pretraining_data: bert_jsonl bert_bags bert_tfrecords labrador_jsonl labrador_bags labrador_tfrecords

bert_jsonl:
	python scripts/preprocessing/pretraining_raw_data_to_labrador_jsonl.py \
		$(raw_data_dir) \
		$(processed_data_dir) \
		$(random_seed) \
		$(train_pct) \
		$(val_pct) \
		$(test_pct) \
		$(min_freq) \
		$(num_bins)

labrador_jsonl: 
	python scripts/preprocessing/pretraining_raw_data_to_labrador_jsonl.py \
		$(raw_data_dir) \
		$(processed_data_dir) \
		$(random_seed) \
		$(train_pct) \
		$(val_pct) \
		$(test_pct) \
		$(min_freq)

bert_bags:
	python scripts/preprocessing/pretraining_jsonl_to_bert_bags.py \
		$(processed_data_dir) \
		$(max_time_delta) \
		$(min_bag_length)

labrador_bags:
	python scripts/preprocessing/pretraining_jsonl_to_labrador_bags.py \
		$(processed_data_dir) \
		$(max_time_delta) \
		$(min_bag_length) \
		$(null_threshold)

bert_tfrecords: bert_train_tfrecords bert_val_tfrecords

bert_train_tfrecords: bert_special_tokens
	python scripts/preprocessing/pretraining_bags_to_bert_tfrecords.py \
		$(processed_data_dir) \
		$(random_seed) \
		$(MASK_TOKEN) \
		bert_train_bags.jsonl \
		$(processed_data_dir)/bert_tfrecords_train

bert_val_tfrecords: bert_special_tokens
	python scripts/preprocessing/pretraining_bags_to_bert_tfrecords.py \
		$(processed_data_dir) \
		$(random_seed) \
		$(MASK_TOKEN) \
		bert_validation_bags.jsonl \
		$(processed_data_dir)/bert_tfrecords_val

labrador_tfrecords: labrador_train_tfrecords labrador_val_tfrecords

labrador_train_tfrecords: labrador_special_tokens
	python scripts/preprocessing/pretraining_bags_to_labrador_tfrecords.py \
		$(processed_data_dir) \
		$(random_seed) \
		$(MASK_TOKEN) \
		$(NULL_TOKEN) \
		$(processed_data_dir)/labrador_train_bags.jsonl \
		$(processed_data_dir)/labrador_tfrecords_train

labrador_val_tfrecords: labrador_special_tokens
	python scripts/preprocessing/pretraining_bags_to_labrador_tfrecords.py \
		$(processed_data_dir) \
		$(random_seed) \
		$(MASK_TOKEN) \
		$(NULL_TOKEN) \
		$(processed_data_dir)/labrador_validation_bags.jsonl \
		$(processed_data_dir)/labrador_tfrecords_val


#################################################################################
# Pre-processing the Evaluation Data                                            #
#################################################################################

evaluation_data: cancer_baseline_data cancer_transformer_data covid_baseline_data covid_transformer_data alcohol_transformer_data sepsis_jsonlines sepsis_baseline_data sepsis_transformer_data

cancer_baseline_data:
	python scripts/preprocessing/cancer_diagnosis_baselines_data.py

cancer_transformer_data: labrador_special_tokens
	python scripts/preprocessing/cancer_diagnosis_transformer_data.py \
		$(evaluation_data_dir) \
		$(processed_data_dir)/mimic4_ecdfs.npz \
		Disease \
		$(processed_data_dir)/labcode_codebook_labrador.csv \
		$(processed_data_dir)/labcode_codebook_bert.csv \
		$(NULL_TOKEN)

covid_baseline_data:
	python scripts/preprocessing/covid_diagnosis_baselines_data.py \
		$(raw_data_dir) \
		$(evaluation_data_dir)

covid_transformer_data: labrador_special_tokens
	python scripts/preprocessing/covid_diagnosis_transformer_data.py \
		$(raw_data_dir) \
		$(evaluation_data_dir) \
		$(processed_data_dir)/mimic4_ecdfs.npz \
		target \
		$(processed_data_dir)/labcode_codebook_labrador.csv \
		$(processed_data_dir)/labcode_codebook_bert.csv \
		$(NULL_TOKEN)

alcohol_transformer_data:
	python scripts/preprocessing/drinks_per_day_transformer_data.py

sepsis_jsonlines:
	python scripts/preprocessing/sepsis_make_jsonl.py \
		$(raw_data_dir) \
		$(processed_data_dir)

sepsis_baseline_data:
	python scripts/preprocessing/sepsis_mortality_baselines_data.py \
		$(processed_data_dir) \
		$(evaluation_data_dir)

sepsis_transformer_data: labrador_special_tokens
	python scripts/preprocessing/sepsis_mortality_transformer_data.py \
		$(processed_data_dir) \
		$(evaluation_data_dir) \
		$(processed_data_dir)/mimic4_ecdfs.npz \
		$(processed_data_dir)/labcode_codebook_labrador.csv \
		$(processed_data_dir)/labcode_codebook_bert.csv \
		$(NULL_TOKEN)

#################################################################################
# Pre-training Labrador & BERT				                                    #
#################################################################################

train_labrador: labrador_special_tokens
	python scripts/pretraining/train_labrador \
	$(random_seed) \
	$(MASK_TOKEN) \
	$(NULL_TOKEN) \
	$(PAD_TOKEN) \
	$(VOCAB_SIZE) \
	$(EMBED_DIM)

train_bert: bert_special_tokens
	python scripts/pretraining/train_bert \
	$(processed_data_dir) \
	$(random_seed) \
	$(PAD_TOKEN) \
	$(VOCAB_SIZE) \
	$(EMBED_DIM)

#################################################################################
# Evaluations           														#
#################################################################################

cancer_logreg_evaluation:
	python scripts/evaluations/cancer_diagnosis_logistic_regression_baseline.py \
		$(results_dir) \
		$(evaluation_data_dir) \
		$(k_inner) \
		$(k_outer) \
		$(random_seed) \
		1

cancer_randomforest_evaluation:
	python scripts/evaluations/cancer_diagnosis_random_forest_baseline.py \
		$(results_dir) \
		$(evaluation_data_dir) \
		$(k_inner) \
		$(k_outer) \
		$(random_seed) \
		1

cancer_xgboost_evaluation:
	python scripts/evaluations/cancer_diagnosis_xgboost_baseline.py \
		$(results_dir) \
		$(evaluation_data_dir) \
		$(k_inner) \
		$(k_outer) \
		$(random_seed) \
		1

cancer_labrador_evaluation:
	python scripts/evaluations/cancer_diagnosis_labrador.py \
		configs/cancer_diagnosis/cancer_diagnosis_labrador.json \
		1

cancer_bert_evaluation:
	python scripts/evaluations/cancer_diagnosis_bert.py \
		configs/cancer_diagnosis/cancer_diagnosis_bert.json \
		1

covid_logreg_evaluation:
	python scripts/evaluations/covid_diagnosis_logistic_regression_baseline.py \
		$(results_dir) \
		$(evaluation_data_dir) \
		$(k_inner) \
		$(k_outer) \
		$(random_seed) \
		1

covid_randomforest_evaluation:
	python scripts/evaluations/covid_diagnosis_random_forest_baseline.py \
		$(results_dir) \
		$(evaluation_data_dir) \
		$(k_inner) \
		$(k_outer) \
		$(random_seed) \
		1

covid_xgboost_evaluation:
	python scripts/evaluations/covid_diagnosis_xgboost_baseline.py \
		$(results_dir) \
		$(evaluation_data_dir) \
		$(k_inner) \
		$(k_outer) \
		$(random_seed) \
		1

covid_labrador_evaluation:
	python scripts/evaluations/covid_diagnosis_labrador.py \
		configs/covid_diagnosis/covid_diagnosis_labrador.json \
		1

covid_bert_evaluation:
	python scripts/evaluations/covid_diagnosis_bert.py \
		configs/covid_diagnosis/covid_diagnosis_bert.json \
		1

alcohol_baseline_evaluation:
	python scripts/evaluations/drinks_per_day_baselines.py \
		$(results_dir) \
		$(evaluation_data_dir) \
		$(k_inner) \
		$(k_outer) \
		$(random_seed) \
		1

alcohol_labrador_evaluation:
	python scripts/evaluations/drinks_per_day_labrador.py \
		configs/drinks_per_day/drinks_per_day_labrador.json \
		1

alcohol_bert_evaluation:
	python scripts/evaluations/drinks_per_day_bert.py \
		configs/drinks_per_day/drinks_per_day_bert.json \
		1

sepsis_logreg_evaluation:
	python scripts/evaluations/sepsis_mortality_logistic_regression_baseline.py \
		$(results_dir) \
		$(evaluation_data_dir) \
		$(k_inner) \
		$(k_outer) \
		$(random_seed) \
		1

sepsis_randomforest_evaluation:
	python scripts/evaluations/sepsis_mortality_random_forest_baseline.py \
		$(results_dir) \
		$(evaluation_data_dir) \
		$(k_inner) \
		$(k_outer) \
		$(random_seed) \
		1

sepsis_xgboost_evaluation:
	python scripts/evaluations/sepsis_mortality_xgboost_baseline.py \
		$(results_dir) \
		$(evaluation_data_dir) \
		$(k_inner) \
		$(k_outer) \
		$(random_seed) \
		1

sepsis_labrador_evaluation:
	python scripts/evaluations/sepsis_mortality_labrador.py \
		configs/sepsis_mortality_prediction/sepsis_mortality_labrador.json \
		1

sepsis_bert_evaluation:
	python scripts/evaluations/sepsis_mortality_bert.py \
		configs/sepsis_mortality_prediction/sepsis_mortality_bert.json \
		1

imputation_labrador_evaluation:
	python scripts/evaluations/intrinsic_imputation_labrador.py \
		configs/intrinsic_imputation/intrinsic_imputation_labrador.json

imputation_bert_evaluation:
	python scripts/evaluations/intrinsic_imputation_bert.py \
		configs/intrinsic_imputation/intrinsic_imputation_BERT68M.json

umap_labrador:
	python scripts/evaluations/labrador_umap.py \
		configs/umap/labrador_umap.json

umap_bert:
	python scripts/evaluations/bert_umap.py \
		configs/umap/bert_umap.json

#################################################################################
# Helper Functions to Get Special Tokens  									    #
#################################################################################

labrador_special_tokens:
# Path to the codebook (this is created by the recipe: make json_lines)
	$(eval codebook = $(addprefix $(processed_data_dir), /labcode_codebook_labrador.csv))
# Compute vocab_size based on the codebook's length
	$(eval VOCAB_SIZE = $(shell echo $(shell wc -l < $(codebook)) - 1 | bc))

# Define special tokens accordingly
	$(eval MASK_TOKEN = $(shell echo $(VOCAB_SIZE) + 1 | bc))
	$(eval PAD_TOKEN = 0)
	$(eval NULL_TOKEN = $(shell echo $(VOCAB_SIZE) + 2 | bc))

bert_special_tokens:
# Path to the codebook (this is created by the recipe: make json_lines)
	$(eval codebook = $(addprefix $(processed_data_dir), /labcode_codebook_bert.csv))
# Compute vocab_size based on the codebook's length
	$(eval VOCAB_SIZE = $(shell echo $(shell wc -l < $(codebook)) - 1 | bc))

# Define special tokens accordingly
	$(eval MASK_TOKEN = $(shell echo $(VOCAB_SIZE) + 1 | bc))
	$(eval PAD_TOKEN = 0)
