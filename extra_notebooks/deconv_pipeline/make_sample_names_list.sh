#!/bin/bash

cd ~/camunaslab/pipelines/hiv_deconv

# This assumes first column is gene names and all other columns are sample names
head -n 1 gene_names_x_samples.csv | tr ',' '\n' | tail -n +2 > sample_names.txt
