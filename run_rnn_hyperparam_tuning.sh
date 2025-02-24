#!/usr/bin/env bash
cd $(dirname "$0")
export PYTHONPATH=src
python src/run_rnn_hyperparam_tuning.py $1 $2

