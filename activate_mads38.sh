#!/usr/bin/env bash

source /hard_data/user/xiefeiyang/miniforge3/etc/profile.d/conda.sh
conda activate mads38

export MPLCONFIGDIR=/tmp/mads_mpl
mkdir -p "$MPLCONFIGDIR"

echo "Activated conda env: mads38"
echo "Python: $(which python)"
