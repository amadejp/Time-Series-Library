#!/usr/bin/env bash

echo "===== Starting All TimeXer Experiments (Batch 2) ====="
SECONDS=0


SCRIPT_DIR="./my_scripts/batch_run"

# --- Batch 2 (v5-v8) ---
echo ">>>>> RUNNING EXPERIMENT V5 <<<<<"
bash "${SCRIPT_DIR}/run_timexer_v5.sh"
if [ $? -ne 0 ]; then echo "ERROR: Experiment V5 failed!"; echo "Continuing..."; else echo ">>>>> EXPERIMENT V5 COMPLETE <<<<<"; fi
echo ""

echo ">>>>> RUNNING EXPERIMENT V6 <<<<<"
bash "${SCRIPT_DIR}/run_timexer_v6.sh"
if [ $? -ne 0 ]; then echo "ERROR: Experiment V6 failed!"; echo "Continuing..."; else echo ">>>>> EXPERIMENT V6 COMPLETE <<<<<"; fi
echo ""

echo ">>>>> RUNNING EXPERIMENT V7 <<<<<"
bash "${SCRIPT_DIR}/run_timexer_v7.sh"
if [ $? -ne 0 ]; then echo "ERROR: Experiment V7 failed!"; echo "Continuing..."; else echo ">>>>> EXPERIMENT V7 COMPLETE <<<<<"; fi
echo ""

echo ">>>>> RUNNING EXPERIMENT V8 <<<<<"
bash "${SCRIPT_DIR}/run_timexer_v8.sh"
if [ $? -ne 0 ]; then echo "ERROR: Experiment V8 failed!"; else echo ">>>>> EXPERIMENT V8 COMPLETE <<<<<"; fi
echo ""


duration=$SECONDS
echo "===== All TimeXer Experiments Finished ====="
echo "Total execution time: $(($duration / 3600))h $(($duration / 60 % 60))m $(($duration % 60))s"