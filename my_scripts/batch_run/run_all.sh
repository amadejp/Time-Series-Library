#!/usr/bin/env bash

echo "===== Starting All TimeXer Experiments ====="
SECONDS=0 # Bash variable to track script execution time

SCRIPT_DIR="./my_scripts/batch_run"

# Run Experiment 1
echo ""
echo ">>>>> RUNNING EXPERIMENT V1 <<<<<"
bash "${SCRIPT_DIR}/run_timexer_v1.sh"
if [ $? -ne 0 ]; then
    echo "ERROR: Experiment V1 failed!"
    # Decide if you want to stop or continue:
    # exit 1 # Stop
    echo "Continuing to next experiment despite V1 failure..." # Continue
fi
echo ">>>>> EXPERIMENT V1 COMPLETE <<<<<"
echo ""

# Run Experiment 2
echo ">>>>> RUNNING EXPERIMENT V2 <<<<<"
bash "${SCRIPT_DIR}/run_timexer_v2.sh"
if [ $? -ne 0 ]; then
    echo "ERROR: Experiment V2 failed!"
    # exit 1
    echo "Continuing to next experiment despite V2 failure..."
fi
echo ">>>>> EXPERIMENT V2 COMPLETE <<<<<"
echo ""

# Run Experiment 3
echo ">>>>> RUNNING EXPERIMENT V3 <<<<<"
bash "${SCRIPT_DIR}/run_timexer_v3.sh"
if [ $? -ne 0 ]; then
    echo "ERROR: Experiment V3 failed!"
    # exit 1
    echo "Continuing to next experiment despite V3 failure..."
fi
echo ">>>>> EXPERIMENT V3 COMPLETE <<<<<"
echo ""

# Run Experiment 4
echo ">>>>> RUNNING EXPERIMENT V4 <<<<<"
bash "${SCRIPT_DIR}/run_timexer_v4.sh"
if [ $? -ne 0 ]; then
    echo "ERROR: Experiment V4 failed!"
    # exit 1
    echo "Continuing to next experiment despite V4 failure..."
fi
echo ">>>>> EXPERIMENT V4 COMPLETE <<<<<"
echo ""


duration=$SECONDS
echo "===== All TimeXer Experiments Finished ====="
echo "Total execution time: $(($duration / 3600))h $(($duration / 60 % 60))m $(($duration % 60))s"