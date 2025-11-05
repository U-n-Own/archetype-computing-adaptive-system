#!/bin/bash

# Hyperparameter Search for Multiple UCR Datasets
# Runs experiments for: FordA, FordB, Adiac, OliveOil, CinCECGTorso
# Tests: ESN, Antisymmetric ESN, RON, Antisymmetric RON

#!/bin/bash

# Hyperparameter Search for Multiple UCR Datasets
# Runs experiments for: FordA, FordB, Adiac, OliveOil, CinCECGTorso
# Tests: ESN, Antisymmetric ESN, RON, Antisymmetric RON
# Large datasets: 300 units, Small datasets: K-fold CV with 150 units

# Configuration
N_TRIALS=30
BATCH_SIZE=32
DEVICE="cuda"
DATAROOT="./data/UCR"
RESULTROOT="./ucr_results"
N_FOLDS=5  # For small datasets

# Large datasets (use standard hyperparameter search with 500 units)
LARGE_DATASETS=("FordA" "FordB" "Adiac")

# Small datasets (use k-fold cross-validation with 150 units)
SMALL_DATASETS=("OliveOil" "CinCECGTorso")

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Arrays to track results
FAILED_EXPERIMENTS=()
SUCCESSFUL_EXPERIMENTS=()

echo "======================================================================"
echo "UCR Hyperparameter Search - Multiple Datasets"
echo "======================================================================"
echo "Large Datasets (300 units): ${LARGE_DATASETS[@]}"
echo "Small Datasets (150 units, K-Fold): ${SMALL_DATASETS[@]}"
echo "Models: ESN, ESN-Antisym, RON, RON-Antisym"
echo "Trials per experiment: ${N_TRIALS}"
echo "Device: ${DEVICE}"
echo "======================================================================"
echo ""

# Counter for completed experiments
TOTAL_EXPERIMENTS=$(((${#LARGE_DATASETS[@]} + ${#SMALL_DATASETS[@]}) * 4))
COMPLETED=0

# Loop through large datasets
for dataset in "${LARGE_DATASETS[@]}"; do
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Starting experiments for: ${dataset}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    # ESN
    echo -e "${GREEN}[${COMPLETED}/${TOTAL_EXPERIMENTS}] Running: ${dataset} - ESN${NC}"
    if python ucr_hyperparameter_search.py \
        --dataset ${dataset} \
        --model esn \
        --n_trials ${N_TRIALS} \
        --batch_size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --dataroot ${DATAROOT} \
        --resultroot ${RESULTROOT}; then
        SUCCESSFUL_EXPERIMENTS+=("${dataset} - ESN")
        echo -e "${GREEN}✓ Success${NC}"
    else
        FAILED_EXPERIMENTS+=("${dataset} - ESN")
        echo -e "${RED}✗ Failed${NC}"
    fi
    ((COMPLETED++))
    echo ""
    
    # ESN Antisymmetric
    echo -e "${GREEN}[${COMPLETED}/${TOTAL_EXPERIMENTS}] Running: ${dataset} - ESN Antisymmetric${NC}"
    if python ucr_hyperparameter_search.py \
        --dataset ${dataset} \
        --model esn \
        --antisymmetric \
        --n_trials ${N_TRIALS} \
        --batch_size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --dataroot ${DATAROOT} \
        --resultroot ${RESULTROOT}; then
        SUCCESSFUL_EXPERIMENTS+=("${dataset} - ESN Antisymmetric")
        echo -e "${GREEN}✓ Success${NC}"
    else
        FAILED_EXPERIMENTS+=("${dataset} - ESN Antisymmetric")
        echo -e "${RED}✗ Failed${NC}"
    fi
    ((COMPLETED++))
    echo ""
    
    # RON
    echo -e "${GREEN}[${COMPLETED}/${TOTAL_EXPERIMENTS}] Running: ${dataset} - RON${NC}"
    if python ucr_hyperparameter_search.py \
        --dataset ${dataset} \
        --model ron \
        --n_trials ${N_TRIALS} \
        --batch_size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --dataroot ${DATAROOT} \
        --resultroot ${RESULTROOT}; then
        SUCCESSFUL_EXPERIMENTS+=("${dataset} - RON")
        echo -e "${GREEN}✓ Success${NC}"
    else
        FAILED_EXPERIMENTS+=("${dataset} - RON")
        echo -e "${RED}✗ Failed${NC}"
    fi
    ((COMPLETED++))
    echo ""
    
    # RON Antisymmetric
    echo -e "${GREEN}[${COMPLETED}/${TOTAL_EXPERIMENTS}] Running: ${dataset} - RON Antisymmetric${NC}"
    if python ucr_hyperparameter_search.py \
        --dataset ${dataset} \
        --model ron \
        --antisymmetric \
        --n_trials ${N_TRIALS} \
        --batch_size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --dataroot ${DATAROOT} \
        --resultroot ${RESULTROOT}; then
        SUCCESSFUL_EXPERIMENTS+=("${dataset} - RON Antisymmetric")
        echo -e "${GREEN}✓ Success${NC}"
    else
        FAILED_EXPERIMENTS+=("${dataset} - RON Antisymmetric")
        echo -e "${RED}✗ Failed${NC}"
    fi
    ((COMPLETED++))
    echo ""
    
    echo -e "${YELLOW}Completed all experiments for ${dataset}${NC}"
    echo ""
done

# Loop through small datasets (using K-Fold CV)
for dataset in "${SMALL_DATASETS[@]}"; do
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Starting K-FOLD experiments for: ${dataset}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    # ESN with K-Fold
    echo -e "${GREEN}[${COMPLETED}/${TOTAL_EXPERIMENTS}] Running K-Fold: ${dataset} - ESN${NC}"
    if python ucr_hyperparameter_search_kfold.py \
        --dataset ${dataset} \
        --model esn \
        --n_folds ${N_FOLDS} \
        --n_trials ${N_TRIALS} \
        --batch_size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --dataroot ${DATAROOT} \
        --resultroot ${RESULTROOT}; then
        SUCCESSFUL_EXPERIMENTS+=("${dataset} - ESN (K-Fold)")
        echo -e "${GREEN}✓ Success${NC}"
    else
        FAILED_EXPERIMENTS+=("${dataset} - ESN (K-Fold)")
        echo -e "${RED}✗ Failed${NC}"
    fi
    ((COMPLETED++))
    echo ""
    
    # ESN Antisymmetric with K-Fold
    echo -e "${GREEN}[${COMPLETED}/${TOTAL_EXPERIMENTS}] Running K-Fold: ${dataset} - ESN Antisymmetric${NC}"
    if python ucr_hyperparameter_search_kfold.py \
        --dataset ${dataset} \
        --model esn \
        --antisymmetric \
        --n_folds ${N_FOLDS} \
        --n_trials ${N_TRIALS} \
        --batch_size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --dataroot ${DATAROOT} \
        --resultroot ${RESULTROOT}; then
        SUCCESSFUL_EXPERIMENTS+=("${dataset} - ESN Antisymmetric (K-Fold)")
        echo -e "${GREEN}✓ Success${NC}"
    else
        FAILED_EXPERIMENTS+=("${dataset} - ESN Antisymmetric (K-Fold)")
        echo -e "${RED}✗ Failed${NC}"
    fi
    ((COMPLETED++))
    echo ""
    
    # RON with K-Fold
    echo -e "${GREEN}[${COMPLETED}/${TOTAL_EXPERIMENTS}] Running K-Fold: ${dataset} - RON${NC}"
    if python ucr_hyperparameter_search_kfold.py \
        --dataset ${dataset} \
        --model ron \
        --n_folds ${N_FOLDS} \
        --n_trials ${N_TRIALS} \
        --batch_size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --dataroot ${DATAROOT} \
        --resultroot ${RESULTROOT}; then
        SUCCESSFUL_EXPERIMENTS+=("${dataset} - RON (K-Fold)")
        echo -e "${GREEN}✓ Success${NC}"
    else
        FAILED_EXPERIMENTS+=("${dataset} - RON (K-Fold)")
        echo -e "${RED}✗ Failed${NC}"
    fi
    ((COMPLETED++))
    echo ""
    
    # RON Antisymmetric with K-Fold
    echo -e "${GREEN}[${COMPLETED}/${TOTAL_EXPERIMENTS}] Running K-Fold: ${dataset} - RON Antisymmetric${NC}"
    if python ucr_hyperparameter_search_kfold.py \
        --dataset ${dataset} \
        --model ron \
        --antisymmetric \
        --n_folds ${N_FOLDS} \
        --n_trials ${N_TRIALS} \
        --batch_size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --dataroot ${DATAROOT} \
        --resultroot ${RESULTROOT}; then
        SUCCESSFUL_EXPERIMENTS+=("${dataset} - RON Antisymmetric (K-Fold)")
        echo -e "${GREEN}✓ Success${NC}"
    else
        FAILED_EXPERIMENTS+=("${dataset} - RON Antisymmetric (K-Fold)")
        echo -e "${RED}✗ Failed${NC}"
    fi
    ((COMPLETED++))
    echo ""
    
    echo -e "${YELLOW}Completed all K-Fold experiments for ${dataset}${NC}"
    echo ""
done

echo "======================================================================"
echo -e "${GREEN}ALL EXPERIMENTS COMPLETED!${NC}"
echo "======================================================================"
echo "Total experiments attempted: ${COMPLETED}"
echo "Successful: ${#SUCCESSFUL_EXPERIMENTS[@]}"
echo "Failed: ${#FAILED_EXPERIMENTS[@]}"
echo ""

if [ ${#SUCCESSFUL_EXPERIMENTS[@]} -gt 0 ]; then
    echo -e "${GREEN}Successful experiments:${NC}"
    for exp in "${SUCCESSFUL_EXPERIMENTS[@]}"; do
        echo -e "  ${GREEN}✓${NC} $exp"
    done
    echo ""
fi

if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
    echo -e "${RED}Failed experiments:${NC}"
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo -e "  ${RED}✗${NC} $exp"
    done
    echo ""
fi

echo "Results saved in: ${RESULTROOT}"
echo "======================================================================"
