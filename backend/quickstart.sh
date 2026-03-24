#!/bin/bash

echo "=========================================="
echo "Shirt Size CV System - Quick Start"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the backend directory
if [ ! -f "main.py" ]; then
    echo -e "${RED}Error: Please run this script from the backend directory${NC}"
    echo "Usage: cd backend && bash quickstart.sh"
    exit 1
fi

echo -e "${YELLOW}Step 1: Checking dependencies...${NC}"
python -c "import torch; import fastapi; import cv2; import mediapipe" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
fi

echo ""
echo -e "${YELLOW}Step 2: Generating synthetic dataset...${NC}"
if [ -f "data/synthetic_sizes/metadata.json" ]; then
    echo -e "${GREEN}✓ Dataset already exists (skipping)${NC}"
else
    python -m training.generate_dataset
fi

echo ""
echo -e "${YELLOW}Step 3: Training the model...${NC}"
if [ -f "trained_models/best_model.pth" ]; then
    echo -e "${GREEN}✓ Trained model already exists (skipping)${NC}"
else
    python -m training.train_model
fi

echo ""
echo -e "${YELLOW}Step 4: Evaluating the model...${NC}"
python -m training.evaluate_model

echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete! 🎉"
echo "==========================================${NC}"
echo ""
echo "Model Performance:"
if [ -f "evaluation_results/metrics.json" ]; then
    python -c "import json; m=json.load(open('evaluation_results/metrics.json')); print(f\"  Size Accuracy: {m['size_accuracy']*100:.1f}%\"); print(f\"  Fit Accuracy: {m['fit_accuracy']*100:.1f}%\")"
fi

echo ""
echo "Next steps:"
echo "  1. Start MongoDB: mongod --dbpath /path/to/db"
echo "  2. Start backend: python main.py"
echo "  3. Start frontend: cd ../frontend && npm start"
echo ""
echo "For detailed instructions, see INTEGRATION_GUIDE.md"
