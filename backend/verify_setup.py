#!/usr/bin/env python3
"""
Verification Script for Shirt Size CV System

This script tests all components to ensure proper integration.
Run from backend directory: python verify_setup.py
"""

import sys
from pathlib import Path
import json

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def test_folder_structure():
    """Test if required folders exist"""
    print_header("Testing Folder Structure")
    
    required_folders = [
        "config",
        "database",
        "models",
        "training",
        "utils"
    ]
    
    optional_folders = [
        "data/synthetic_sizes",
        "trained_models",
        "evaluation_results"
    ]
    
    all_good = True
    
    for folder in required_folders:
        if Path(folder).exists():
            print_success(f"Required folder exists: {folder}/")
        else:
            print_error(f"Missing required folder: {folder}/")
            all_good = False
    
    for folder in optional_folders:
        if Path(folder).exists():
            print_success(f"Optional folder exists: {folder}/")
        else:
            print_warning(f"Optional folder missing: {folder}/ (will be created)")
    
    return all_good

def test_dependencies():
    """Test if all required packages are installed"""
    print_header("Testing Dependencies")
    
    required_packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'fastapi': 'FastAPI',
        'motor': 'Motor (MongoDB)',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'ultralytics': 'Ultralytics (YOLO)'
    }
    
    all_good = True
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print_success(f"{name} installed")
        except ImportError:
            print_error(f"{name} NOT installed")
            all_good = False
    
    return all_good

def test_dataset():
    """Test if dataset exists and is valid"""
    print_header("Testing Dataset")
    
    data_dir = Path("data/synthetic_sizes")
    
    if not data_dir.exists():
        print_warning("Dataset not generated yet")
        print("  Run: python -m training.generate_dataset")
        return False
    
    required_files = ['train.csv', 'val.csv', 'test.csv', 'metadata.json']
    all_good = True
    
    for file in required_files:
        filepath = data_dir / file
        if filepath.exists():
            print_success(f"Dataset file exists: {file}")
            
            # Additional checks
            if file.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(filepath)
                print(f"  → {len(df)} samples")
            elif file == 'metadata.json':
                with open(filepath) as f:
                    metadata = json.load(f)
                print(f"  → {metadata['num_sizes']} sizes, {metadata['num_fits']} fits")
        else:
            print_error(f"Missing dataset file: {file}")
            all_good = False
    
    return all_good

def test_trained_model():
    """Test if trained model exists"""
    print_header("Testing Trained Model")
    
    model_dir = Path("trained_models")
    
    if not model_dir.exists():
        print_warning("Model not trained yet")
        print("  Run: python -m training.train_model")
        return False
    
    required_files = ['best_model.pth', 'normalization_stats.json']
    all_good = True
    
    for file in required_files:
        filepath = model_dir / file
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print_success(f"Model file exists: {file} ({size_mb:.2f} MB)")
        else:
            print_error(f"Missing model file: {file}")
            all_good = False
    
    return all_good

def test_model_loading():
    """Test if model can be loaded"""
    print_header("Testing Model Loading")
    
    try:
        from models.size_predictor import SizePredictor
        
        # Test rule-based fallback
        predictor_rule = SizePredictor(model_path=None)
        print_success("SizePredictor initialized (rule-based)")
        
        # Test neural model
        model_path = Path("trained_models/best_model.pth")
        if model_path.exists():
            predictor_neural = SizePredictor(model_path=str(model_path))
            print_success("SizePredictor initialized (neural model)")
            
            # Test prediction
            test_measurements = {
                'shoulder_ratio': 0.22,
                'chest_ratio': 0.24,
                'waist_ratio': 0.20,
                'torso_proportion': 2.1
            }
            
            result = predictor_neural.predict(test_measurements)
            print_success(f"Test prediction: {result['estimated_size']} / {result['fit_type']} (conf: {result['confidence']:.3f})")
            
            return True
        else:
            print_warning("Trained model not found, using rule-based prediction")
            return False
    
    except Exception as e:
        print_error(f"Failed to load model: {str(e)}")
        return False

def test_evaluation_results():
    """Test if evaluation results exist"""
    print_header("Testing Evaluation Results")
    
    eval_dir = Path("evaluation_results")
    
    if not eval_dir.exists():
        print_warning("Evaluation not run yet")
        print("  Run: python -m training.evaluate_model")
        return False
    
    metrics_file = eval_dir / "metrics.json"
    
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        print_success("Evaluation metrics found:")
        print(f"  → Size accuracy: {metrics['size_accuracy']*100:.1f}%")
        print(f"  → Fit accuracy: {metrics['fit_accuracy']*100:.1f}%")
        print(f"  → Avg size confidence: {metrics['avg_size_confidence']:.3f}")
        print(f"  → Avg fit confidence: {metrics['avg_fit_confidence']:.3f}")
        
        return True
    else:
        print_error("Evaluation metrics not found")
        return False

def test_configuration():
    """Test if configuration is valid"""
    print_header("Testing Configuration")
    
    try:
        from config.settings import settings
        
        print_success("Settings loaded successfully")
        print(f"  → API Host: {settings.API_HOST}:{settings.API_PORT}")
        print(f"  → MongoDB: {settings.MONGODB_DB_NAME}")
        print(f"  → YOLO Model: {settings.YOLO_MODEL_PATH}")
        print(f"  → Size Model: {settings.SIZE_MODEL_PATH}")
        print(f"  → Use Trained Model: {settings.USE_TRAINED_MODEL}")
        
        return True
    
    except Exception as e:
        print_error(f"Failed to load configuration: {str(e)}")
        return False

def main():
    """Run all verification tests"""
    print(f"\n{BLUE}{'='*60}")
    print("Shirt Size CV System - Setup Verification")
    print(f"{'='*60}{RESET}")
    
    results = {
        "Folder Structure": test_folder_structure(),
        "Dependencies": test_dependencies(),
        "Dataset": test_dataset(),
        "Trained Model": test_trained_model(),
        "Model Loading": test_model_loading(),
        "Evaluation Results": test_evaluation_results(),
        "Configuration": test_configuration()
    }
    
    # Summary
    print_header("Verification Summary")
    
    for test_name, passed in results.items():
        if passed:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n{BLUE}{'='*60}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*60}{RESET}\n")
    
    if passed == total:
        print_success("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Start MongoDB: mongod --dbpath /path/to/db")
        print("  2. Start backend: python main.py")
        print("  3. Start frontend: cd ../frontend && npm start")
    else:
        print_warning(f"⚠️  {total - passed} test(s) failed. Please address issues above.")
        print("\nCommon fixes:")
        print("  • Install dependencies: pip install -r requirements.txt")
        print("  • Generate dataset: python -m training.generate_dataset")
        print("  • Train model: python -m training.train_model")
        print("  • Evaluate model: python -m training.evaluate_model")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
