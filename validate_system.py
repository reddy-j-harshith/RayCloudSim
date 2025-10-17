#!/usr/bin/env python3
"""
Validation Script for Mid-Semester GNN Trust System
Tests the system with a small dataset to ensure everything works before full evaluation
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path

# Add the RayCloudSim directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def validate_environment():
    """Validate the Python environment and dependencies"""
    print("🔍 Validating environment...")
    
    required_packages = [
        'torch', 'torch_geometric', 'numpy', 'pandas', 
        'sklearn', 'matplotlib', 'seaborn', 'networkx'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        return False
    
    print("✅ All required packages available")
    return True

def validate_datasets():
    """Validate that datasets are available"""
    print("\n🔍 Validating datasets...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    datasets = {
        'pakistan': ['Tuple30K', 'Tuple50K', 'Tuple100K'],
        'topo4mec': ['25N50E', '50N50E', '100N150E']
    }
    
    available_datasets = []
    
    for dataset_name, dataset_flags in datasets.items():
        for dataset_flag in dataset_flags:
            if dataset_name == 'pakistan':
                data_dir = os.path.join(base_dir, "eval", "benchmarks", "Pakistan", "data", dataset_flag)
            else:
                data_dir = os.path.join(base_dir, "eval", "benchmarks", "Topo4MEC", "data", dataset_flag)
            
            trainset_path = os.path.join(data_dir, "trainset.csv")
            testset_path = os.path.join(data_dir, "testset.csv")
            
            if os.path.exists(trainset_path) and os.path.exists(testset_path):
                print(f"  ✅ {dataset_name}/{dataset_flag}")
                available_datasets.append((dataset_name, dataset_flag))
                
                # Check dataset size
                try:
                    trainset = pd.read_csv(trainset_path)
                    testset = pd.read_csv(testset_path)
                    print(f"    📊 Train: {len(trainset)} samples, Test: {len(testset)} samples")
                except Exception as e:
                    print(f"    ⚠️  Error reading dataset: {str(e)}")
            else:
                print(f"  ❌ {dataset_name}/{dataset_flag}")
    
    print(f"\n✅ Found {len(available_datasets)} available datasets")
    return available_datasets

def test_attack_system():
    """Test the attack-aware system with a small example"""
    print("\n🔍 Testing attack-aware system...")
    
    try:
        from research_attack_aware_system import ResearchAttackAwareSystem
        
        # Initialize system
        system = ResearchAttackAwareSystem()
        print("  ✅ System initialized successfully")
        
        # Test dataset loading
        try:
            trainset, testset, metadata = system.load_dataset('pakistan', 'Tuple30K')
            print(f"  ✅ Dataset loading works - Train: {len(trainset)}, Test: {len(testset)}")
            print(f"    Features: {list(trainset.columns)}")
            return True
        except Exception as e:
            print(f"  ❌ Dataset loading failed: {str(e)}")
            return False
            
    except Exception as e:
        print(f"  ❌ System initialization failed: {str(e)}")
        return False

def run_quick_validation():
    """Run a quick validation with minimal training"""
    print("\n🔍 Running quick validation test...")
    
    try:
        from research_attack_aware_system import ResearchAttackAwareSystem
        
        # Initialize system
        system = ResearchAttackAwareSystem()
        
        # Create a small test output directory
        test_dir = "validation_test"
        os.makedirs(test_dir, exist_ok=True)
        
        # Run very short simulation
        print("  🧪 Running minimal simulation...")
        results = system.run_comprehensive_attack_simulation(
            dataset_name='pakistan',
            dataset_flag='Tuple30K',
            output_dir=test_dir,
            model_type='gat',
            malicious_ratio=0.30,
            num_epochs=2,  # Very short for validation
            task_cycles=5,  # Very short for validation
            save_models=False  # Don't save models for validation
        )
        
        print("  ✅ Simulation completed successfully")
        print(f"    Results keys: {list(results.keys())}")
        
        # Clean up
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Quick validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def validate_pytorch():
    """Validate PyTorch and CUDA availability"""
    print("\n🔍 Validating PyTorch setup...")
    
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("  ⚠️  CUDA not available - will use CPU")
    
    # Test tensor operations
    try:
        x = torch.randn(10, 10)
        y = torch.matmul(x, x.T)
        print("  ✅ Basic tensor operations work")
        return True
    except Exception as e:
        print(f"  ❌ Tensor operations failed: {str(e)}")
        return False

def main():
    """Main validation function"""
    print("🎯 Mid-Semester System Validation")
    print("=" * 50)
    
    validation_results = {}
    
    # Run all validation tests
    validation_results['environment'] = validate_environment()
    validation_results['pytorch'] = validate_pytorch()
    validation_results['datasets'] = len(validate_datasets()) > 0
    validation_results['system'] = test_attack_system()
    validation_results['quick_test'] = run_quick_validation()
    
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():<15}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ System is ready for comprehensive evaluation")
        print("\nTo run the full evaluation:")
        print("python midsem_comprehensive_system.py")
    else:
        print("⚠️  SOME VALIDATIONS FAILED!")
        print("❌ Please fix the issues before running full evaluation")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)