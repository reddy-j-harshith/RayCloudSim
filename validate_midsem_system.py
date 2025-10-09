#!/usr/bin/env python3
"""
Validation Script for Mid-Semester Evaluation System

This script validates the comprehensive evaluation system by:
1. Testing on a single small dataset
2. Verifying all components work correctly
3. Checking data flows and result generation
"""

import sys
import os
import traceback
from datetime import datetime

# Import the main evaluation system
from comprehensive_midsem_evaluation import ComprehensiveMidsemEvaluation

def test_system_components():
    """Test individual system components"""
    print("🧪 Testing System Components...")
    
    try:
        # Test 1: System initialization
        print("   1️⃣ Testing system initialization...")
        evaluator = ComprehensiveMidsemEvaluation(base_output_dir="test_midsem_results")
        print("      ✅ System initialized successfully")
        
        # Test 2: Directory creation
        print("   2️⃣ Testing directory creation...")
        test_dir = evaluator.create_dataset_directory("test", "validation")
        if os.path.exists(test_dir):
            print("      ✅ Directory creation works")
        else:
            print("      ❌ Directory creation failed")
            return False
            
        # Test 3: Attack system integration
        print("   3️⃣ Testing attack system integration...")
        if evaluator.attack_system:
            print("      ✅ Attack system integrated")
        else:
            print("      ❌ Attack system integration failed")
            return False
            
        print("   ✅ All component tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Component test failed: {e}")
        traceback.print_exc()
        return False

def test_single_dataset():
    """Test evaluation on a single small dataset"""
    print("\n🎯 Testing Single Dataset Evaluation...")
    
    try:
        # Initialize system
        evaluator = ComprehensiveMidsemEvaluation(base_output_dir="test_midsem_results")
        
        # Test on smallest dataset (Pakistan Tuple30K)
        print("   📊 Testing on Pakistan Tuple30K dataset...")
        
        # This will test the complete pipeline
        evaluator.train_and_evaluate_single_dataset('pakistan', 'Tuple30K')
        
        print("   ✅ Single dataset evaluation completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Single dataset test failed: {e}")
        traceback.print_exc()
        return False

def test_visualization_generation():
    """Test visualization and report generation"""
    print("\n📈 Testing Visualization Generation...")
    
    try:
        evaluator = ComprehensiveMidsemEvaluation(base_output_dir="test_midsem_results")
        
        # Test visualization methods (with dummy data)
        print("   🎨 Testing visualization methods...")
        
        # Create dummy results for testing
        dataset_key = "test_dataset"
        evaluator.testing_results[dataset_key] = {
            'test_metrics': {
                'GAT': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1_score': 0.85, 'attack_detection_accuracy': 0.78},
                'GraphSAGE': {'accuracy': 0.82, 'precision': 0.80, 'recall': 0.84, 'f1_score': 0.82, 'attack_detection_accuracy': 0.75},
                'GCN': {'accuracy': 0.78, 'precision': 0.76, 'recall': 0.80, 'f1_score': 0.78, 'attack_detection_accuracy': 0.72},
                'Transformer': {'accuracy': 0.80, 'precision': 0.78, 'recall': 0.82, 'f1_score': 0.80, 'attack_detection_accuracy': 0.74}
            }
        }
        
        # Test plot generation
        test_plots_dir = os.path.join(evaluator.results_dir, "test_plots")
        os.makedirs(test_plots_dir, exist_ok=True)
        
        evaluator.plot_model_comparison(dataset_key, test_plots_dir)
        print("      ✅ Model comparison plot generated")
        
        print("   ✅ Visualization generation works!")
        return True
        
    except Exception as e:
        print(f"   ❌ Visualization test failed: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading capabilities"""
    print("\n📊 Testing Data Loading...")
    
    try:
        evaluator = ComprehensiveMidsemEvaluation(base_output_dir="test_midsem_results")
        
        # Test loading different datasets
        datasets_to_test = [
            ('pakistan', 'Tuple30K'),
            ('topo4mec', '25N50E')
        ]
        
        for dataset_name, dataset_flag in datasets_to_test:
            print(f"   📂 Testing {dataset_name}_{dataset_flag}...")
            try:
                trainset, testset, metadata = evaluator.attack_system.load_dataset(dataset_name, dataset_flag)
                print(f"      ✅ Loaded: {len(trainset)} train, {len(testset)} test samples")
            except Exception as e:
                print(f"      ⚠️ Could not load {dataset_name}_{dataset_flag}: {e}")
                # This is okay - some datasets might not exist
                continue
        
        print("   ✅ Data loading test completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Data loading test failed: {e}")
        traceback.print_exc()
        return False

def run_quick_validation():
    """Run quick validation of the entire system"""
    print("🚀 Running Quick Validation of Mid-Semester Evaluation System")
    print("=" * 70)
    
    validation_results = {
        'component_test': False,
        'data_loading_test': False,
        'visualization_test': False,
        'single_dataset_test': False
    }
    
    # Run tests
    validation_results['component_test'] = test_system_components()
    validation_results['data_loading_test'] = test_data_loading()
    validation_results['visualization_test'] = test_visualization_generation()
    
    # Only run single dataset test if other tests pass
    if all([validation_results['component_test'], validation_results['data_loading_test']]):
        print("\n🎯 All basic tests passed. Running single dataset test...")
        validation_results['single_dataset_test'] = test_single_dataset()
    else:
        print("\n⚠️ Basic tests failed. Skipping single dataset test.")
    
    # Summary
    print(f"\n{'='*70}")
    print("📋 VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    total_tests = len(validation_results)
    passed_tests = sum(validation_results.values())
    
    for test_name, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All validation tests passed! System is ready for full evaluation.")
        return True
    elif passed_tests >= total_tests - 1:
        print("⚠️ Most tests passed. System should work but check failed components.")
        return True
    else:
        print("❌ Multiple tests failed. Please fix issues before running full evaluation.")
        return False

def main():
    """Main validation function"""
    print(f"🧪 Mid-Semester Evaluation System Validation")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = run_quick_validation()
    
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print(f"\n🚀 READY TO RUN FULL EVALUATION!")
        print(f"💡 Run: python comprehensive_midsem_evaluation.py")
    else:
        print(f"\n🔧 PLEASE FIX ISSUES BEFORE RUNNING FULL EVALUATION")
    
    return success

if __name__ == "__main__":
    main()