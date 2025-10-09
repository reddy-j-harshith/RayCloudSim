#!/usr/bin/env python3
"""
Quick Test Run for Mid-Semester Evaluation
"""

from midsem_comprehensive_system import ComprehensiveMidsemEvaluation

def main():
    print("🎯 Quick Test Run - Mid-Semester Evaluation")
    print("=" * 60)
    
    # Create evaluation system
    evaluator = ComprehensiveMidsemEvaluation(base_output_dir="midsem_results")
    
    # Test with just one small dataset first
    print("🚀 Testing with pakistan/Tuple30K dataset...")
    evaluator.train_and_evaluate_single_dataset('pakistan', 'Tuple30K')
    
    # Generate a quick report
    report_path = evaluator.generate_comprehensive_html_report()
    print(f"\n✅ Quick test completed!")
    print(f"📄 Report: {report_path}")

if __name__ == "__main__":
    main()