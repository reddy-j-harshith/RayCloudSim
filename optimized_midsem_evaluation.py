#!/usr/bin/env python3
"""
Optimized Mid-Semester GNN Trust System Evaluation
Focuses on comprehensive coverage with optimized parameters for reasonable runtime
"""

from midsem_comprehensive_system import ComprehensiveMidsemEvaluation

def main():
    print("🎯 Optimized Mid-Semester Comprehensive GNN Trust System Evaluation")
    print("=" * 80)
    
    # Create evaluation system with optimized parameters
    evaluator = ComprehensiveMidsemEvaluation(base_output_dir="midsem_results")
    
    # Override some settings for faster execution while maintaining quality
    evaluator.gnn_models = ['GAT', 'GraphSAGE']  # Focus on 2 best models
    
    # Reduce dataset scope for demonstration while maintaining diversity
    evaluator.datasets = {
        'pakistan': ['Tuple30K', 'Tuple50K'],  # 2 sizes from Pakistan
        'topo4mec': ['25N50E', '50N50E']       # 2 sizes from Topo4MEC
    }
    
    print(f"📊 Optimized evaluation plan:")
    print(f"   - Datasets: {sum(len(flags) for flags in evaluator.datasets.values())} datasets")
    print(f"   - Models: {len(evaluator.gnn_models)} GNN models")
    print(f"   - Total experiments: {sum(len(flags) for flags in evaluator.datasets.values()) * len(evaluator.gnn_models) * 3}")
    print(f"   - Estimated runtime: 2-3 hours")
    
    # Run complete evaluation
    results_dir = evaluator.run_complete_evaluation()
    
    print(f"\n🎉 Optimized evaluation completed successfully!")
    print(f"📁 Check results in: {results_dir}")

if __name__ == "__main__":
    main()