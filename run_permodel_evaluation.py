#!/usr/bin/env python3
"""
Run Per-Model GNN Trust Evaluation
===================================
This script executes the fixed enhanced midsem system with PER-MODEL logging.
Each GNN model (GAT, GraphSAGE, GCN, Transformer) will have separate:
- Performance metrics (success rate, latency, energy)
- Trust trajectories
- Classification metrics (precision/recall/F1)
- Protection analysis
- Confusion matrices
- Visualizations
"""

import sys
import os
from fixed_enhanced_midsem_system import FixedEnhancedMidsemSystem

def main():
    print("=" * 80)
    print("🚀 PER-MODEL GNN TRUST EVALUATION SYSTEM")
    print("=" * 80)
    print("\n📊 This run will execute experiments for EACH model separately:")
    print("   1. GAT (Graph Attention Network)")
    print("   2. GraphSAGE")
    print("   3. GCN (Graph Convolutional Network)")
    print("   4. Transformer")
    print("\n🎯 Each model will have its own:")
    print("   - Metrics (success rates, latency, energy)")
    print("   - Trust trajectories during attacks")
    print("   - Classification performance (P/R/F1)")
    print("   - Protection analysis")
    print("   - Confusion matrix")
    print("   - All visualizations")
    print("\n📁 Plus cross-model comparison plots!")
    print("=" * 80)
    
    # Initialize system
    print("\n🔧 Initializing system...")
    system = FixedEnhancedMidsemSystem(malicious_ratio=0.3)
    
    # Run complete evaluation
    print("\n🚀 Starting per-model evaluation...")
    print("⏱️  This will take some time as we run 4 models × 4 phases × 7 datasets")
    print("=" * 80)
    
    results = system.run_complete_evaluation()
    
    print("\n" + "=" * 80)
    print("🎉 PER-MODEL EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\n📈 Results saved in: {system.results_dir}")
    print(f"🌐 HTML Report: {os.path.join(system.results_dir, 'fixed_comprehensive_report.html')}")
    print(f"\n📊 Total model-dataset combinations evaluated: {len(system.extracted_metrics)}")
    print(f"   Expected: {len(system.gnn_models)} models × {sum(len(v) for v in system.datasets.values())} datasets = {len(system.gnn_models) * sum(len(v) for v in system.datasets.values())}")
    print("\n✅ Each dataset now has separate results for:")
    for model in system.gnn_models:
        print(f"   - {model}")
    print("\n🎯 Check the results directory for per-model folders and visualizations!")
    print("=" * 80)

if __name__ == "__main__":
    main()
