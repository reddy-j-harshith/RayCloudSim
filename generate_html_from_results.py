"""
Standalone HTML Report Generator
Generates per-model HTML report from existing experiment results
Can be run separately from the main evaluation
"""

import os
import json
import sys
from generate_permodel_html_report import generate_permodel_html_report

def load_extracted_metrics(results_dir):
    """Load all extracted metrics from JSON files in the results directory"""
    extracted_metrics = {}
    
    print(f"📂 Scanning results directory: {results_dir}")
    
    # Iterate through dataset directories
    for dataset_name in os.listdir(results_dir):
        dataset_path = os.path.join(results_dir, dataset_name)
        
        # Skip non-directories
        if not os.path.isdir(dataset_path):
            continue
        
        # Look for model subdirectories
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            
            # Check if it's a model directory
            if os.path.isdir(item_path) and item.startswith('model_'):
                # Extract model type from directory name
                model_key = item.replace('model_', '')  # gat, graphsage, gcn, transformer
                model_type = model_key.upper() if model_key != 'graphsage' else 'GraphSAGE'
                
                # Look for results JSON
                json_pattern = f"{dataset_name}_{model_key}_results.json"
                json_path = os.path.join(item_path, json_pattern)
                
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            metrics = json.load(f)
                        
                        # Store with key format: dataset_name_MODEL
                        key = f"{dataset_name}_{model_type}"
                        extracted_metrics[key] = metrics
                        print(f"  ✅ Loaded: {key}")
                    except Exception as e:
                        print(f"  ⚠️ Error loading {json_path}: {e}")
    
    return extracted_metrics

def generate_report_from_results(results_dir):
    """Generate HTML report from existing results"""
    print(f"\n🔬 Generating HTML Report from Existing Results")
    print(f"{'='*70}")
    
    # Load extracted metrics
    extracted_metrics = load_extracted_metrics(results_dir)
    
    if not extracted_metrics:
        print(f"❌ No extracted metrics found in {results_dir}")
        print(f"   Make sure the directory contains dataset folders with model_* subdirectories")
        return None
    
    print(f"\n📊 Found {len(extracted_metrics)} model evaluations")
    
    # Infer datasets structure from metrics
    datasets = {}
    for key in extracted_metrics.keys():
        # Extract dataset type and subset
        # Format: pakistan_Tuple30K_GAT -> pakistan, Tuple30K
        parts = key.rsplit('_', 1)  # Split from right, max 1 split
        dataset_full = parts[0]
        
        # Try to split dataset_full into type and subset
        if 'pakistan' in dataset_full.lower():
            dataset_type = 'pakistan'
            subset_name = dataset_full.replace('pakistan_', '')
        elif 'topo4mec' in dataset_full.lower():
            dataset_type = 'topo4mec'
            subset_name = dataset_full.replace('topo4mec_', '')
        else:
            # Fallback: use first part as type, rest as subset
            type_parts = dataset_full.split('_', 1)
            dataset_type = type_parts[0] if len(type_parts) > 0 else dataset_full
            subset_name = type_parts[1] if len(type_parts) > 1 else ''
        
        if dataset_type not in datasets:
            datasets[dataset_type] = []
        if subset_name and subset_name not in datasets[dataset_type]:
            datasets[dataset_type].append(subset_name)
    
    print(f"\n📁 Inferred datasets structure:")
    for dataset_type, subsets in datasets.items():
        print(f"  • {dataset_type}: {', '.join(subsets)}")
    
    # Infer malicious ratio from first metrics entry
    malicious_ratio = 0.3  # Default
    if extracted_metrics:
        first_metrics = next(iter(extracted_metrics.values()))
        if 'network_info' in first_metrics:
            total = first_metrics['network_info'].get('total_nodes', 0)
            malicious = first_metrics['network_info'].get('malicious_nodes', 0)
            if total > 0:
                malicious_ratio = malicious / total
    
    print(f"\n📊 Malicious ratio: {malicious_ratio:.1%}")
    
    # Generate HTML report
    print(f"\n📄 Generating HTML report...")
    report_path = generate_permodel_html_report(
        extracted_metrics=extracted_metrics,
        datasets=datasets,
        malicious_ratio=malicious_ratio,
        output_dir=results_dir
    )
    
    print(f"\n✅ HTML Report Generated Successfully!")
    print(f"{'='*70}")
    print(f"📁 Location: {report_path}")
    print(f"🌐 Open in browser to view interactive report")
    
    return report_path

def main():
    """Main execution"""
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Use most recent results directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        midsem_results = os.path.join(base_dir, 'midsem_results')
        
        if os.path.exists(midsem_results):
            # Find most recent evaluation directory
            eval_dirs = [d for d in os.listdir(midsem_results) 
                        if os.path.isdir(os.path.join(midsem_results, d)) 
                        and 'evaluation' in d.lower()]
            
            if eval_dirs:
                # Sort by name (which includes timestamp)
                eval_dirs.sort(reverse=True)
                results_dir = os.path.join(midsem_results, eval_dirs[0])
                print(f"🔍 Using most recent results: {eval_dirs[0]}")
            else:
                print(f"❌ No evaluation directories found in {midsem_results}")
                return
        else:
            print(f"❌ midsem_results directory not found")
            print(f"Usage: python {sys.argv[0]} <results_directory>")
            return
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    generate_report_from_results(results_dir)

if __name__ == "__main__":
    main()
