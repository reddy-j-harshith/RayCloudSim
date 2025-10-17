#!/usr/bin/env python3
"""
Enhanced Report Generator using Existing Results
Generates comprehensive visualizations and HTML reports from previously completed evaluations
"""

import os
import json
import sys
import numpy as np
from datetime import datetime
from enhanced_visualization import EnhancedVisualizationSystem
from enhanced_html_generator import EnhancedHTMLReportGenerator

def load_existing_results(results_dir: str):
    """Load results from the most recent proper evaluation"""
    
    # Find the most recent proper evaluation directory
    midsem_dir = os.path.join(results_dir, 'midsem_results')
    if not os.path.exists(midsem_dir):
        # Try alternative path
        midsem_dir = os.path.join(os.path.dirname(results_dir), 'midsem_results')
        if not os.path.exists(midsem_dir):
            print(f"❌ No midsem_results directory found at: {midsem_dir}")
            return None
    
    # Get all proper evaluation directories
    proper_dirs = [d for d in os.listdir(midsem_dir) if d.startswith('proper_evaluation_')]
    if not proper_dirs:
        print(f"❌ No proper evaluation results found in: {midsem_dir}")
        return None
    
    # Get the most recent one
    proper_dirs.sort(reverse=True)
    latest_dir = os.path.join(midsem_dir, proper_dirs[0])
    
    print(f"📂 Loading results from: {latest_dir}")
    
    # Load the JSON results
    results_file = os.path.join(latest_dir, 'proper_evaluation_results.json')
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return None
    
    try:
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        print(f"✅ Loaded results for {len(results_data)} combinations")
        return results_data, latest_dir
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

def enhance_results_with_research_data(results_data: dict, results_dir: str):
    """Enhance results with data from individual research result directories"""
    
    # Find all research result directories - check multiple locations
    base_dir = os.path.dirname(results_dir)
    research_dirs = []
    
    # Check in base directory
    if os.path.exists(base_dir):
        research_dirs.extend([d for d in os.listdir(base_dir) if d.startswith('research_results_')])
    
    # Check in RayCloudSim directory
    raycloud_dir = os.path.dirname(base_dir) if 'midsem_results' in base_dir else base_dir
    if os.path.exists(raycloud_dir):
        potential_dirs = [d for d in os.listdir(raycloud_dir) if d.startswith('research_results_')]
        for pd in potential_dirs:
            if pd not in research_dirs:
                research_dirs.append(pd)
                base_dir = raycloud_dir  # Update base_dir to correct location
    
    print(f"📊 Found {len(research_dirs)} research result directories")
    
    # Map research directories to dataset combinations
    for dataset_key in results_data.keys():
        # Find corresponding research directories (multiple per dataset/model combination)
        matching_dirs = []
        
        # Look for directories that might contain data for this combination
        for research_dir in research_dirs:
            research_path = os.path.join(base_dir, research_dir)
            
            # Check if this directory has data for our dataset
            if os.path.exists(research_path):
                # Look for phase logger data or temporal trust data
                log_files = [f for f in os.listdir(research_path) if f.endswith('.json') or f.endswith('.csv')]
                if log_files:
                    matching_dirs.append(research_path)
        
        # Take the most recent matching directory for this dataset
        if matching_dirs:
            matching_dirs.sort(reverse=True)
            research_path = matching_dirs[0]
            
            # Try to load additional data
            enhanced_data = load_research_directory_data(research_path)
            if enhanced_data:
                # Merge the enhanced data
                if 'training_results' not in results_data[dataset_key]:
                    results_data[dataset_key]['training_results'] = {}
                
                results_data[dataset_key]['training_results'].update(enhanced_data)
                print(f"   ✅ Enhanced {dataset_key} with research data")
    
    return results_data

def load_research_directory_data(research_path: str):
    """Load real data from a research results directory"""
    enhanced_data = {}
    
    try:
        print(f"   📂 Checking for real data in: {os.path.basename(research_path)}")
        
        # Look for saved temporal trust data
        temporal_csv_path = os.path.join(research_path, 'temporal_trust_data.csv')
        if os.path.exists(temporal_csv_path):
            import pandas as pd
            temporal_df = pd.read_csv(temporal_csv_path)
            enhanced_data['temporal_trust_data'] = temporal_df.to_dict('records')
            print(f"   ✅ Loaded {len(temporal_df)} temporal trust records")
        else:
            print(f"   ❌ No temporal trust data found at: {temporal_csv_path}")
            return None
        
        # Look for saved task logs
        task_logs_path = os.path.join(research_path, 'detailed_task_logs.csv')
        if os.path.exists(task_logs_path):
            import pandas as pd
            task_logs_df = pd.read_csv(task_logs_path)
            task_logs = task_logs_df.to_dict('records')
            
            # Create phase logger structure
            enhanced_data['phase_logger'] = type('PhaseLogger', (), {
                'task_logs': task_logs
            })()
            print(f"   ✅ Loaded {len(task_logs)} task log records")
        else:
            print(f"   ❌ No task logs found at: {task_logs_path}")
            return None
        
        # Load study results to get malicious nodes info
        study_results_path = os.path.join(research_path, 'study_results.json')
        if os.path.exists(study_results_path):
            with open(study_results_path, 'r') as f:
                study_data = json.load(f)
                
            # Extract malicious nodes from training results
            training_results = study_data.get('training_results', {})
            malicious_nodes = training_results.get('malicious_nodes', [])
            enhanced_data['malicious_nodes'] = malicious_nodes
            print(f"   ✅ Found {len(malicious_nodes)} malicious nodes")
        else:
            print(f"   ⚠️ No study results found, using defaults")
            enhanced_data['malicious_nodes'] = []
        
    except Exception as e:
        print(f"   ❌ Error loading research data from {research_path}: {e}")
        return None
    
    return enhanced_data

def generate_enhanced_reports(results_data: dict, results_dir: str):
    """Generate enhanced visualizations and HTML report"""
    
    print("🎨 Generating enhanced visualizations...")
    
    # Create enhanced visualization system
    viz_system = EnhancedVisualizationSystem(results_dir)
    
    # Generate all enhanced plots
    viz_system.generate_all_enhanced_plots(results_data)
    
    print("📄 Generating enhanced HTML report...")
    
    # Create HTML report generator
    html_generator = EnhancedHTMLReportGenerator(results_dir)
    
    # Generate comprehensive HTML report
    html_output_path = os.path.join(results_dir, 'enhanced_evaluation_report.html')
    html_generator.generate_enhanced_html_report(results_data, html_output_path)
    
    return html_output_path

def main():
    """Main function to generate enhanced reports from existing results"""
    
    print("🔥 ENHANCED REPORT GENERATION")
    print("=" * 80)
    print("📊 Using existing evaluation results to create comprehensive visualizations")
    print("🎯 No retraining required - utilizing completed evaluation data")
    print()
    
    # Get current directory
    current_dir = os.getcwd()
    
    # Load existing results
    result = load_existing_results(current_dir)
    if result is None:
        print("❌ Could not load existing results. Please run proper evaluation first.")
        return
    
    results_data, results_dir = result
    
    # Enhance results with research data
    print("🔬 Enhancing results with detailed research data...")
    enhanced_results = enhance_results_with_research_data(results_data, results_dir)
    
    # Generate enhanced reports
    print("🎨 Creating enhanced visualizations and reports...")
    html_report_path = generate_enhanced_reports(enhanced_results, results_dir)
    
    print()
    print("🏆 ENHANCED REPORT GENERATION COMPLETE!")
    print("=" * 80)
    print(f"📊 Enhanced plots saved to: {os.path.join(results_dir, 'enhanced_plots')}")
    print(f"📄 Enhanced HTML report: {html_report_path}")
    print()
    print("🎯 Features added:")
    print("   ✅ Individual GNN trust trajectories (separate plots per model)")
    print("   ✅ Trust distribution analysis (malicious vs honest nodes)")
    print("   ✅ Trust-based offloading effectiveness analysis")
    print("   ✅ Attack impact visualization (with/without trust protection)")
    print("   ✅ Median trust values and separation analysis")
    print("   ✅ Comprehensive performance comparison heatmaps")
    print("   ✅ Radar charts for dataset performance profiles")
    print("   ✅ Enhanced HTML report with embedded visualizations")
    print()
    print("🚀 Open the HTML report in your browser to view all visualizations!")

if __name__ == "__main__":
    main()