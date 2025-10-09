#!/usr/bin/env python3
"""
Retrain Specific Combinations to Generate Missing Detailed Data
This script retrains specific dataset/model combinations to generate the temporal trust data 
and task logs needed for enhanced visualizations.
"""

import os
import sys
from datetime import datetime
from research_attack_aware_system import ResearchAttackAwareSystem

class DetailedDataRetrainer:
    """Retrain specific combinations to get detailed visualization data"""
    
    def __init__(self):
        self.base_dir = os.getcwd()
        self.malicious_ratio = 0.3
        
    def get_missing_combinations(self):
        """Identify combinations that are missing detailed data"""
        missing_combinations = []
        
        # Check existing research directories for missing data
        research_dirs = [d for d in os.listdir(self.base_dir) if d.startswith('research_results_')]
        
        # Define all expected combinations
        expected_combinations = [
            ('pakistan', 'Tuple30K', 'GAT'),
            ('pakistan', 'Tuple30K', 'GraphSAGE'), 
            ('pakistan', 'Tuple30K', 'GCN'),
            ('pakistan', 'Tuple30K', 'Transformer'),
            ('pakistan', 'Tuple50K', 'GAT'),
            ('pakistan', 'Tuple50K', 'GraphSAGE'),
            ('pakistan', 'Tuple50K', 'GCN'),
            ('pakistan', 'Tuple50K', 'Transformer'),
            ('pakistan', 'Tuple100K', 'GAT'),
            ('pakistan', 'Tuple100K', 'GraphSAGE'),
            ('pakistan', 'Tuple100K', 'GCN'),
            ('pakistan', 'Tuple100K', 'Transformer'),
            ('topo4mec', '25N50E', 'GAT'),
            ('topo4mec', '25N50E', 'GraphSAGE'),
            ('topo4mec', '25N50E', 'GCN'),
            ('topo4mec', '25N50E', 'Transformer'),
            ('topo4mec', '50N50E', 'GAT'),
            ('topo4mec', '50N50E', 'GraphSAGE'),
            ('topo4mec', '50N50E', 'GCN'),
            ('topo4mec', '50N50E', 'Transformer'),
            ('topo4mec', '100N150E', 'GAT'),
            ('topo4mec', '100N150E', 'GraphSAGE'),
            ('topo4mec', '100N150E', 'GCN'),
            ('topo4mec', '100N150E', 'Transformer'),
        ]
        
        for dataset_name, dataset_flag, model_type in expected_combinations:
            has_detailed_data = self._check_for_detailed_data(dataset_name, dataset_flag, model_type)
            if not has_detailed_data:
                missing_combinations.append((dataset_name, dataset_flag, model_type))
        
        return missing_combinations
    
    def _check_for_detailed_data(self, dataset_name: str, dataset_flag: str, model_type: str):
        """Check if detailed data exists for a specific combination"""
        
        # Look for research directories that might contain this combination's data
        research_dirs = [d for d in os.listdir(self.base_dir) if d.startswith('research_results_')]
        
        for research_dir in research_dirs:
            research_path = os.path.join(self.base_dir, research_dir)
            
            # Check if this directory has the detailed data files we need
            temporal_csv = os.path.join(research_path, 'temporal_trust_data.csv')
            task_logs_csv = os.path.join(research_path, 'detailed_task_logs.csv')
            
            if os.path.exists(temporal_csv) and os.path.exists(task_logs_csv):
                # Check if this directory corresponds to our dataset/model combination
                # This is a heuristic - in practice we'd need better mapping
                dataset_dir = os.path.join(research_path, f"{dataset_name}_{dataset_flag}")
                if os.path.exists(dataset_dir):
                    return True
        
        return False
    
    def retrain_combination(self, dataset_name: str, dataset_flag: str, model_type: str):
        """Retrain a specific combination to generate detailed data"""
        
        print(f"🔥 RETRAINING: {dataset_name}/{dataset_flag} with {model_type}")
        print(f"   🎯 Generating detailed temporal trust data and task logs")
        
        try:
            # Initialize research system
            system = ResearchAttackAwareSystem(malicious_ratio=self.malicious_ratio)
            
            # Load dataset and setup
            trainset, testset, dataset_info = system.load_dataset(dataset_name, dataset_flag)
            
            # Create network graph based on dataset info
            import networkx as nx
            if 'Nodes' in dataset_info and 'Edges' in dataset_info:
                nodes = dataset_info['Nodes']
                edges = dataset_info['Edges']
                
                network_graph = nx.Graph()
                network_graph.add_nodes_from(range(nodes))
                # Check if edges is a list of tuples or number
                if isinstance(edges, list):
                    network_graph.add_edges_from(edges)
                else:
                    # If edges is a number, create a connected graph
                    for i in range(nodes):
                        for j in range(i+1, min(i+3, nodes)):  # Connect to next 2 nodes
                            network_graph.add_edge(i, j)
            else:
                # Fallback to complete graph
                import networkx as nx
                if 'pakistan' in dataset_name.lower():
                    if 'tuple30k' in dataset_flag.lower():
                        num_nodes = 8
                    elif 'tuple50k' in dataset_flag.lower():
                        num_nodes = 11
                    else:  # tuple100k
                        num_nodes = 15
                elif 'topo4mec' in dataset_name.lower():
                    if '25n50e' in dataset_flag.lower():
                        num_nodes = 25
                    elif '50n50e' in dataset_flag.lower():
                        num_nodes = 50
                    else:  # 100n150e
                        num_nodes = 100
                else:
                    num_nodes = 8
                
                network_graph = nx.complete_graph(num_nodes)
            
            # Select malicious nodes
            malicious_nodes = system.select_malicious_nodes(network_graph)
            honest_nodes = [n for n in network_graph.nodes() if n not in malicious_nodes]
            
            print(f"   🌐 Network: {len(network_graph.nodes())} nodes ({len(malicious_nodes)} malicious)")
            print(f"   📊 Training: {len(trainset)} tasks, Testing: {len(testset)} tasks")
            
            # Initialize trust matrix and attack simulator
            trust_matrix = system.initialize_trust_matrix(network_graph)
            from research_attack_aware_system import AdvancedAttackSimulator
            system.attack_simulator = AdvancedAttackSimulator(malicious_nodes)
            
            # Run training phase with detailed logging
            print(f"   📚 Phase 1: Training simulation...")
            train_results = system.simulate_task_execution_phase(
                trainset, network_graph, trust_matrix, malicious_nodes,
                honest_nodes, phase='training', enable_temporal_logging=True
            )
            
            print(f"   ✅ Training: {train_results['successful_tasks']} success, {train_results['failed_tasks']} failed")
            
            # Train GNN models
            print(f"   🧠 Phase 2: GNN training...")
            gnn_results = system.train_gnn_models(train_results, network_graph)
            
            # Verify the target model was trained
            if model_type not in gnn_results:
                raise ValueError(f"Model {model_type} training failed!")
            
            print(f"   ✅ Model trained: RMSE={gnn_results[model_type].get('train_rmse', 0):.4f}")
            
            # Run testing phase
            print(f"   🧪 Phase 3: Testing on test set...")
            test_results = system.simulate_task_execution_phase(
                testset, network_graph, trust_matrix, malicious_nodes,
                honest_nodes, phase='testing', use_detection=True, 
                enable_temporal_logging=True
            )
            
            print(f"   ✅ Testing: {test_results['successful_tasks']} success, {test_results['failed_tasks']} failed")
            
            # Run evaluation phases
            print(f"   📈 Phase 4: Downstream evaluation...")
            downstream_metrics = system.evaluate_gnns_on_phase(
                test_results, network_graph, gnn_results, malicious_nodes
            )
            
            print(f"   🔍 Phase 5: Attack detection...")
            detection_results = system.detect_malicious_nodes(
                train_results, malicious_nodes, honest_nodes
            )
            
            print(f"   🚀 Phase 6: Trust-based offloading...")
            offloading_results = system.analyze_trust_based_offloading(
                train_results, test_results, malicious_nodes
            )
            
            # Compile comprehensive results
            study_results = {
                'dataset_info': {
                    'name': dataset_name,
                    'flag': dataset_flag,
                    'nodes': len(network_graph.nodes()),
                    'edges': len(network_graph.edges()),
                    'malicious_nodes': malicious_nodes,
                    'honest_nodes': honest_nodes
                },
                'training_results': train_results,
                'gnn_results': gnn_results,
                'testing_results': test_results,
                'downstream_metrics': downstream_metrics,
                'detection_results': detection_results,
                'offloading_results': offloading_results
            }
            
            # Save results with detailed data
            system.save_dataset_results(study_results, dataset_name, dataset_flag)
            
            print(f"   ✅ {model_type} COMPLETE with detailed data saved")
            return True
            
        except Exception as e:
            print(f"   ❌ {dataset_name}_{dataset_flag}/{model_type} FAILED: {str(e)}")
            return False
    
    def retrain_missing_combinations(self, combinations_to_retrain=None):
        """Retrain all missing combinations or specific ones"""
        
        if combinations_to_retrain is None:
            missing_combinations = self.get_missing_combinations()
        else:
            missing_combinations = combinations_to_retrain
        
        if not missing_combinations:
            print("✅ All combinations already have detailed data!")
            return
        
        print(f"🔥 DETAILED DATA RETRAINING")
        print("=" * 80)
        print(f"📊 Found {len(missing_combinations)} combinations missing detailed data")
        print(f"🎯 Will retrain to generate temporal trust data and task logs")
        print()
        
        successful_retrains = 0
        failed_retrains = 0
        
        for i, (dataset_name, dataset_flag, model_type) in enumerate(missing_combinations, 1):
            print(f"[{i}/{len(missing_combinations)}] Retraining {dataset_name}/{dataset_flag} - {model_type}")
            
            success = self.retrain_combination(dataset_name, dataset_flag, model_type)
            
            if success:
                successful_retrains += 1
            else:
                failed_retrains += 1
            
            print()
        
        print("🏆 DETAILED DATA RETRAINING COMPLETE!")
        print("=" * 80)
        print(f"✅ Successful: {successful_retrains}")
        print(f"❌ Failed: {failed_retrains}")
        print(f"📊 Total: {len(missing_combinations)}")
        print()
        print("🎨 Now you can run generate_enhanced_reports.py to create visualizations!")

def main():
    """Main function for retraining missing combinations"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Retrain combinations to generate detailed visualization data')
    parser.add_argument('--dataset', help='Specific dataset to retrain (e.g., pakistan)')
    parser.add_argument('--flag', help='Specific dataset flag (e.g., Tuple30K)')
    parser.add_argument('--model', help='Specific model to retrain (e.g., GAT)')
    parser.add_argument('--list', action='store_true', help='List missing combinations without retraining')
    
    args = parser.parse_args()
    
    retrainer = DetailedDataRetrainer()
    
    if args.list:
        missing = retrainer.get_missing_combinations()
        print(f"📊 Missing detailed data for {len(missing)} combinations:")
        for dataset, flag, model in missing:
            print(f"   - {dataset}/{flag} - {model}")
        return
    
    if args.dataset and args.flag and args.model:
        # Retrain specific combination
        combinations_to_retrain = [(args.dataset, args.flag, args.model)]
        retrainer.retrain_missing_combinations(combinations_to_retrain)
    else:
        # Retrain all missing combinations
        retrainer.retrain_missing_combinations()

if __name__ == "__main__":
    main()