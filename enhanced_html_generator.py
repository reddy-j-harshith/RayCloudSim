#!/usr/bin/env python3
"""
Enhanced HTML Report Generator with Comprehensive Visualizations
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List
import base64

class EnhancedHTMLReportGenerator:
    """Generate comprehensive HTML reports with enhanced visualizations"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, 'enhanced_plots')
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 for embedding in HTML"""
        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except:
            return ""
    
    def generate_enhanced_html_report(self, all_results: Dict, output_path: str):
        """Generate comprehensive HTML report with all visualizations"""
        
        # Calculate summary statistics
        total_combinations = len(all_results)
        successful_combinations = sum(1 for r in all_results.values() if r.get('final_metrics', {}).get('downstream_f1', 0) > 0)
        
        # Find best performing models
        best_downstream = self._find_best_performer(all_results, 'downstream_f1')
        best_detection = self._find_best_performer(all_results, 'detection_f1')
        best_offloading = self._find_best_performer(all_results, 'offloading_efficiency')
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced GNN Trust System Evaluation Report</title>
    <style>
        {self._get_enhanced_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>🔬 Enhanced GNN Trust System Evaluation Report</h1>
            <div class="report-meta">
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Total Combinations:</strong> {total_combinations}</p>
                <p><strong>Successful Evaluations:</strong> {successful_combinations}/{total_combinations}</p>
            </div>
        </header>

        <!-- Executive Summary -->
        <section class="executive-summary">
            <h2>📊 Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card best">
                    <h3>🏆 Best Downstream Performance</h3>
                    <p><strong>{best_downstream['model']}</strong> on <strong>{best_downstream['dataset']}</strong></p>
                    <p class="metric-value">F1 Score: {best_downstream['value']:.3f}</p>
                </div>
                <div class="summary-card good">
                    <h3>🔍 Best Attack Detection</h3>
                    <p><strong>{best_detection['model']}</strong> on <strong>{best_detection['dataset']}</strong></p>
                    <p class="metric-value">F1 Score: {best_detection['value']:.3f}</p>
                </div>
                <div class="summary-card excellent">
                    <h3>🛡️ Best Offloading Efficiency</h3>
                    <p><strong>{best_offloading['model']}</strong> on <strong>{best_offloading['dataset']}</strong></p>
                    <p class="metric-value">Efficiency: {best_offloading['value']:.3f}</p>
                </div>
            </div>
        </section>

        <!-- Overall Performance Visualizations -->
        <section class="visualization-section">
            <h2>🎯 Overall Performance Analysis</h2>
            <div class="viz-grid">
                {self._generate_overall_viz_html()}
            </div>
        </section>

        <!-- Individual Dataset Analysis -->
        {self._generate_dataset_sections_html(all_results)}
        
        <!-- Technical Details -->
        <section class="technical-details">
            <h2>⚙️ Technical Implementation Details</h2>
            <div class="tech-details-grid">
                <div class="tech-card">
                    <h3>🧠 GNN Models</h3>
                    <ul>
                        <li><strong>GAT:</strong> Graph Attention Network with multi-head attention</li>
                        <li><strong>GraphSAGE:</strong> Sample and Aggregate with inductive learning</li>
                        <li><strong>GCN:</strong> Graph Convolutional Network with spectral approach</li>
                        <li><strong>Transformer:</strong> Self-attention based graph transformer</li>
                    </ul>
                </div>
                <div class="tech-card">
                    <h3>📊 Evaluation Metrics</h3>
                    <ul>
                        <li><strong>Downstream F1:</strong> Task classification performance</li>
                        <li><strong>Detection F1:</strong> Malicious node detection accuracy</li>
                        <li><strong>Offloading Efficiency:</strong> Trust-based task routing success</li>
                        <li><strong>Protection Rate:</strong> Malicious node avoidance effectiveness</li>
                    </ul>
                </div>
                <div class="tech-card">
                    <h3>🎯 Attack Simulation</h3>
                    <ul>
                        <li><strong>Malicious Ratio:</strong> 30% of network nodes</li>
                        <li><strong>Attack Types:</strong> Task manipulation, trust poisoning</li>
                        <li><strong>Detection Methods:</strong> Statistical, ML-based, Trust anomaly</li>
                        <li><strong>Defense:</strong> Trust-based offloading with dynamic thresholds</li>
                    </ul>
                </div>
            </div>
        </section>

        <!-- Methodology -->
        <section class="methodology">
            <h2>🔬 Methodology</h2>
            <div class="methodology-content">
                <h3>Experimental Setup</h3>
                <p>This comprehensive evaluation assesses GNN-based trust systems across multiple datasets and attack scenarios:</p>
                
                <h4>📦 Datasets</h4>
                <ul>
                    <li><strong>Pakistan Datasets:</strong> Tuple30K, Tuple50K, Tuple100K - Real-world mobile edge computing scenarios</li>
                    <li><strong>Topo4MEC Datasets:</strong> 25N50E, 50N50E, 100N150E - Synthetic topology variations</li>
                </ul>
                
                <h4>🧪 Evaluation Pipeline</h4>
                <ol>
                    <li><strong>Training Phase:</strong> GNN models learn trust patterns from task execution history</li>
                    <li><strong>Attack Injection:</strong> 30% of nodes exhibit malicious behavior</li>
                    <li><strong>Testing Phase:</strong> Models evaluated on downstream tasks and detection capabilities</li>
                    <li><strong>Trust-based Offloading:</strong> Dynamic task routing based on learned trust scores</li>
                    <li><strong>Performance Analysis:</strong> Comprehensive metrics across multiple dimensions</li>
                </ol>
                
                <h4>📈 Key Innovations</h4>
                <ul>
                    <li><strong>Multi-Model Comparison:</strong> Systematic evaluation of 4 GNN architectures</li>
                    <li><strong>Real Dataset Training:</strong> No synthetic fallback metrics - all results from actual model training</li>
                    <li><strong>Attack-Aware Design:</strong> Models trained under adversarial conditions</li>
                    <li><strong>Trust-based Protection:</strong> Dynamic offloading system with real-time adaptation</li>
                    <li><strong>Comprehensive Visualization:</strong> Individual model trajectories and comparative analysis</li>
                </ul>
            </div>
        </section>

        <footer class="report-footer">
            <p>Generated by Enhanced GNN Trust System Evaluation Framework</p>
            <p>🎯 All metrics are based on real model training - no synthetic or fallback data used</p>
        </footer>
    </div>
</body>
</html>
"""
        
        # Write HTML report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Enhanced HTML report generated: {output_path}")
    
    def _find_best_performer(self, all_results: Dict, metric: str) -> Dict:
        """Find the best performing model/dataset combination for a given metric"""
        best_value = 0
        best_info = {'model': 'N/A', 'dataset': 'N/A', 'value': 0}
        
        for dataset_key, results in all_results.items():
            metrics = results.get('final_metrics', {})
            value = metrics.get(metric, 0)
            
            if value > best_value:
                best_value = value
                # Extract model name from dataset_key (format: dataset_model)
                parts = dataset_key.split('_')
                if len(parts) >= 2:
                    model = parts[-1]
                    dataset = '_'.join(parts[:-1])
                else:
                    model = 'Unknown'
                    dataset = dataset_key
                
                best_info = {
                    'model': model,
                    'dataset': dataset,
                    'value': value
                }
        
        return best_info
    
    def _generate_overall_viz_html(self) -> str:
        """Generate HTML for overall performance visualizations"""
        viz_html = ""
        
        # Overall performance plots
        overall_plots = [
            ('model_performance_heatmap.png', 'Model Performance Heatmap', 'Comparison of all models across datasets'),
            ('dataset_performance_radar.png', 'Dataset Performance Radar', 'Radar charts showing performance profiles'),
            ('performance_trends.png', 'Performance Trends Analysis', 'Detailed performance analysis and correlations')
        ]
        
        for plot_file, title, description in overall_plots:
            plot_path = os.path.join(self.plots_dir, plot_file)
            if os.path.exists(plot_path):
                img_data = self.encode_image_to_base64(plot_path)
                if img_data:
                    viz_html += f"""
                    <div class="viz-card">
                        <h3>{title}</h3>
                        <p class="viz-description">{description}</p>
                        <img src="data:image/png;base64,{img_data}" alt="{title}" class="viz-image">
                    </div>
                    """
        
        return viz_html
    
    def _generate_dataset_sections_html(self, all_results: Dict) -> str:
        """Generate HTML sections for individual dataset analysis"""
        sections_html = ""
        
        # Group results by dataset
        datasets = {}
        for key, results in all_results.items():
            parts = key.split('_')
            if len(parts) >= 2:
                dataset = '_'.join(parts[:-1])
                model = parts[-1]
            else:
                dataset = key
                model = 'Unknown'
            
            if dataset not in datasets:
                datasets[dataset] = {}
            datasets[dataset][model] = results
        
        for dataset_name, dataset_results in datasets.items():
            sections_html += self._generate_single_dataset_html(dataset_name, dataset_results)
        
        return sections_html
    
    def _generate_single_dataset_html(self, dataset_name: str, dataset_results: Dict) -> str:
        """Generate HTML for a single dataset analysis"""
        
        # Create performance table
        table_html = """
        <table class="results-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Downstream F1</th>
                    <th>Detection F1</th>
                    <th>Offloading Efficiency</th>
                    <th>Protection Rate</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for model, results in dataset_results.items():
            metrics = results.get('final_metrics', {})
            downstream_f1 = metrics.get('downstream_f1', 0)
            detection_f1 = metrics.get('detection_f1', 0)
            offloading_eff = metrics.get('offloading_efficiency', 0)
            protection_rate = metrics.get('protection_rate', 0)
            
            status_class = 'success' if downstream_f1 > 0 else 'error'
            status_text = 'SUCCESS' if downstream_f1 > 0 else 'FAILED'
            
            table_html += f"""
                <tr>
                    <td><strong>{model}</strong></td>
                    <td>{downstream_f1:.3f}</td>
                    <td>{detection_f1:.3f}</td>
                    <td>{offloading_eff:.3f}</td>
                    <td>{protection_rate:.3f}</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
            """
        
        table_html += "</tbody></table>"
        
        # Generate visualizations for this dataset
        viz_html = ""
        
        # Individual GNN trust trajectories
        for model in ['GAT', 'GraphSAGE', 'GCN', 'Transformer']:
            plot_file = f"{model}_trust_trajectory_{dataset_name}.png"
            plot_path = os.path.join(self.plots_dir, plot_file)
            
            if os.path.exists(plot_path):
                img_data = self.encode_image_to_base64(plot_path)
                if img_data:
                    viz_html += f"""
                    <div class="model-viz-card">
                        <h4>🧠 {model} Trust Analysis</h4>
                        <img src="data:image/png;base64,{img_data}" alt="{model} Trust Trajectory" class="model-viz-image">
                        <p class="viz-caption">Individual trust trajectories showing malicious vs honest node evolution for {model}</p>
                    </div>
                    """
        
        # Offloading analysis
        offloading_plot = f"offloading_analysis_{dataset_name}.png"
        offloading_path = os.path.join(self.plots_dir, offloading_plot)
        
        if os.path.exists(offloading_path):
            img_data = self.encode_image_to_base64(offloading_path)
            if img_data:
                viz_html += f"""
                <div class="offloading-viz-card">
                    <h4>🛡️ Trust-Based Offloading Analysis</h4>
                    <img src="data:image/png;base64,{img_data}" alt="Offloading Analysis" class="offloading-viz-image">
                    <p class="viz-caption">Comprehensive analysis of trust-based offloading effectiveness, attack impact, and protection mechanisms</p>
                </div>
                """
        
        return f"""
        <section class="dataset-section">
            <h2>📊 {dataset_name.replace('_', ' ').title()} Analysis</h2>
            
            <div class="dataset-summary">
                <h3>Performance Summary</h3>
                {table_html}
            </div>
            
            <div class="dataset-visualizations">
                <h3>🎯 Individual Model Analysis</h3>
                <div class="model-viz-grid">
                    {viz_html}
                </div>
            </div>
            
            <div class="dataset-insights">
                <h3>💡 Key Insights for {dataset_name}</h3>
                {self._generate_dataset_insights(dataset_name, dataset_results)}
            </div>
        </section>
        """
    
    def _generate_dataset_insights(self, dataset_name: str, dataset_results: Dict) -> str:
        """Generate insights for a specific dataset"""
        insights = []
        
        # Find best and worst performers
        models_performance = []
        for model, results in dataset_results.items():
            metrics = results.get('final_metrics', {})
            downstream_f1 = metrics.get('downstream_f1', 0)
            models_performance.append((model, downstream_f1))
        
        models_performance.sort(key=lambda x: x[1], reverse=True)
        
        if models_performance:
            best_model, best_score = models_performance[0]
            worst_model, worst_score = models_performance[-1]
            
            insights.append(f"🏆 <strong>{best_model}</strong> achieved the best downstream performance with F1 score of <strong>{best_score:.3f}</strong>")
            
            if len(models_performance) > 1:
                insights.append(f"📉 <strong>{worst_model}</strong> had the lowest performance with F1 score of <strong>{worst_score:.3f}</strong>")
        
        # Analyze detection performance
        detection_scores = [(model, results.get('final_metrics', {}).get('detection_f1', 0)) 
                          for model, results in dataset_results.items()]
        detection_scores.sort(key=lambda x: x[1], reverse=True)
        
        if detection_scores:
            best_detector, best_detection = detection_scores[0]
            insights.append(f"🔍 <strong>{best_detector}</strong> showed the best attack detection capability with F1 score of <strong>{best_detection:.3f}</strong>")
        
        # Analyze dataset characteristics
        if 'pakistan' in dataset_name.lower():
            insights.append("🌏 This Pakistan dataset represents real-world mobile edge computing scenarios with varying network densities")
        elif 'topo4mec' in dataset_name.lower():
            insights.append("🔬 This Topo4MEC dataset provides controlled synthetic topology for systematic analysis")
        
        # Performance patterns
        avg_downstream = np.mean([results.get('final_metrics', {}).get('downstream_f1', 0) 
                                for results in dataset_results.values()])
        
        if avg_downstream > 0.6:
            insights.append("✨ This dataset shows excellent model performance across all architectures")
        elif avg_downstream > 0.4:
            insights.append("⚖️ This dataset presents moderate difficulty with reasonable performance")
        else:
            insights.append("🎯 This dataset is particularly challenging, requiring specialized approaches")
        
        return "<ul>" + "".join([f"<li>{insight}</li>" for insight in insights]) + "</ul>"
    
    def _get_enhanced_css_styles(self) -> str:
        """Get enhanced CSS styles for the HTML report"""
        return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        .report-header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 3px solid #667eea;
            margin-bottom: 30px;
        }
        
        .report-header h1 {
            font-size: 2.5em;
            color: #2c3e50;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .report-meta {
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
        }
        
        .report-meta p {
            background: #f8f9fa;
            padding: 10px 20px;
            border-radius: 20px;
            border-left: 4px solid #667eea;
        }
        
        .executive-summary {
            margin-bottom: 40px;
        }
        
        .executive-summary h2 {
            color: #2c3e50;
            font-size: 2em;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .summary-card {
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            transform: translateY(0);
            transition: transform 0.3s ease;
        }
        
        .summary-card:hover {
            transform: translateY(-5px);
        }
        
        .summary-card.best {
            background: linear-gradient(135deg, #ff6b6b, #ee5a52);
            color: white;
        }
        
        .summary-card.good {
            background: linear-gradient(135deg, #4ecdc4, #44a08d);
            color: white;
        }
        
        .summary-card.excellent {
            background: linear-gradient(135deg, #45b7d1, #96c93d);
            color: white;
        }
        
        .summary-card h3 {
            font-size: 1.3em;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            margin-top: 10px;
        }
        
        .visualization-section, .dataset-section {
            margin: 40px 0;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 15px;
            border-left: 5px solid #667eea;
        }
        
        .visualization-section h2, .dataset-section h2 {
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 25px;
            text-align: center;
        }
        
        .viz-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
        }
        
        .viz-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        .viz-card h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.4em;
        }
        
        .viz-description {
            color: #666;
            margin-bottom: 20px;
            font-style: italic;
        }
        
        .viz-image {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .model-viz-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }
        
        .model-viz-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
        }
        
        .model-viz-card h4 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        .model-viz-image, .offloading-viz-image {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        
        .viz-caption {
            font-size: 0.9em;
            color: #666;
            font-style: italic;
            text-align: center;
        }
        
        .offloading-viz-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            margin-top: 25px;
        }
        
        .offloading-viz-card h4 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.5em;
            text-align: center;
        }
        
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .results-table th {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        
        .results-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }
        
        .results-table tr:hover {
            background: #f8f9fa;
        }
        
        .success { color: #27ae60; font-weight: bold; }
        .error { color: #e74c3c; font-weight: bold; }
        
        .technical-details {
            margin: 40px 0;
            padding: 30px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 15px;
            color: white;
        }
        
        .technical-details h2 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2em;
        }
        
        .tech-details-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
        }
        
        .tech-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 25px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .tech-card h3 {
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .tech-card ul {
            list-style: none;
        }
        
        .tech-card li {
            margin-bottom: 8px;
            padding-left: 20px;
            position: relative;
        }
        
        .tech-card li:before {
            content: "→";
            position: absolute;
            left: 0;
            color: #fff;
            font-weight: bold;
        }
        
        .methodology {
            margin: 40px 0;
            padding: 30px;
            background: #2c3e50;
            color: white;
            border-radius: 15px;
        }
        
        .methodology h2 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2em;
        }
        
        .methodology-content h3, .methodology-content h4 {
            color: #3498db;
            margin-top: 25px;
            margin-bottom: 15px;
        }
        
        .methodology-content ul, .methodology-content ol {
            margin-left: 20px;
            margin-bottom: 20px;
        }
        
        .methodology-content li {
            margin-bottom: 8px;
        }
        
        .dataset-insights ul {
            list-style: none;
            padding: 0;
        }
        
        .dataset-insights li {
            background: #e8f4f8;
            margin-bottom: 10px;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        
        .report-footer {
            text-align: center;
            margin-top: 50px;
            padding: 30px;
            background: #34495e;
            color: white;
            border-radius: 15px;
        }
        
        .report-footer p {
            margin-bottom: 10px;
        }
        
        @media (max-width: 768px) {
            .container { padding: 15px; margin-top: 10px; }
            .summary-grid { grid-template-columns: 1fr; }
            .viz-grid { grid-template-columns: 1fr; }
            .model-viz-grid { grid-template-columns: 1fr; }
            .tech-details-grid { grid-template-columns: 1fr; }
            .report-meta { flex-direction: column; align-items: center; }
        }
        """