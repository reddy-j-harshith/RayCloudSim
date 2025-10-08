# GNN Trust Regression Attack Simulation Analysis Report

Generated on: 2025-10-08 13:14:17

## Executive Summary

This report analyzes the performance of GNN-based trust regression models for malicious node detection across different attack scenarios and network topologies.

**Best Performance**: GAT with Static Threshold (F1: 1.0000)

## Key Findings

### Model Performance Ranking
1. **GCN**: 0.8918 average F1 score
2. **GraphSAGE**: 0.8918 average F1 score
3. **Transformer**: 0.8807 average F1 score
4. **GAT**: 0.8629 average F1 score

### Scenario Effectiveness
1. **Adaptive Threshold**: 1.0000 average F1 score
2. **Static Threshold**: 1.0000 average F1 score
3. **Statistical Z-Score**: 1.0000 average F1 score
4. **IQR Outlier Detection**: 0.9569 average F1 score
5. **Bottom 20% Percentile**: 0.4519 average F1 score

## Detailed Analysis

- **Average Trust Value**: 0.7567
- **Average Trust Std Dev**: 0.1395
- **Total Malicious Nodes**: 180
- **Total Detections**: 325
- **True Positives**: 180
- **False Positives**: 145

## Recommendations

1. **Model Selection**: Use the best performing model for production deployment
2. **Threshold Tuning**: Fine-tune detection thresholds based on network characteristics
3. **Ensemble Methods**: Consider combining multiple detection scenarios
4. **Continuous Learning**: Retrain models periodically with new attack patterns

## Technical Notes

- Trust values are continuous in [0,1] range (regression approach)
- Real network topologies used (pakistan, topo4mec)
- Multiple detection methods evaluated (threshold, statistical)
- Performance metrics: Accuracy, Precision, Recall, F1-Score
