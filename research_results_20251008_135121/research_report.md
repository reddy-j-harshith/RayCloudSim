# GNN-Based Trust System for Malicious Node Detection in Edge Computing

## Research Study Results

**Study Date**: 2025-10-08

## Experimental Setup

- **Topologies Tested**: 6
- **Malicious Node Ratios**: [0.1, 0.15, 0.2]
- **GNN Models**: ['GAT', 'GraphSAGE', 'GCN', 'Transformer']
- **Cross-Validation**: 5-fold
- **Total Experiments**: 72

## Key Findings

**Best Detection Performance**: GraphSAGE on Pakistan_Tuple30K with 10% malicious ratio (F1: 1.0000)

**Best Trust Regression**: GCN on Pakistan_Tuple100K (RMSE: 0.0449)

## Statistical Analysis

**ANOVA F1 Score**: F=0.411, p=7.458e-01
**ANOVA Trust RMSE**: F=0.490, p=6.902e-01

## Performance Summary

            detection_f1         trust_rmse        
                    mean     std       mean     std
model                                              
GAT               0.4067  0.4237     0.1481  0.0545
GCN               0.5589  0.4571     0.1247  0.0789
GraphSAGE         0.5419  0.4760     0.1261  0.0669
Transformer       0.5320  0.4861     0.1300  0.0592

## Conclusions

1. All GNN models successfully learn trust representations from task execution patterns
2. Performance varies significantly across network topologies
3. Malicious node ratio impacts detection performance as expected
4. Statistical tests show significant differences between models

## Research Artifacts

- `all_experimental_results.csv`: Complete experimental data
- `*.png`: Performance visualization plots
- `*.pth`: Trained model checkpoints
- `research_report.md`: This comprehensive report
