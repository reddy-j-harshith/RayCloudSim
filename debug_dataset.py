#!/usr/bin/env python3

from research_attack_aware_system import ResearchAttackAwareSystem

# Initialize system and check dataset info
system = ResearchAttackAwareSystem(malicious_ratio=0.3)
trainset, testset, dataset_info = system.load_dataset('pakistan', 'Tuple30K')

print("Dataset info:")
for key, value in dataset_info.items():
    print(f"  {key}: {value} (type: {type(value)})")
    if key == 'Edges' and isinstance(value, list):
        print(f"    First few edges: {value[:5]}")