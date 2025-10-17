#!/usr/bin/env python3
"""
Test Research System - Single Dataset
"""

import sys
sys.path.append('.')

from research_attack_aware_system import ResearchAttackAwareSystem

def test_single_dataset():
    """Test the system on a single small dataset"""
    print("🧪 Testing Research System on Single Dataset")
    
    # Initialize system
    system = ResearchAttackAwareSystem(malicious_ratio=0.25)
    
    # Test on smallest dataset
    try:
        result = system.run_single_dataset_study('pakistan', 'Tuple30K')
        print("✅ Single dataset study completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_dataset()
    if success:
        print("\n🎉 Test successful!")
    else:
        print("\n❌ Test failed!")