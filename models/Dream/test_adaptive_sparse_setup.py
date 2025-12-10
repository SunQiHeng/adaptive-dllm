#!/usr/bin/env python3
"""
Test script to verify adaptive sparse setup for Dream model.
This script tests imports, configuration creation, and basic functionality
without requiring actual model loading.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import torch
import numpy as np

def test_imports():
    """Test that all modules can be imported."""
    print("="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    try:
        from models.Dream.attribution.head_attribution_block_dream import (
            BlockwiseIntegratedGradientsAttributionDream
        )
        print("✓ Attribution module imported successfully")
    except Exception as e:
        print(f"✗ Attribution import failed: {e}")
        return False
    
    try:
        from models.Dream.sparse.adaptive_utils_dream import (
            create_adaptive_sparsity_config,
            print_adaptive_sparsity_summary,
            allocate_adaptive_sparsity_from_importance,
            generate_random_head_importance
        )
        print("✓ Adaptive utils module imported successfully")
    except Exception as e:
        print(f"✗ Adaptive utils import failed: {e}")
        return False
    
    try:
        from models.Dream.sparse.SparseD_utils_dream import (
            create_block_mask_cached,
            customize_mask,
            create_attention_block_mask
        )
        print("✓ Sparse utils module imported successfully")
    except Exception as e:
        print(f"✗ Sparse utils import failed: {e}")
        return False
    
    print("\nAll imports successful! ✓\n")
    return True


def test_config_creation():
    """Test adaptive sparsity config creation."""
    print("="*70)
    print("TEST 2: Adaptive Sparsity Configuration")
    print("="*70)
    
    try:
        from models.Dream.sparse.adaptive_utils_dream import (
            create_adaptive_sparsity_config,
            print_adaptive_sparsity_summary
        )
        
        # Test parameters (simulating a small model)
        n_layers = 4
        n_heads = 8
        
        print(f"\nCreating adaptive config for:")
        print(f"  Layers: {n_layers}")
        print(f"  Heads per layer: {n_heads}")
        
        # Create config with random importance
        config = create_adaptive_sparsity_config(
            n_layers=n_layers,
            n_heads=n_heads,
            strategy='uniform',
            base_sparsity=0.5,
            min_sparsity=0.1,
            max_sparsity=0.9,
            seed=42
        )
        
        print("\n✓ Config created successfully")
        
        # Verify config structure
        assert 'importance_scores' in config, "Missing importance_scores"
        assert 'sparsity_levels' in config, "Missing sparsity_levels"
        assert 'metadata' in config, "Missing metadata"
        
        print("✓ Config structure verified")
        
        # Verify dimensions
        assert len(config['importance_scores']) == n_layers, "Wrong number of layers"
        for layer_idx in range(n_layers):
            assert config['importance_scores'][layer_idx].shape[0] == n_heads, \
                f"Wrong number of heads in layer {layer_idx}"
            assert config['sparsity_levels'][layer_idx].shape[0] == n_heads, \
                f"Wrong sparsity shape in layer {layer_idx}"
        
        print("✓ Config dimensions verified")
        
        # Print summary
        print_adaptive_sparsity_summary(config)
        
        print("\nConfiguration test successful! ✓\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Config creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_importance_to_sparsity():
    """Test conversion from importance scores to sparsity levels."""
    print("="*70)
    print("TEST 3: Importance to Sparsity Conversion")
    print("="*70)
    
    try:
        from models.Dream.sparse.adaptive_utils_dream import (
            allocate_adaptive_sparsity_from_importance
        )
        
        # Create mock importance scores
        n_layers = 2
        n_heads = 4
        
        importance_scores = {
            0: torch.tensor([1.0, 0.8, 0.5, 0.2]),  # Layer 0
            1: torch.tensor([0.9, 0.7, 0.4, 0.3])   # Layer 1
        }
        
        print("\nTest importance scores:")
        for layer_idx, scores in importance_scores.items():
            print(f"  Layer {layer_idx}: {scores.tolist()}")
        
        # Convert to sparsity
        sparsity_levels = allocate_adaptive_sparsity_from_importance(
            importance_scores=importance_scores,
            base_sparsity=0.5,
            min_sparsity=0.2,
            max_sparsity=0.8,
            inverse_importance=True
        )
        
        print("\nResulting keep ratios (1 - sparsity):")
        for layer_idx, keep_ratios in sparsity_levels.items():
            print(f"  Layer {layer_idx}: {keep_ratios.tolist()}")
        
        # Verify inverse relationship
        for layer_idx in range(n_layers):
            imp = importance_scores[layer_idx]
            keep = sparsity_levels[layer_idx]
            
            # Most important head should have highest keep ratio
            most_important_idx = imp.argmax().item()
            least_important_idx = imp.argmin().item()
            
            assert keep[most_important_idx] > keep[least_important_idx], \
                f"Layer {layer_idx}: Inverse importance relationship violated"
        
        print("\n✓ Inverse importance relationship verified")
        print("\nImportance to sparsity conversion test successful! ✓\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attribution_class():
    """Test attribution class initialization (without model)."""
    print("="*70)
    print("TEST 4: Attribution Class Structure")
    print("="*70)
    
    try:
        from models.Dream.attribution.head_attribution_block_dream import (
            BlockwiseIntegratedGradientsAttributionDream
        )
        
        print("\nAttribution class structure:")
        
        # Check class methods
        methods = [
            'compute_block_attribution_for_layer',
            'get_head_ranking',
            '_compute_layer_head_att',
            '_forward_with_layer_head_cache'
        ]
        
        for method in methods:
            if hasattr(BlockwiseIntegratedGradientsAttributionDream, method):
                print(f"  ✓ Method '{method}' exists")
            else:
                print(f"  ✗ Method '{method}' missing")
                return False
        
        print("\n✓ All required methods present")
        print("\nAttribution class test successful! ✓\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Attribution class test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_classes():
    """Test model class structure (without loading weights)."""
    print("="*70)
    print("TEST 5: Model Class Structure")
    print("="*70)
    
    try:
        from models.Dream.core.adaptive_sparsed_modeling_dream import (
            AdaptiveDreamAttention,
            AdaptiveDreamDecoderLayer,
            AdaptiveDreamModel,
            AdaptiveDreamForMaskedLM
        )
        
        print("\nModel classes imported:")
        print("  ✓ AdaptiveDreamAttention")
        print("  ✓ AdaptiveDreamDecoderLayer")
        print("  ✓ AdaptiveDreamModel")
        print("  ✓ AdaptiveDreamForMaskedLM")
        
        # Check key methods
        print("\nVerifying key methods:")
        
        if hasattr(AdaptiveDreamAttention, 'set_adaptive_sparsity'):
            print("  ✓ AdaptiveDreamAttention.set_adaptive_sparsity")
        else:
            print("  ✗ Missing set_adaptive_sparsity in AdaptiveDreamAttention")
            return False
        
        if hasattr(AdaptiveDreamAttention, '_adaptive_sparse_attention'):
            print("  ✓ AdaptiveDreamAttention._adaptive_sparse_attention")
        else:
            print("  ✗ Missing _adaptive_sparse_attention in AdaptiveDreamAttention")
            return False
        
        if hasattr(AdaptiveDreamModel, 'set_adaptive_config'):
            print("  ✓ AdaptiveDreamModel.set_adaptive_config")
        else:
            print("  ✗ Missing set_adaptive_config in AdaptiveDreamModel")
            return False
        
        print("\n✓ All required methods present")
        print("\nModel class test successful! ✓\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Model class test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("DREAM ADAPTIVE SPARSE - SETUP VERIFICATION")
    print("="*70 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("Config Creation", test_config_creation()))
    results.append(("Importance to Sparsity", test_importance_to_sparsity()))
    results.append(("Attribution Class", test_attribution_class()))
    results.append(("Model Classes", test_model_classes()))
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 All tests passed! Setup is complete and working correctly.\n")
        print("Next steps:")
        print("1. Run attribution computation on your dataset")
        print("2. Use adaptive_sparsed_generate.py to generate with adaptive sparsity")
        print("\nSee README_ADAPTIVE_SPARSE.md for detailed usage instructions.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    exit(main())

