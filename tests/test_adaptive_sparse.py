"""
Test script for Adaptive Sparse Attention implementation.

This script tests the basic functionality of the adaptive sparse attention system
without requiring a full model download.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from models.LLaDA.sparse.adaptive_utils import (
    generate_random_head_importance,
    allocate_adaptive_sparsity_from_importance,
    create_adaptive_sparsity_config,
    print_importance_summary,
    print_adaptive_sparsity_summary
)


def test_random_importance_generation():
    """Test random head importance generation."""
    print("\n" + "="*70)
    print("TEST 1: Random Head Importance Generation")
    print("="*70)
    
    n_layers = 4
    n_heads = 8
    
    # Test different distributions
    for dist in ['uniform', 'normal', 'exponential']:
        print(f"\nTesting {dist} distribution...")
        importance = generate_random_head_importance(
            n_layers=n_layers,
            n_heads=n_heads,
            distribution=dist,
            seed=42
        )
        
        assert len(importance) == n_layers, f"Expected {n_layers} layers, got {len(importance)}"
        assert all(scores.shape[0] == n_heads for scores in importance.values()), \
            f"Expected {n_heads} heads per layer"
        
        print(f"✓ {dist} distribution: Success")
        print(f"  Sample (Layer 0): {importance[0][:4]}...")
    
    print("\n✓ All distributions work correctly")


def test_sparsity_allocation():
    """Test sparsity allocation from importance scores."""
    print("\n" + "="*70)
    print("TEST 2: Sparsity Allocation")
    print("="*70)
    
    n_layers = 4
    n_heads = 8
    
    # Generate importance
    importance = generate_random_head_importance(
        n_layers=n_layers,
        n_heads=n_heads,
        distribution='uniform',
        seed=42
    )
    
    # Test with different sparsity ranges
    test_configs = [
        {'base': 0.5, 'min': 0.1, 'max': 0.9},
        {'base': 0.3, 'min': 0.2, 'max': 0.7},
        {'base': 0.7, 'min': 0.5, 'max': 0.95},
    ]
    
    for config in test_configs:
        print(f"\nTesting sparsity range: base={config['base']}, [{config['min']}, {config['max']}]")
        
        sparsity = allocate_adaptive_sparsity_from_importance(
            importance_scores=importance,
            base_sparsity=config['base'],
            min_sparsity=config['min'],
            max_sparsity=config['max']
        )
        
        # Verify sparsity is within bounds
        for layer_idx in range(n_layers):
            keep_ratios = sparsity[layer_idx]
            expected_min_keep = 1.0 - config['max']
            expected_max_keep = 1.0 - config['min']
            
            assert (keep_ratios >= expected_min_keep - 1e-6).all(), \
                f"Some keep ratios below minimum"
            assert (keep_ratios <= expected_max_keep + 1e-6).all(), \
                f"Some keep ratios above maximum"
        
        print(f"✓ Sparsity allocation correct")
        print(f"  Sample (Layer 0): {sparsity[0][:4]}...")
    
    print("\n✓ All sparsity allocations work correctly")


def test_adaptive_config_creation():
    """Test complete adaptive configuration creation."""
    print("\n" + "="*70)
    print("TEST 3: Adaptive Configuration Creation")
    print("="*70)
    
    n_layers = 4
    n_heads = 8
    
    config = create_adaptive_sparsity_config(
        n_layers=n_layers,
        n_heads=n_heads,
        strategy='uniform',
        base_sparsity=0.5,
        seed=42
    )
    
    # Verify config structure
    assert 'importance_scores' in config, "Missing importance_scores"
    assert 'sparsity_levels' in config, "Missing sparsity_levels"
    assert 'metadata' in config, "Missing metadata"
    
    # Verify metadata
    metadata = config['metadata']
    assert metadata['n_layers'] == n_layers, "Incorrect n_layers in metadata"
    assert metadata['n_heads'] == n_heads, "Incorrect n_heads in metadata"
    assert metadata['base_sparsity'] == 0.5, "Incorrect base_sparsity in metadata"
    
    print("✓ Configuration structure correct")
    print(f"  Layers: {metadata['n_layers']}")
    print(f"  Heads: {metadata['n_heads']}")
    print(f"  Strategy: {metadata['strategy']}")
    
    # Test with verbose output
    print("\nConfiguration summary:")
    print_adaptive_sparsity_summary(config)
    
    print("\n✓ Configuration creation successful")


def test_gqa_compatibility():
    """Test GQA (Grouped Query Attention) compatibility."""
    print("\n" + "="*70)
    print("TEST 4: GQA Compatibility")
    print("="*70)
    
    n_layers = 4
    n_kv_heads = 8  # GQA: fewer KV heads
    n_query_heads = 32  # More query heads
    
    print(f"\nGQA Configuration:")
    print(f"  Query heads: {n_query_heads}")
    print(f"  KV heads: {n_kv_heads}")
    print(f"  Heads per group: {n_query_heads // n_kv_heads}")
    
    # Generate importance for KV heads
    importance = generate_random_head_importance(
        n_layers=n_layers,
        n_heads=n_kv_heads,  # Use KV head count
        distribution='uniform',
        seed=42
    )
    
    # Allocate sparsity
    sparsity = allocate_adaptive_sparsity_from_importance(
        importance_scores=importance,
        base_sparsity=0.5
    )
    
    # Verify dimensions
    for layer_idx in range(n_layers):
        assert importance[layer_idx].shape[0] == n_kv_heads, \
            f"Layer {layer_idx}: incorrect importance shape"
        assert sparsity[layer_idx].shape[0] == n_kv_heads, \
            f"Layer {layer_idx}: incorrect sparsity shape"
    
    print("✓ GQA dimensions correct")
    print(f"  Importance shape per layer: ({n_kv_heads},)")
    print(f"  Sparsity shape per layer: ({n_kv_heads},)")
    
    print("\n✓ GQA compatibility verified")


def test_importance_increases_keep_logic():
    """Test that importance_increases_keep works correctly (higher importance -> higher keep ratio)."""
    print("\n" + "="*70)
    print("TEST 5: Inverse Importance Logic")
    print("="*70)
    
    # Create controlled importance scores
    importance = {
        0: torch.tensor([1.0, 0.5, 0.1])  # High, medium, low importance
    }
    
    # Test importance_increases_keep (default: True)
    sparsity_inverse = allocate_adaptive_sparsity_from_importance(
        importance_scores=importance,
        base_sparsity=0.5,
        min_sparsity=0.2,
        max_sparsity=0.8,
        importance_increases_keep=True
    )
    
    keep_ratios = sparsity_inverse[0]
    
    # With importance_increases_keep=True:
    # High importance (1.0) -> Low sparsity -> High keep ratio
    # Low importance (0.1) -> High sparsity -> Low keep ratio
    assert keep_ratios[0] > keep_ratios[2], \
        "High importance should have higher keep ratio than low importance"
    
    print("✓ importance_increases_keep logic correct")
    print(f"  Importance: {importance[0].tolist()}")
    print(f"  Keep ratios: {keep_ratios.tolist()}")
    print(f"  High importance -> High keep ratio: {keep_ratios[0]:.3f}")
    print(f"  Low importance -> Low keep ratio: {keep_ratios[2]:.3f}")
    
    print("\n✓ importance_increases_keep verified")


def test_determinism():
    """Test that results are deterministic with same seed."""
    print("\n" + "="*70)
    print("TEST 6: Determinism")
    print("="*70)
    
    n_layers = 4
    n_heads = 8
    
    # Generate twice with same seed
    config1 = create_adaptive_sparsity_config(
        n_layers=n_layers,
        n_heads=n_heads,
        strategy='uniform',
        base_sparsity=0.5,
        seed=42
    )
    
    config2 = create_adaptive_sparsity_config(
        n_layers=n_layers,
        n_heads=n_heads,
        strategy='uniform',
        base_sparsity=0.5,
        seed=42
    )
    
    # Compare results
    for layer_idx in range(n_layers):
        imp1 = config1['importance_scores'][layer_idx]
        imp2 = config2['importance_scores'][layer_idx]
        assert torch.allclose(imp1, imp2), f"Layer {layer_idx}: importance not deterministic"
        
        sp1 = config1['sparsity_levels'][layer_idx]
        sp2 = config2['sparsity_levels'][layer_idx]
        assert torch.allclose(sp1, sp2), f"Layer {layer_idx}: sparsity not deterministic"
    
    print("✓ Results are deterministic with same seed")
    
    # Generate with different seed
    config3 = create_adaptive_sparsity_config(
        n_layers=n_layers,
        n_heads=n_heads,
        strategy='uniform',
        base_sparsity=0.5,
        seed=123
    )
    
    # Verify they are different
    different = False
    for layer_idx in range(n_layers):
        imp1 = config1['importance_scores'][layer_idx]
        imp3 = config3['importance_scores'][layer_idx]
        if not torch.allclose(imp1, imp3):
            different = True
            break
    
    assert different, "Different seeds should produce different results"
    print("✓ Different seeds produce different results")
    
    print("\n✓ Determinism verified")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("ADAPTIVE SPARSE ATTENTION TEST SUITE")
    print("="*70)
    
    tests = [
        test_random_importance_generation,
        test_sparsity_allocation,
        test_adaptive_config_creation,
        test_gqa_compatibility,
        test_importance_increases_keep_logic,
        test_determinism,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test.__name__}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
    
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

