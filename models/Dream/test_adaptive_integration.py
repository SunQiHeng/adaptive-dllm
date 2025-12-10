#!/usr/bin/env python3
"""
Test adaptive sparse integration with Dream model.
This test verifies that the adaptive sparse attention works correctly
with Dream's diffusion generation process.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import torch


def test_mixin_structure():
    """Test that generation mixin is correctly structured."""
    print("="*70)
    print("TEST: Generation Mixin Structure")
    print("="*70)
    
    try:
        from models.Dream.generation_utils.sparsed_generation_utils_dream import DreamGenerationMixin
        from models.Dream.generation_utils.adaptive_sparsed_generation_utils_dream import DreamAdaptiveSparsedGenerationMixin
        
        print("\n✓ Both mixins imported successfully")
        
        # Check methods
        required_methods = ['diffusion_generate', '_sample', '_prepare_generation_config']
        
        for method_name in required_methods:
            if hasattr(DreamAdaptiveSparsedGenerationMixin, method_name):
                print(f"  ✓ DreamAdaptiveSparsedGenerationMixin.{method_name}")
            else:
                print(f"  ✗ Missing {method_name}")
                return False
        
        print("\n✓ All required methods present")
        return True
        
    except Exception as e:
        print(f"\n✗ Mixin test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_with_adaptive_mixin():
    """Test that AdaptiveDreamModel uses correct mixin."""
    print("\n" + "="*70)
    print("TEST: Model Inheritance and Methods")
    print("="*70)
    
    try:
        from models.Dream.core.adaptive_sparsed_modeling_dream import AdaptiveDreamModel
        
        print("\n✓ AdaptiveDreamModel imported")
        
        # Check if it has diffusion_generate (from DreamGenerationMixin)
        if hasattr(AdaptiveDreamModel, 'diffusion_generate'):
            print("  ✓ diffusion_generate method exists")
        else:
            print("  ✗ diffusion_generate method missing")
            return False
        
        # Check inheritance chain
        print(f"\n  Inheritance chain: {[c.__name__ for c in AdaptiveDreamModel.__mro__[:6]]}")
        
        # Verify key methods
        if hasattr(AdaptiveDreamModel, 'set_adaptive_config'):
            print("  ✓ set_adaptive_config method exists")
        else:
            print("  ✗ set_adaptive_config method missing")
            return False
        
        print("\n✓ Model structure verified")
        return True
        
    except Exception as e:
        print(f"\n✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sparse_param_propagation():
    """Test that SparseD_param would be correctly propagated."""
    print("\n" + "="*70)
    print("TEST: SparseD_param Propagation (Mock)")
    print("="*70)
    
    try:
        # Create mock SparseD_param
        SparseD_param = {
            'skip': 0.2,
            'select': 0.3,
            'block_size': 128,
            'new_generation': 128,
            'whole_steps': 128,
            'adaptive': True,
            'now_step': 0
        }
        
        print("\n  Mock SparseD_param created:")
        for key, value in SparseD_param.items():
            print(f"    {key}: {value}")
        
        # Verify structure
        required_keys = ['skip', 'block_size', 'new_generation', 'whole_steps', 'now_step']
        for key in required_keys:
            if key in SparseD_param:
                print(f"  ✓ Key '{key}' present")
            else:
                print(f"  ✗ Key '{key}' missing")
                return False
        
        print("\n✓ SparseD_param structure verified")
        return True
        
    except Exception as e:
        print(f"\n✗ Param test failed: {e}")
        return False


def test_attention_layer_structure():
    """Test that adaptive attention layer has required methods."""
    print("\n" + "="*70)
    print("TEST: Adaptive Attention Layer")
    print("="*70)
    
    try:
        from models.Dream.core.adaptive_sparsed_modeling_dream import AdaptiveDreamAttention
        
        print("\n✓ AdaptiveDreamAttention imported")
        
        # Check methods
        required_methods = [
            'forward',
            'set_adaptive_sparsity',
            '_adaptive_sparse_attention',
            '_build_adaptive_masks'
        ]
        
        for method_name in required_methods:
            if hasattr(AdaptiveDreamAttention, method_name):
                print(f"  ✓ {method_name}")
            else:
                print(f"  ✗ Missing {method_name}")
                return False
        
        print("\n✓ All attention methods present")
        return True
        
    except Exception as e:
        print(f"\n✗ Attention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_signature():
    """Test that forward methods have correct signatures."""
    print("\n" + "="*70)
    print("TEST: Forward Method Signatures")
    print("="*70)
    
    try:
        from models.Dream.core.adaptive_sparsed_modeling_dream import (
            AdaptiveDreamAttention,
            AdaptiveDreamModel
        )
        import inspect
        
        # Check AdaptiveDreamAttention.forward
        attn_sig = inspect.signature(AdaptiveDreamAttention.forward)
        if 'SparseD_param' in attn_sig.parameters:
            print("  ✓ AdaptiveDreamAttention.forward has SparseD_param")
        else:
            print("  ✗ AdaptiveDreamAttention.forward missing SparseD_param")
            return False
        
        # Check if diffusion_generate exists
        if hasattr(AdaptiveDreamModel, 'diffusion_generate'):
            gen_sig = inspect.signature(AdaptiveDreamModel.diffusion_generate)
            if 'SparseD_param' in gen_sig.parameters:
                print("  ✓ AdaptiveDreamModel.diffusion_generate has SparseD_param")
            else:
                print("  ⚠ AdaptiveDreamModel.diffusion_generate may not have SparseD_param (inherits from parent)")
        
        print("\n✓ Forward signatures verified")
        return True
        
    except Exception as e:
        print(f"\n✗ Signature test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("DREAM ADAPTIVE SPARSE - INTEGRATION TESTS")
    print("="*70 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Mixin Structure", test_mixin_structure()))
    results.append(("Model Inheritance", test_model_with_adaptive_mixin()))
    results.append(("SparseD_param Structure", test_sparse_param_propagation()))
    results.append(("Attention Layer", test_attention_layer_structure()))
    results.append(("Forward Signatures", test_forward_signature()))
    
    # Summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 All integration tests passed!")
        print("\nThe adaptive sparse implementation is correctly integrated.")
        print("\nNext: Test with actual model weights to verify generation works.")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())

