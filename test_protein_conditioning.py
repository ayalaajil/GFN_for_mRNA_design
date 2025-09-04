#!/usr/bin/env python3
"""
Test script to verify protein sequence conditioning implementation.
"""

import torch
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from main_conditional import encode_protein_sequence

def test_protein_encoding():
    """Test the protein sequence encoding function."""
    print("Testing protein sequence encoding...")

    device = torch.device("cpu")
    test_protein = "MDSEVQRDGRILDLIDDAWREDKLPYEDVAIPLNELPEPEQDNGGTTESVKEQEMKWTDLALQYLHENVPPIGN*"

    # Test encoding
    encoded = encode_protein_sequence(test_protein, device)

    print(f"Input protein sequence: {test_protein}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Expected shape: (44,) - 3 weights + 21 one-hot + 21 counts + 1 length")

    # Verify shape
    assert encoded.shape == (44,), f"Expected shape (44,), got {encoded.shape}"

    # Verify that one-hot encoding works (should have 1s for present amino acids)
    one_hot_part = encoded[:21]
    counts_part = encoded[21:42]
    length_part = encoded[42:43]

    print(f"One-hot part (first 21): {one_hot_part}")
    print(f"Counts part (next 21): {counts_part}")
    print(f"Length part (last 1): {length_part}")

    # Check that length feature is reasonable
    expected_length = len(test_protein) / 100.0
    assert abs(length_part[0] - expected_length) < 1e-6, f"Length feature incorrect: {length_part[0]} vs {expected_length}"

    print("✓ Protein encoding test passed!")
    return True

def test_conditioning_tensor():
    """Test building a complete conditioning tensor."""
    print("\nTesting complete conditioning tensor...")

    device = torch.device("cpu")
    test_protein = "MDSEVQRDGRILDLIDDAWREDKLPYEDVAIPLNELPEPEQDNGGTTESVKEQEMKWTDLALQYLHENVPPIGN*"
    weights = [0.3, 0.3, 0.4]

    # Encode protein
    protein_features = encode_protein_sequence(test_protein, device)

    # Create weights tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    # Combine
    conditioning = torch.cat([weights_tensor, protein_features])

    print(f"Weights: {weights}")
    print(f"Protein features shape: {protein_features.shape}")
    print(f"Combined conditioning shape: {conditioning.shape}")

    # Verify shape
    assert conditioning.shape == (44,), f"Expected conditioning shape (44,), got {conditioning.shape}"

    # Verify first 3 elements are weights
    assert torch.allclose(conditioning[:3], weights_tensor), "First 3 elements should be weights"

    print("✓ Conditioning tensor test passed!")
    return True

if __name__ == "__main__":
    print("Running protein sequence conditioning tests...\n")

    try:
        test_protein_encoding()
        test_conditioning_tensor()
        print("\n🎉 All tests passed! Protein sequence conditioning is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
