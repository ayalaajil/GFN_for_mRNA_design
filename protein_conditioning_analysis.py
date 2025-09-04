#!/usr/bin/env python3
"""
Analysis framework for protein sequence conditioning effectiveness.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from main_conditional import encode_protein_sequence

class ProteinConditioningAnalyzer:
    """Analyzes the effectiveness of protein sequence conditioning."""

    def __init__(self):
        self.results = {}

    def analyze_encoding_diversity(self, protein_sequences: List[str]) -> Dict:
        """Analyze how diverse the protein encodings are."""
        print("Analyzing protein encoding diversity...")

        device = torch.device("cpu")
        encodings = []

        for seq in protein_sequences:
            encoded = encode_protein_sequence(seq, device)
            encodings.append(encoded)

        encodings_tensor = torch.stack(encodings)

        # Calculate pairwise distances
        distances = torch.cdist(encodings_tensor, encodings_tensor, p=2)

        # Remove diagonal (self-distances)
        mask = torch.eye(len(encodings), dtype=torch.bool)
        distances = distances[~mask].view(len(encodings), len(encodings) - 1)

        analysis = {
            'mean_distance': float(distances.mean()),
            'std_distance': float(distances.std()),
            'min_distance': float(distances.min()),
            'max_distance': float(distances.max()),
            'encoding_variance': float(encodings_tensor.var(dim=0).mean())
        }

        print(f"Mean pairwise distance: {analysis['mean_distance']:.4f}")
        print(f"Encoding variance: {analysis['encoding_variance']:.4f}")

        return analysis

    def compare_conditioning_approaches(self, protein_seq: str, weights: List[float]) -> Dict:
        """Compare different conditioning approaches."""
        device = torch.device("cpu")

        # Approach 1: Weights only
        weights_only = torch.tensor(weights, dtype=torch.float32, device=device)

        # Approach 2: Weights + protein features
        protein_features = encode_protein_sequence(protein_seq, device)
        weights_plus_protein = torch.cat([weights_only, protein_features])

        # Approach 3: Protein features only
        protein_only = protein_features

        return {
            'weights_only': weights_only,
            'weights_plus_protein': weights_plus_protein,
            'protein_only': protein_only,
            'protein_features_shape': protein_features.shape
        }

    def suggest_improvements(self, analysis_results: Dict) -> List[str]:
        """Suggest improvements based on analysis."""
        suggestions = []

        if analysis_results.get('encoding_variance', 0) < 0.1:
            suggestions.append("Low encoding variance - consider more diverse protein sequences")

        if analysis_results.get('mean_distance', 0) < 1.0:
            suggestions.append("Low pairwise distances - encodings might be too similar")

        suggestions.extend([
            "Consider using learned embeddings instead of one-hot encoding",
            "Add protein length normalization",
            "Include amino acid composition features",
            "Test with multiple protein sequences during training"
        ])

        return suggestions

def main():
    """Main analysis function."""
    analyzer = ProteinConditioningAnalyzer()

    # Test with different protein sequences
    test_proteins = [
        "MDSEVQRDGRILDLIDDAWREDKLPYEDVAIPLNELPEPEQDNGGTTESVKEQEMKWTDLALQYLHENVPPIGN*",
        "MTSMTQSLREVIKAMTKARNFERVLGKITLVSAAPGKVICEMKVEEEHTNAIGTLHGGLTATLVDNISTMALLCTERGAPGVSVDMNITYMSPAKLGEDIVITAHVLKQGKTLAFTSVDLTNKATGKLIAQGRHTKHLGN*",
        "MFV*",  # Short sequence
        "A" * 50 + "*",  # Repetitive sequence
    ]

    print("=== Protein Conditioning Analysis ===\n")

    # Analyze diversity
    diversity_analysis = analyzer.analyze_encoding_diversity(test_proteins)

    # Compare approaches
    print("\n=== Conditioning Approach Comparison ===")
    comparison = analyzer.compare_conditioning_approaches(
        test_proteins[0], [0.3, 0.3, 0.4]
    )

    for approach, tensor in comparison.items():
        if isinstance(tensor, torch.Tensor):
            print(f"{approach}: shape {tensor.shape}")

    # Get suggestions
    suggestions = analyzer.suggest_improvements(diversity_analysis)

    print("\n=== Recommendations ===")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")

if __name__ == "__main__":
    main()
