import pandas as pd
import numpy as np

''' DataLoader for Curriculum Learning --- Protein sequences are sampled from the training datasets based on the length range.'''

class ProteinDatasetLoader:
    """Loads and manages protein sequences from datasets for curriculum learning"""

    def __init__(self):
        self.protein_cache = {}
        self.length_index = {}
        self._load_datasets()

    def _load_datasets(self):
        """Load protein sequences from available training datasets"""

        dataset_files = [
            'training_dataset_very_short.csv',
            'training_dataset_short.csv',
            'training_dataset_medium.csv',
            'training_dataset_long.csv',
            'training_dataset_very_long.csv',
        ]

        all_proteins = []

        for dataset_file in dataset_files:
            try:
                df = pd.read_csv(dataset_file)
                if 'protein_sequence' in df.columns and 'protein_length' in df.columns:

                    valid_proteins = df[
                        (df['protein_sequence'].str.len() >= 10) &
                        (df['protein_sequence'].str.match(r'^[ACDEFGHIKLMNPQRSTVWY\*]+$'))
                    ]

                    for _, row in valid_proteins.iterrows():
                        protein_seq = row['protein_sequence']
                        length = len(protein_seq)

                        if length not in self.length_index:
                            self.length_index[length] = []
                        self.length_index[length].append(protein_seq)
                        all_proteins.append(protein_seq)

                    print(f"Loaded {len(valid_proteins)} proteins from {dataset_file}")

            except FileNotFoundError:
                print(f"Warning: {dataset_file} not found, skipping...")
                continue
            except Exception as e:
                print(f"Warning: Error loading {dataset_file}: {e}")
                continue

        self.protein_cache = {seq: len(seq) for seq in all_proteins}

        print(f"Total proteins loaded: {len(all_proteins)}")
        print(f"Length range: {min(self.length_index.keys()) if self.length_index else 0} - {max(self.length_index.keys()) if self.length_index else 0}")

    def _is_valid_protein_sequence(self, protein_seq):
        """Check if protein sequence is valid for the environment"""
        # Check if all amino acids are in the codon table
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY*")
        return all(aa in valid_aas for aa in protein_seq) and len(protein_seq) > 0

    def sample_protein_by_length_range(self, min_length, max_length):
        """Sample a real protein sequence within the specified length range (inclusive)"""

        # Find all proteins within the length range
        valid_proteins = []
        for length in range(min_length, max_length + 1):
            if length in self.length_index:
                proteins_at_length = [p for p in self.length_index[length] if self._is_valid_protein_sequence(p)]
                valid_proteins.extend(proteins_at_length)

        if valid_proteins:
            return np.random.choice(valid_proteins)

        # If no proteins in range, find the closest available length
        available_lengths = sorted(self.length_index.keys())
        if not available_lengths:
            raise ValueError("No protein data available in the dataset")

        # Find closest length to the range
        closest_length = None
        min_distance = float('inf')

        for length in available_lengths:
            if length < min_length:
                distance = min_length - length
            elif length > max_length:
                distance = length - max_length
            else:
                distance = 0  # Within range

            if distance < min_distance:
                min_distance = distance
                closest_length = length

        if closest_length is not None:
            proteins_at_closest = [p for p in self.length_index[closest_length] if self._is_valid_protein_sequence(p)]
            if proteins_at_closest:
                print(f"Warning: No proteins in range [{min_length}, {max_length}], using closest length {closest_length}")
                return np.random.choice(proteins_at_closest)

        raise ValueError(f"No valid proteins found in range [{min_length}, {max_length}] or nearby lengths")


    def sample_protein_by_length(self, target_length, tolerance=5):
        """Sample a protein sequence of approximately the target length (legacy method)"""
        return self.sample_protein_by_length_range(
            max(1, target_length - tolerance),
            target_length + tolerance
        )

    def get_available_lengths(self):
        """Get list of available protein lengths in the dataset"""
        return sorted(self.length_index.keys())

    def get_length_distribution(self):
        """Get distribution of protein lengths"""
        return {length: len(proteins) for length, proteins in self.length_index.items()}

    def generate_protein_sequence(self, task):
        """Generate a protein sequence for the given task using real protein datasets"""
        if isinstance(task, dict):
            # Complexity-based task
            return self._generate_complexity_based_protein(task)
        elif isinstance(task, (list, tuple)) and len(task) == 2:
            min_length, max_length = task

            print(f"Generating protein sequence for length range {min_length}-{max_length}")
            return self.sample_protein_by_length_range(min_length, max_length)
        else:
            # Single length task
            return self.sample_protein_by_length(task)
