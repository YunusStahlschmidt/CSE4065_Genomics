import numpy as np
import random
import copy
nucleotides = ['A', 'G', 'T', 'C']

sequences = np.random.choice(nucleotides, size=(10,490))
motif = np.random.choice(nucleotides, size=(1,10)) 
print(sequences)

def mutate_motif(motif):
    new_motif = copy.deepcopy(motif)
    indices = [i for i in range(10)]
    idx = random.sample(indices, 4)
    for i in idx:
        nucleotides_to_choice = ['A', 'G', 'T', 'C']
        nucleotides_to_choice.remove(new_motif[0][i])
        selected_nucleotide = random.choice(nucleotides_to_choice)
        new_motif[0][i] = selected_nucleotide
    return new_motif[0]
    
def generate_new_sequence(sequence, motif):
    new_sequence = []
    for row in sequences:
        mutated_motif = mutate_motif(motif)
        random_idx = random.randint(0, 490)
        new_sequence.append(row[:random_idx].tolist() + mutated_motif.tolist() + row[random_idx:].tolist()) # [random_idx]
    return np.array(new_sequence)

print(sequences.shape)
print(motif.shape)
print(generate_new_sequence(sequences, motif).shape)
