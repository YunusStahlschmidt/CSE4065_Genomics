import numpy as np
import random
import copy
import pandas as pd
nucleotides = ['A', 'G', 'T', 'C']

sequences = np.random.choice(nucleotides, size=(10,490)) # generate random 10 array with 490 length from nucleotides array 
motif = np.random.choice(nucleotides, size=(1,10)) # generate 10mer motif randomly from nucleotides

def mutate_motif(motif): # functio to generate (10,4) new motif
    new_motif = copy.deepcopy(motif) # copy given motif to not effect main object
    indices = [i for i in range(10)] # list of numbers until 10
    idx = random.sample(indices, 4) # take 4 random indices to mutate in this 10mer
    for i in idx: # iterate over these randomly choosen 4 indices
        nucleotides_to_choice = ['A', 'G', 'T', 'C'] # list of nucleotides
        nucleotides_to_choice.remove(new_motif[0][i]) # removing nucleotide  to make sure this new random nucleotide is not same with old one  
        selected_nucleotide = random.choice(nucleotides_to_choice) # randomly choose from remaining nucleotides
        new_motif[0][i] = selected_nucleotide # assign new mutated nucleotide to related index
    return new_motif[0] # return mutated motif 
    
def generate_new_sequence(sequences, motif): # generate and insert mutated motifs to sequences
    new_sequence = [] # list of new sequences that mutated motifs added 
    for row in sequences: # iterateing every sequence
        mutated_motif = mutate_motif(motif) # randomly generate motif and mutate 4 nucleotide
        random_idx = random.randint(0, 490) # choose random index to insert mutation to sequnce
        new_sequence.append(row[:random_idx].tolist() + mutated_motif.tolist() + row[random_idx:].tolist()) # insert mutation to motif
    return np.array(new_sequence) # return as numpy array

generated_sequence = generate_new_sequence(sequences, motif) # related function call

print(sequences.shape)
print(motif.shape)
print(generated_sequence.shape)

pd.DataFrame(generated_sequence).to_csv("./sequence.csv", index=False) # save sequences into a csv file