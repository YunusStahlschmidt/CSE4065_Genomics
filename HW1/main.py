import numpy as np
import pandas as pd
import sys
from collections import Counter
import random

# if len(sys.argv) != 3:
#     print("Wrong Arguments\n\t- 1: Path to input file\n\t- 2: k (length of the consensus string)")
#     exit()

# generated_sequence = sys.argv[1]
# k = sys.argv[2]


alphabet = [ 'A', 'C', 'G', 'T' ]
generated_sequence = pd.read_csv("./HW1/sequence.csv").values
k = 10

def randomly_select_motifs(k):
    motif_indexes = np.random.choice(500-k, k) # chooses k substring starting indexes from a list ranging from 0 to 500-k 
    motifs = [] # list which will hold motifs matrix
    for index, sequence in enumerate(generated_sequence): # go over each sequence in the input
        selected_motif = sequence[motif_indexes[index]:(motif_indexes[index]+k)] # select substring according to randomly chosen substring starting index
        motifs.append(selected_motif) # add selected substring to motifs list
    return np.array(motifs) # return array of motifs

def generate_motif_profile(motifs):
    profile = {'A':[], 'C':[], 'G':[], 'T':[]} # to keep profile matrix
    for row in motifs.T: # instead of finding the counts of each nucleotide column-wise, we are looking row-wise
        counter = Counter(row) # finding the counts of each nucleotide
        for nucleotide in profile: # iterating over nucleotide counts
            profile[nucleotide].append(counter[nucleotide]/10) # dividing each count by row count (10) to find probability
    return profile # return the profiles as dictionary

def find_probabilities( k, profile, sequences ):
    total_probs = [] # list of probablitios for each row
    for seq in sequences: # iterating over every given sequence
        row_prob = [] # list to store every k len sub seq prob based on index
        for i in range(500-k+1): # iterating through every sub seq to find prob
            sub_prob = 1 # initial prob
            for j in range(k): # iterate to update probability of the occurence of that letter
                sub_prob *= profile[seq[i+j]][j] # setting prob based on that letters prob at that index that comes from profile matrix
            row_prob.append(sub_prob) # add to row list 
        total_probs.append(row_prob) # adding row prop to total
    return np.array(total_probs) # return final probs as numpy array

def score(probs, sequences, k):
    indices = np.argmax(probs, axis=1)
    current_motifs = np.array([sequences[i, idx:(idx+k)] for i, idx in enumerate(indices)])
    s = 0
    for row in current_motifs.T:
        for count in Counter(row).most_common()[1:]:
            s += count[1]
    return current_motifs, s

def score_gibbs(probs, sequences, rsm, k):
    idx = random.choices(list(range(probs.shape[1])), weights=probs[0])[0]
    current_motif = sequences[0, idx:(idx+k)]
    current_motifs = np.concatenate([rsm, current_motif.reshape((1,-1))], axis=0)
    s = 0
    for row in current_motifs.T:
        for count in Counter(row).most_common()[1:]:
            s += count[1]
    return current_motifs, s

def laplace(motifs):
    profile = {'A':[], 'C':[], 'G':[], 'T':[]} # to keep profile matrix
    for row in motifs.T: # instead of finding the counts of each nucleotide column-wise, we are looking row-wise
        counter = Counter(row) # finding the counts of each nucleotide
        for nucleotide in profile: # iterating over nucleotide counts
            profile[nucleotide].append((counter[nucleotide]+1)/13) # dividing each count by row count (10) to find probability

    return profile # return the profiles as dictionary

def randomized_motif_search(k, itr):
    best_motif = (0, 9999)
    score_update_counter = 0
    while True:
        rsm = randomly_select_motifs(k)
        motif_profile = generate_motif_profile(rsm)
        probabilities = find_probabilities(k, motif_profile, generated_sequence)
        current_motifs, s = score(probabilities, generated_sequence, k)
        if s < best_motif[1]:
            best_motif = (current_motifs, s)
            score_update_counter = 0    
        else:
            score_update_counter += 1
            
        if score_update_counter == itr:
            return '\n'.join([''.join(i) for i in best_motif[0]])

def gibbs_sampler(k, itr):
    best_motif = (0, 9999)
    score_update_counter = 0
    rsm = randomly_select_motifs(k)
    while True:
        rmv_idx = random.choice(range(10))
        removed_motif = rsm.tolist().pop(rmv_idx)

        motif_profile = laplace(rsm) # (n,) (1, n)             
        probabilities = find_probabilities(k, motif_profile, generated_sequence[rmv_idx].reshape((1,-1))) # output ?

        rsm, s = score_gibbs(probabilities, generated_sequence[rmv_idx].reshape((1,-1)), rsm, k)
        if s < best_motif[1]:
            best_motif = (rsm, s)
            score_update_counter = 0    
        else:
            score_update_counter += 1
            
        if score_update_counter == itr:
            return '\n'.join([''.join(i) for i in best_motif[0]])

print(gibbs_sampler(10, 50))