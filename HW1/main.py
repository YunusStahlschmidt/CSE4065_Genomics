from math import modf
import numpy as np
import pandas as pd
import sys
from collections import Counter
import random
import time

if len(sys.argv) != 3: # check the number of command line arguments 
    print("Wrong Arguments\n\t- 1: Path to input file\n\t- 2: k (length of the consensus string)")
    exit()

k = int(sys.argv[2]) # take k from command line
generated_sequence = pd.read_csv(sys.argv[1]).values # reading given csv file

alphabet = [ 'A', 'C', 'G', 'T' ]


def randomly_select_motifs(k):
    motif_indexes = np.random.choice(500-k, 10) # chooses k substring starting indexes from a list ranging from 0 to 500-k 
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

def score(probs, sequences, k): # score calculator for randomized motif search algorithm
    indices = np.argmax(probs, axis=1) # finding the indices of max probs
    current_motifs = np.array([sequences[z, idx:(idx+k)] for z, idx in enumerate(indices)]) # selected motifs based on probs
    s = 0   # score
    for row in current_motifs.T:
        for count in Counter(row).most_common()[1:]: # Counting the score of least common genomes
            s += count[1]      # updating score
    return current_motifs, s


def score_gibbs(probs, sequences, rsm, k): # score calculator for gibbs sampler
    idx = random.choices(list(range(probs.shape[1])), weights=probs[0])[0]  # rolling an unfair dice based on probabilities
    current_motif = sequences[0, idx:(idx+k)] # selected motif among probs
    current_motifs = np.concatenate([rsm, current_motif.reshape((1,-1))], axis=0) # adding selected motif into motifs
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

def randomized_motif_search(k):
    best_motif = (0, 9999) # keeping the best motif and the score
    while True:         
        rsm = randomly_select_motifs(k) 
        motif_profile = generate_motif_profile(rsm)
        probabilities = find_probabilities(k, motif_profile, generated_sequence)
        current_motifs, s = score(probabilities, generated_sequence, k)
        if s < best_motif[1]: # check if new score is less than previous score
            best_motif = (current_motifs, s) # update motif and score
        else:
            return '\n'.join([''.join(i) for i in best_motif[0]]), best_motif[1]

def gibbs_sampler(k, itr): 
    best_motif = (0, 9999) # keeping the best motif and the score
    score_update_counter = 0 # patient
    rsm = randomly_select_motifs(k) 
    while True:
        rmv_idx = random.choice(range(10))
        rsm = np.delete(rsm, rmv_idx, axis=0)

        motif_profile = laplace(rsm) # (n,) (1, n)             
        probabilities = find_probabilities(k, motif_profile, generated_sequence[rmv_idx].reshape((1,-1))) 

        rsm, s = score_gibbs(probabilities, generated_sequence[rmv_idx].reshape((1,-1)), rsm, k)
        if s < best_motif[1]: # check if new score is less than previous score
            best_motif = (rsm, s) # update motif and score
            score_update_counter = 0 # reset the counter
        else:
            score_update_counter += 1 # else update the patient counter by 1
            
        if score_update_counter == itr: 
            return '\n'.join([''.join(i) for i in best_motif[0]]), best_motif[1]

start = time.time()
avr_score_rand = 0
avr_score_gibbs = 0
for i in range(10):
    motif, score_ = randomized_motif_search(k)
    avr_score_rand += score_
    motif, score_ = gibbs_sampler(k, 50)
    avr_score_gibbs += score_

avr_score_rand = avr_score_rand/10
avr_score_gibbs = avr_score_gibbs/10
print(f"\nExample Motif for Randomized k={k}:\n\n{motif}\n\nAverage Score: {avr_score_rand}\n\n")
print(f"\nExample Motif for Gibbs k={k}:\n\n{motif}\n\nAverage Score: {avr_score_gibbs}\n")

end = time.time()
print(end - start)