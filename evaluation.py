"""
Programma che effettua la valutazione intrinseca degli Embeddings:
viene costruito il grafo computazionale del modello e viene caricato il modello
addestrato dalla cartella di log dove è stato salvato in fase di addestramento.
Vengono caricati 3 Dataset: UMNSRS-Sim, UMNSRS-Rel e WordSim353.
Vengono valutate le coppie di termini all'interno dei dataset utilizzando la similarità
del coseno, e viene infine calcolato il coefficiente di correlazione di Spearmann tra 
i valori ottenuti e i valori assegnati da operatori umani.
"""


import math
import os
import pickle
import random
import sys
import pandas as pd

from scipy.stats.stats import spearmanr
from scipy.spatial.distance import cosine

import numpy as np
import tensorflow as tf


embedding_size = 60
num_sampled = 5
window_size = 4
subsampling_threshold = 1e-5
batch_size=500
alpha_sgd = 0.2

def read_file(filename):
    with open(filename, 'rb') as f:
        file_caricato = pickle.load(f)
    return file_caricato

#Carico i dizionari
words_to_int = read_file('./dizionari/words_to_int.pickle')
int_to_words = read_file('./dizionari/int_to_words.pickle')
words_count = read_file('./dizionari/words_count.pickle')

vocabulary_size = len(words_to_int)

#Carico i dati relativi al Dataset di relazionalità
rel_data = pd.read_csv('/mnt/4CC6A887C6A8733E/Tesi/sim_rel_dataset/UMNSRS_relatedness_mod458_word2vec.csv')
term1_rel = rel_data['Term1'].values
term2_rel = rel_data['Term2'].values
mean_rel = rel_data['Mean'].values
mean_rel_list = []
rel = []

#Carico i dati relativi al Dataset di similarità
sim_data = pd.read_csv('/mnt/4CC6A887C6A8733E/Tesi/sim_rel_dataset/UMNSRS_similarity_mod449_word2vec.csv')
term1_sim = sim_data['Term1'].values
term2_sim = sim_data['Term2'].values
mean_sim = sim_data['Mean'].values
mean_sim_list = []
sim = []

#Carico i dati relativi al Dataset WordSim353
sim353_data = pd.read_csv('/mnt/4CC6A887C6A8733E/Tesi/sim_rel_dataset/wordsim_353.csv')
term1_word_sim = sim353_data['Word1'].values
term2_word_sim = sim353_data['Word2'].values
mean_word_sim = sim353_data['mean'].values
mean_word_sim_list = []
word_sim = []

valid_examples = np.array([words_to_int['one'], words_to_int['two']])

#Costruisco il Grafo del modello Skip-Gram
graph = tf.Graph()
with graph.as_default():

    #Input
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    #Lookup degli embeddings, inizialmente gli embedding vengono inizializzati con le componenti random tra -1.0 e +1.0 da una distribuzione uniforme 
    with tf.name_scope('embeddings'):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    #Variabili per la funzione obbiettivo NCE loss:
        # nce_weights  la matrice dei pesi di dimensioni vocabulary_size * embedding_size
        # nce_biases  un vettore di zeri lungo vocabulary_size
    with tf.name_scope('weights'):
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    with tf.name_scope('biases'):
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


    #Funzione obbiettivo NCE loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.nce_loss(
                                        weights = nce_weights,
                                        biases = nce_biases,
                                        labels = train_labels,
                                        inputs = embed,
                                        num_sampled = num_sampled,
                                        num_classes = vocabulary_size))

    tf.summary.scalar('loss', loss)

    #Funzione di ottimizzazione (SGD con passo di apprendimento di alpha_sgd)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(alpha_sgd).minimize(loss)

    norm= tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    merged = tf.summary.merge_all()

    #Op che inizializza le variabili
    init = tf.global_variables_initializer()

    #Saver
    saver = tf.train.Saver()

#Recupero la sessione
with tf.Session(graph=graph) as session:
    saver.restore(session, '/mnt/4CC6A887C6A8733E/Tesi/log_25_06/model.ckpt')
    print('Modello caricato')
    
    #Calcolo i valori di similarità
    for i in range(len(term1_sim)):
        if term1_sim[i] in words_to_int and term2_sim[i] in words_to_int:
            valid_dataset = np.array([words_to_int[term1_sim[i]], words_to_int[term2_sim[i]]])
            vectors = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset).eval()
            similarity = 1 - cosine(vectors[0,:], vectors[1,:])
            sim.append(similarity)
            mean_sim_list.append(mean_sim[i])

    #Calcolo i valori di relatedness
    for i in range(len(term1_rel)):
        if term1_rel[i] in words_to_int and term2_rel[i] in words_to_int:
            valid_dataset = np.array([words_to_int[term1_rel[i]], words_to_int[term2_rel[i]]])
            vectors = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset).eval()
            relatedness = 1 - cosine(vectors[0,:], vectors[1,:])
            rel.append(relatedness)
            mean_rel_list.append(mean_rel[i])
    

    #Calcolo i valori di similarità per WordSim353
    for i in range(len(term1_word_sim)):
        if term1_word_sim[i] in words_to_int and term2_word_sim[i] in words_to_int:
            valid_dataset = np.array([words_to_int[term1_word_sim[i]], words_to_int[term2_word_sim[i]]])
            vectors = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset).eval()
            similarity = 1 - cosine(vectors[0,:], vectors[1,:])
            word_sim.append(similarity)
            mean_word_sim_list.append(mean_word_sim[i])

    spearman_sim = spearmanr(mean_sim_list, sim)
    spearman_rel = spearmanr(mean_rel_list, rel)
    spearman_word_sim = spearmanr(mean_word_sim_list, word_sim)

print('Correlazione UMNSRS-Sim:', spearman_sim)
print('Correlazione UMNSRS-Rel:', spearman_rel)
print('Correlazione WordSim353:', spearman_word_sim)   

