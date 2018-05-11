import math
import os
import pickle
import random
import sys

import numpy as np
#import tensorflow as tf

embedding_size = 200
num_sampled = 50
window_size = 4
subsampling_threshold = 1e-5
batch_size=500
cartella_input=sys.argv[1]

def read_file(filename):
    with open(filename, 'rb') as f:
        file_caricato = pickle.load(f)
    return file_caricato

def get_num_samples(testo, window_size):
    lunghezza = len(testo)
    somma=0
    if lunghezza < window_size:
        somma = lunghezza * (lunghezza -1)
        return somma
    else:
        for i in range(window_size):
            somma = somma + (2*((2*window_size) - (i+1)))
        somma= somma + (lunghezza - 2*window_size)*2*window_size
    return somma

def get_total_count():
    total_count=0
    for count in words_count.values():
        total_count+=count
    return total_count

def generate_batch_from_file(filename):
    with open(filename, 'r') as f:
        words=f.read().split()
    #Elimino le parole non presenti nel dizionario
    words=[word for word in words if word in words_count]
    #Effettuo il subsampling
    p_drop = {word: 1- np.sqrt(subsampling_threshold/(words_count[word]/total_count)) for word in words}
    train_words = [word for word in words if (random.random() < (1-p_drop[word]))]
    p_drop.clear()
    words.clear()
    #Effettuo la conversione word ---> int
    train_words = [words_to_int[word] for word in train_words]
    #genero le coppie (center_word,context_word)
    center_queue=list()
    context_queue=list()
    data_index=0
    for i in range(len(train_words)):
        for j in range(1, window_size+1):
            if(i-j)>=0:
                center_queue.append(train_words[i])
                context_queue.append(train_words[i-j])
            if(i+j)<len(train_words):
                center_queue.append(train_words[i])
                context_queue.append(train_words[i+j])
    train_words.clear()
    return center_queue,context_queue

def generate_batch(batch_size):
    global center_temp
    global context_temp
    global lista_file

    center_words = np.ndarray(shape=(batch_size), dtype=np.int32)
    context_words = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    
    k=0
    flag=True

    #in precedenza center_temp non era stato svuotato e quello che rimane ci sta tutto
    if len(center_temp)>0 and len(center_temp)+k<=batch_size:
        while len(center_temp)>0:
            center_words[k] = center_temp.pop(0)
            context_words[k] = context_temp.pop(0)
            k+=1
    #in precendenza center_temp non era stato svuotato ma non ci sta tutto
    if len(center_temp)>0 and len(center_temp)+k>batch_size:
        while k<batch_size:
            center_words[k] = center_temp.pop(0)
            context_words[k] = context_temp.pop(0)
            k+=1
        return center_words, context_words, flag

    while k<batch_size:
        if len(lista_file)>0:
            center_temp,context_temp = generate_batch_from_file(cartella_input+cartella+'/'+lista_file.pop(0))
        else:
            flag=False
            center_words[k] = -1
            context_words[k] = -1
            return center_words, context_words, flag
        if len(center_temp)+k<=batch_size:
            while len(center_temp)>0:
                center_words[k] = center_temp.pop(0)
                context_words[k] = context_temp.pop(0)
                k+=1
        else:
            for i in range(batch_size-k):
                center_words[k] = center_temp.pop(0)
                context_words[k] = context_temp.pop(0)
                k+=1
            return center_words, context_words, flag
    return center_words, context_words, flag


#1 Step: carico i dizionari
words_to_int = read_file('./dizionari/words_to_int.pickle')
int_to_words = read_file('./dizionari/int_to_words.pickle')
words_count = read_file('./dizionari/words_count.pickle')

total_count = get_total_count()
vocabulary_size = len(words_to_int)

center_temp=list()
context_temp=list()


#2 Step: Costruzione modello skip-gram
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
        # nce_weights è la matrice dei pesi di dimensioni vocabulary_size * embedding_size
        # nce_biases è un vettore di zeri lungo vocabulary_size
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

    #Funzione di ottimizzazione (SGD con passo di apprendimento di 1.0)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm= tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    merged = tf.summary.merge_all()

    #Op che inizializza le variabili
    init = tf.global_variables_initializer()

    #Saver
    saver = tf.train.Saver()

#3 Step: Allenamento del modello

with tf.Session(graph=graph) as session:

    writer = tf.summary.FileWriter('./log', session.graph)
    init.run()
    print('Inizializzazione completata')
    average_loss = 0
    step = 0
    lista_cartelle=sorted(os.listdir(path=cartella_input))

    for cartella in lista_cartelle:
        lista_file=sorted(os.listdir(path=cartella_input+cartella))
        while len(lista_file)>0:
            center,context,flag = generate_batch(batch_size)
            if not flag:
                j=0
                while center[j] != -1:
                    center_temp.append(center[j])
                    context_temp.append(context[j])
                    j+=1
                continue
            step+=1
            feed_dict = {train_inputs:center, train_labels:context}

            run_metadata = tf.RunMetadata()

            _, summary, loss_val = session.run([optimizer, merged, loss], feed_dict=feed_dict, run_metadata=run_metadata)
            average_loss += loss_val
    
            writer.add_summary(summary, step)

    #generiamo l'ultimo batch utilizzando quello che resta nelle liste center_temp e context_temp
    lunghezza=len(center_temp)
    if(lunghezza > 0):
        center=np.ndarray(shape=(batch_size), dtype=np.int32)
        context=np.ndarray(shape=(batch_size,1), dtype=np.int32)
        for k in range(batch_size):
            center[k]=center_temp[k%lunghezza]
            context[k]=context_temp[k%lunghezza]
        step+=1
        feed_dict = {train_inputs:center, train_labels:context}
        
        run_metadata = tf.RunMetadata()

        _,summary,loss_val = session.run([optimizer, merged, loss], feed_dict=feed_dict, run_metadata=run_metadata)
        writer.add_summary(summary,step)
    
    print('Sono stati generati', step, 'batch per l\'allenamento del modello')

    final_embeddings = normalized_embeddings.eval()


    with open('./log/metadata.tsv', 'w') as f:
        for i in range(vocabulary_size):
            f.write(int_to_words[i] + '\n')
    
    saver.save(session, './log/model.ckpt')
    print('Modello salvato nella cartella /log')

    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(writer, config)
    
    writer.close()
