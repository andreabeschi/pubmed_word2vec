import math
import os
import pickle
import random
import collections
import sys

import numpy as np
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

embedding_size = 200
num_sampled = 20
window_size = 5
subsampling_threshold = 1e-5
batch_size=400
cartella_input=sys.argv[1]
cartella_log=sys.argv[2]

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
    center_queue=collections.deque()
    context_queue=collections.deque()
    with open(filename, 'r') as f:
        for line in f:
            words=line.split()
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

    center_words = np.ndarray(shape=(batch_size), dtype=np.int32)
    context_words = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    
    k=0

    #center_temp contiene qualcosa avanzato dal batch_precedente
    while len(center_temp)>0:
        center_words[k] = center_temp.popleft()
        context_words[k] = context_temp.popleft()
        k+=1

    for i in range(k, batch_size):
        center_words[i] = center_queue.popleft()
        context_words[i] = context_queue.popleft()
        k+=1

    return center_words, context_words

#1 Step: carico i dizionari
words_to_int = read_file('./dizionari/words_to_int.pickle')
int_to_words = read_file('./dizionari/int_to_words.pickle')
words_count = read_file('./dizionari/words_count.pickle')

total_count = get_total_count()
vocabulary_size = len(words_to_int)

center_temp=collections.deque()
context_temp=collections.deque()

valid_size = 8
valid_examples = np.array([words_to_int['the'], words_to_int['be'], words_to_int['biopsy'], words_to_int['potassium'], words_to_int['aortic'], words_to_int['maternal'], words_to_int['white'], words_to_int['eight']])

#valid_window = 100
#valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
#valid_examples = np.append(valid_examples, random.sample(range(1000,1000+valid_window), valid_size//2))
#valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


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

    #Funzione di ottimizzazione (SGD con passo di apprendimento di 0.05)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

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

    writer = tf.summary.FileWriter(cartella_log, session.graph)
    init.run()
    print('Inizializzazione completata')
    average_loss = 0
    step = 0
    lista_file=sorted(os.listdir(path=cartella_input))

    for doc in lista_file:
        #1 Step: genero il batch totale dal file
        center_queue, context_queue = generate_batch_from_file(cartella_input+doc)
        print('Generato il batch relativo al file', doc)
        #2 Step: genero i possibili batch e effettuo il passo di allenamento
        dimensione_batch_totale=len(center_queue)+len(center_temp)
        for i in range(dimensione_batch_totale//batch_size):
            center, context = generate_batch(batch_size)
            step+=1
            feed_dict = {train_inputs:center, train_labels:context}
            run_metadata = tf.RunMetadata()

            _, summary, loss_val = session.run([optimizer, merged, loss], feed_dict=feed_dict, run_metadata=run_metadata)
            average_loss += loss_val

            writer.add_summary(summary, step)

            if step % 20000 == 0:
                if step > 0:
                    average_loss /= 20000
                print('Average loss al passo ', step, ':', average_loss)
                average_loss=0

            if step % 100000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_words[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_words[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
                    log_str = ''

        #3 Step: quello che è avanzato lo aggiungo alle liste temp
        while(len(center_queue)>0):
            center_temp.append(center_queue.popleft())
            context_temp.append(context_queue.popleft())
        print('Esaurito il batch relativo al file', doc)
            
    #4 Step: se è rimasto qualcosa in temp, genero l'ultimo batch
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

        writer.add_summary(summary, step)
    
    print('Sono stati generati', step, 'batch per l\'allenamento del modello')

    final_embeddings = normalized_embeddings.eval()

    with open(cartella_log+'metadata.tsv', 'w') as f:
        for i in range(vocabulary_size):
            f.write(int_to_words[i] + '\n')
    
    saver.save(session, cartella_log+'model.ckpt')
    print('Modello salvato nella cartella /log')

    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(writer, config)
    
writer.close()
