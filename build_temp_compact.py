import collections
import sys
import os
import pickle

cartella_input = sys.argv[1]
cartella_output = sys.argv[2]

def read_data(filename):
    with open(filename, "r") as f:
        words = f.read().split()
    return words

print('Inizio scansione file di testo')

lista_file = sorted(os.listdir(path=cartella_input))
totali=len(lista_file)
i=0
for doc in lista_file:
    temp_words = read_data(cartella_input+doc)
    words_count = collections.Counter(temp_words)
    temp_words.clear()
    with open(cartella_output + doc + '.pickle', 'wb') as f:
        pickle.dump(words_count, f)
    words_count.clear()
    i+=1
    print('Conclusa scansione del file',doc,'. Lavoro al', (i/totali) *100, '%')
