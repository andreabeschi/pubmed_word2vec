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
words=list()

lista_cartelle = sorted(os.listdir(path=cartella_input))
i=0
for cartella in lista_cartelle:
    lista_file=sorted(os.listdir(path=cartella_input+cartella+'/'))
    for doc in lista_file:
        temp_words = read_data(cartella_input+cartella+'/'+doc)
        words +=temp_words
        temp_words.clear()
    lista_file.clear()
    words_count = collections.Counter(words)
    with open(cartella_output + cartella + '.pickle', 'wb') as f:
        pickle.dump(words_count, f)
    words_count.clear()
    i+=1
    print('Conclusa scansione della cartella',cartella,'. Lavoro al', (i/len(lista_cartelle)) *100, '%')
