"""
Programma che analizza le cartelle con i dati pubmed e ne estrae le singole parole con il
relativo numero di occorrenze. Vengono tralasciate quelle parole che appaiono nella
collezione un numero di volte inferiore a min_freq. I dizionari vengono salvati su file
nella cartella di output.
Vengono generati tre dizionari:
    -words_to_int: mappa le parole in interi
    -int_to_words: mappa gli interi nelle parole
    -words_count: mappa le parole nella loro frequenza
Utilizzo: python build_data cartella_input cartella_output min_freq
"""
import collections
import sys
import os
import csv
import json
import pickle


cartella_input = sys.argv[1]
cartella_output = sys.argv[2]
min_freq = int(sys.argv[3])

def read_data(filename):
    with open(filename, "r") as f:
        words = f.read().split()
    return words

#1 Step: scorro tutti i file nelle cartelle per avere il conteggio delle parole
print('Inizio scansione file di testo')
words_count = collections.Counter()
words=list()
  
lista_cartelle=sorted(os.listdir(path=cartella_input))
i=0
for cartella in lista_cartelle:
    lista_file=sorted(os.listdir(path=cartella_input+cartella+'/'))
    for doc in lista_file:
        temp_words = read_data(cartella_input+cartella+'/'+doc)
        words +=temp_words
        temp_words.clear()
    lista_file.clear()
    words_count += collections.Counter(words)
    words.clear()
    i+=1
    print('Conclusa scansione della cartella', cartella,'. Lavoro al', (i/len(lista_cartelle)) *100, '%')
    
#2 Step: elimino le parole poco frequenti
print('Eliminazione parole poco frequenti:')
frequent_words = {word:freq for word,freq in words_count.items() if words_count[word]>= min_freq}
words_count.clear() #libero l'oggetto counter per liberare memoria
print('Eliminazione parole poco frequenti completata')

#3 Step: genero i dizionari words_to_int e int_to_words
print('Generazione e salvataggio dizionari:')
words_to_int=dict()
for word in frequent_words:
    words_to_int[word]=len(words_to_int)
int_to_words = dict(zip(words_to_int.values(), words_to_int.keys()))

#4 Step: salvataggio dei dizionari su file
   
with open(cartella_output + 'words_to_int.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(words_to_int.items())
print('Salvataggio words_to_int.csv completata')

with open(cartella_output + 'int_to_words.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(int_to_words.items())
print('Salvataggio int_to_words.csv completata')

with open(cartella_output + 'int_to_words.txt', 'w', encoding='utf-8') as f:
    f.write(str(int_to_words))
with open(cartella_output + 'words_to_int.txt', 'w', encoding='utf-8') as f:
    f.write(str(words_to_int))

with open(cartella_output + 'int_to_words.json', 'w', encoding='utf-8') as f:
    json.dump(int_to_words,f)
with open(cartella_output + 'words_to_int.json', 'w', encoding='utf-8') as f:
    json.dump(words_to_int,f)

with open(cartella_output + 'int_to_words.pickle', 'wb') as f:
    pickle.dump(int_to_words, f)
with open(cartella_output + 'words_to_int.pickle', 'wb') as f:
    pickle.dump(words_to_int, f)
with open(cartella_output + 'words_count.pickle', 'wb') as f:
    pickle.dump(frequent_words, f)
