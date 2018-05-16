"""
Programma che riceve in input la cartella dove sono contenuti i file .pickle generati da 
build_temp.py
Vengono caricati in memoria tutti i Counter relativi alle cartelle di Pubmed, vengono sommati in un
unico oggetto Counter (che viene salvato). Vengono in seguito eliminate le parole poco frequenti, 
cioè che appaiono nella collezione un numero di volte inferiore a min_freq. In seguito le parole 
vengono ordinate per frequenza in modo decresente e vengono infine generati e salvati i dizionari:
    -words_to_int: mappa le parole in interi
    -int_to_words: mappa gli interi nelle parole
    -words_count: mappa le parole nella loro frequenza
    
utilizzo: python build_dict cartella_input cartella_output min_freq
"""
import pickle
import os
import sys
import collections
import operator

cartella_input=sys.argv[1]
cartella_output=sys.argv[2]
min_freq=int(sys.argv[3])

def read_pickle(filename):
    with open(filename, 'rb') as f:
        count = pickle.load(f)
    return count

#1 Step: scorro tutti i file .pickle generati in precedenza e li aggrego in un unico Counter
print('Inizio scansione file .pickle')
words_count = collections.Counter()
lista_file=sorted(os.listdir(path=cartella_input))
print('Verranno caricati',len(lista_file),'file .pickle')

for doc in lista_file:
    temp_count=read_pickle(cartella_input+doc)
    words_count+=temp_count
    temp_count.clear()

print('Scansione file .pickle completata')

with open(cartella_output + 'words_count_complete.pickle', 'wb') as f:
    pickle.dump(words_count,f)
print('Salvataggio words_count_complete completato')

#2 Step: elimino le parole poco frequenti
print('Eliminazione parole poco frequenti:')
frequent_words = {word:freq for word,freq in words_count.items() if words_count[word]>=min_freq}
words_count.clear()
print('Eliminazione parole poco frequenti completata')

#3 Step: genero i dizionari e li salvo su file
print('Generazione dizionari:')
sorted_by_freq=sorted(frequent_words.items(), key=operator.itemgetter(1), reverse=True)
words_to_int=dict()
for word in sorted_by_freq:
    words_to_int[word[0]]=len(words_to_int)
int_to_words = dict(zip(words_to_int.values(), words_to_int.keys()))
sorted_by_freq.clear()
print('Dizionari generati')

print('Salvataggio:')
with open(cartella_output + 'int_to_words.pickle', 'wb') as f:
    pickle.dump(int_to_words, f)
with open(cartella_output + 'words_to_int.pickle', 'wb') as f:
    pickle.dump(words_to_int, f)
with open(cartella_output + 'words_count.pickle', 'wb') as f:
    pickle.dump(frequent_words, f)
print('Salvataggio completato')
