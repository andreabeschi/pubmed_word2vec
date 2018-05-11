"""
Programma che effettua il parsing dei file .xml della collezione di Pubmed.
Viene processato ogni file .xml all'interno della cartella indicata nella variabile cartella
Da ogni file viene generata una cartella con il nome del file, all'interno della quale
vengono salvati i file di testo relativi agli articoli estratti: il nome è L'ID dell'articolo
e al suo interno vengono salvati il titolo dell'articolo e l'abstract (se presente).
Il testo dei titoli e degli abstract viene processato per:
    -convertire tutte le parole in lowercase
    -eliminare la punteggiatura
    -sostituire i numeri con le parole equivalenti
Per il parsing viene utilizzata la libreria ElementTree del pacchetto xml di Python che
permette di caricare l'albero corrispondente al file .xml on the fly evitando di occupare tutta la memoria disponibile solo per caricare il file
Utilizzo: python parsing_xml.py folder1 folder2
folder1=cartella dove sono collocati i file .xml
folder2=cartella dove salvare i risultati
"""

import xml.etree.ElementTree as ET

import os
import time
import string
import sys


def process(text, caratteri):
    text = text.lower()
    text = text.replace('0',' zero ')
    text = text.replace('1',' one ')
    text = text.replace('2',' two ')
    text = text.replace('3',' three ')
    text = text.replace('4',' four ')
    text = text.replace('5',' five ')
    text = text.replace('6',' six ')
    text = text.replace('7',' seven ')
    text = text.replace('8',' eight ')
    text = text.replace('9',' nine ')
    text = text.replace('$',' dollars ')
    char_to_remove = str.maketrans({a:' ' for a in caratteri})
    text = text.translate(char_to_remove)
    text = text.replace('   ', ' ')
    text = text.replace('  ', ' ')
    return text

i = 0
children=['PMID','Article','ArticleTitle','Abstract','AbstractText']
punteggiatura='!\"\'()*+,-./:;<=>?@[]^_{|}&%'
cartella_input = sys.argv[1]
cartella_output = sys.argv[2]
lista_file = sorted(os.listdir(path=cartella_input))
file_totali = len(lista_file)

print('Verranno processati ', file_totali, ' file .xml')
tempo_iniziale = time.time()

for doc in lista_file:
    with open(cartella_input+doc,'r') as xml_file:
        j=0
        nome_doc = doc.replace('.xml','')
        os.mkdir(cartella_output+nome_doc)
        nome=cartella_output+ nome_doc + '/'

        for event, elem in ET.iterparse(xml_file):
            temp=list()
            if elem.tag == 'MedlineCitation':
                pmid = elem.find('PMID')
                article = elem.find('Article')
                if article != None:
                    article_title = article.find('ArticleTitle')
                    if article_title != None:
                        if article_title.text != None:
                            f = open(nome + pmid.text + '.txt', 'w')
                            t = process(article_title.text, punteggiatura)
                            f.write(t + '\n')
                            abstract = article.find('Abstract')
                            if abstract != None:
                                testo_abstract=''
                                for paragrafi in abstract.iter('AbstractText'):
                                    if paragrafi.text != None:
                                        testo_abstract = testo_abstract + paragrafi.get('Label','') + ' ' + paragrafi.text + ' '                    
                                f.write(process(testo_abstract, punteggiatura))
                                abstract.clear()
                            f.close()
                        j+=1
                        article_title.clear()
                    article.clear()
                pmid.clear()
                elem.clear()
                for child in temp:
                    child.clear()
            elif elem.tag not in children and elem.tag != 'MedlineCitation':
                elem.clear()
            else:
                temp.append(elem)

    i=i+1
    perc = ((i)/file_totali)*100
    print('Processati ', j, ' articoli in ', doc, ', processo completo al ', perc, '%' )

tempo_finale = time.time()
print('Processati ', i, ' documenti in ', tempo_finale - tempo_iniziale, ' secondi.')
                            

"""
#Versione che utilizza la funzione findall ===> MEMORY LEAK
for doc in listafile:
    with open(cartella+doc,'r') as xml_file:
        j=0
        tree = ET.parse(xml_file)
        root = tree.getroot()
        nome_doc = doc.replace('.xml','')
        #os.mkdir('/mnt/4CC6A887C6A8733E/pubmed/text/'+nome_doc)
        #nome='/mnt/4CC6A887C6A8733E/pubmed/text/'+ nome_doc + '/'
        #os.mkdir('./text/'+nome_doc)
        nome='./text/'+ nome_doc + '/'
        for article in root.findall('PubmedArticle'):
            medline_citation = article.find('MedlineCitation')
            pmid = medline_citation.find('PMID')
            art = medline_citation.find('Article')
            article_title = art.find('ArticleTitle') 
            if article_title.text != None:
                #f = open(nome + pmid.text + '.txt', 'w')
                #t=process(article_title.text, punteggiatura)
                #f.write(t + '\n')
                abstract = art.find('Abstract')
                if abstract != None:
                    abstract_text = abstract.find('AbstractText')
                    if abstract_text != None:
                        testo_abstract=''
                        for paragrafi in abstract.iter('AbstractText'):
                            if paragrafi.text != None:
                                testo_abstract = testo_abstract + paragrafi.get('Label','') + ' ' + paragrafi.text + ' '                    
                        #f.write(process(testo_abstract, punteggiatura))
                        
                j = j+1
            #f.close()
        i=i+1
        perc = ((i)/totali)*100
        print('Processati ', j, ' articoli in ', doc, ', processo completo al ', perc, '%' )
"""

"""
#Versione che utilizza iterparse di lxml ====> MEMORY LEAK
for doc in listafile:
    with open(cartella+doc,'rb') as xml_file:
        j=0
        nome_doc = doc.replace('.xml','')
        os.mkdir('./text/'+nome_doc)
        nome='./text/'+nome_doc+'/'
        for event, elem in etree.iterparse(xml_file, events=('end',), tag='MedlineCitation'):
            pmid = elem.find('PMID')
            article = elem.find('Article')

            if article != None:
                article_title = article.find('ArticleTitle')
                if article_title != None:
                    if article_title.text != None:
                        f = open(nome + pmid.text + '.txt', 'w')
                        t = process(article_title.text, punteggiatura)
                        f.write(t + '\n')
                        abstract = article.find('Abstract')
                        if abstract != None:
                            testo_abstract=''
                            for paragrafi in abstract.iter('AbstractText'):
                                if paragrafi.text != None:
                                    testo_abstract = testo_abstract + paragrafi.get('Label','') + ' ' + paragrafi.text + ' '                    
                            f.write(process(testo_abstract, punteggiatura))
                            abstract.clear()
                        f.close()
                    j+=1
                article_title.clear()

            article.clear()
            pmid.clear()
            elem.clear()
   
    i=i+1
    perc = ((i)/totali)*100
    print('Processati ', j, ' articoli in ', doc, ', processo completo al ', perc, '%' )

print('Processati ', i, ' documenti')
"""
