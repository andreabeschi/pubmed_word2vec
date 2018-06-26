# pubmed_word2vec
This is my master's degree thesis project in Computer System Engineering

The goal of this project is to apply the skipgram model proposed by Mikolov et al. to the PubMed collection. There are 3 Python programs:
  * parsing_xml.py : takes the PubMed .xml files in input and makes the parsing to extract ID, Title and Abstract (if present). A text file is generated for each article. The file's name is the corresponding Article ID
                    
    **usage**: python parsing_xml.py input_folder output_folder
                    
  * build_temp.py : takes the text files in input_folder, extract the single words and generates for each PubMed folder a Counter object that counts words occurences. Then save the Counter objects in .pickle files in output_folder

    **usage**: python build_temp.py input_folder output_folder

  * build_dict.py : takes the temp .pickle files in input_folder, extracts the Counter objects and sum all the Counter in one single Counter with words occurences for all the PubMed collection. The emoves thoose words that appear less than min_freq times and in the end generates the dictionaries and saves them in output_folder:
    * words_to_int: maps words in integers
    * int_to_words: maps integers in words
    * words_count: maps words in theyr occurrence numbers

    **usage**: python build_dict input_folder output_folder min_freq
    
  * skipgram_training.py : loads the dictionaries previously saved from the dictionaries_folder, generates and trains the skipgram model using the TensorFlow framework. The training batches are generated from the text files with Titles and Asbtracts
  
  * evaluation.py : loads the trained model and embeddings from log directory and performs intrinsic evaluation of the embeddings. In the evaluation we use three dataset: [UMNSRS-Sim](http://rxinformatics.umn.edu/SemanticRelatednessResources.html), [UMNSRS-Rel](http://rxinformatics.umn.edu/SemanticRelatednessResources.html) and [WordSim353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/)


The _compact versions extract the title/abstract information from the pubmed .xml files in one big text file. This version is faster in preprocessing and in training.
