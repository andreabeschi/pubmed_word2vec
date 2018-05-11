# pubmed_word2vec
This is my master's degree thesis project in Computer System Engineering

The goal of this project is to apply the skipgram model proposed by Mikolov et al. to the PubMed collection. There are 3 Python programs:
  * parsing_xml.py : takes the PubMed .xml files in input and makes the parsing to extract ID, Title and Abstract (if present). A text file is generated for each article. The file's name is the corresponding Article ID
                    
    **usage**: python parsing_xml.py input_folder output_folder
                    
  * build_data.py : takes the text files in input, extract the single words, removes thoose words that appear less than min_freq times and in the end generates the dictionaries:
    * words_to_int: maps words in integers
    * int_to_words: maps integers in words
    * words_count: maps words in theyr occurrence numbers

    **usage**: python build_date input_folder output_folder min_freq
    
  * skipgram_training.py : loads the dictionaries previously saved from the dictionaries_folder, generates and trains the skipgram model using the TensorFlow framework. The training batches are generated from the text files with Titles and Asbtracts
