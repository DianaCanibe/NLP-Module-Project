"""
NLP Module Project 
Author: Diana Cañibe Valle   A01749422
"""

def part1():
    
    """-------- Task 1: Sentiment Analysis -------------"""
    # Print Task number 
    width, filler = 62, '='
    print (' Task 1: Sentiment Analysis '.center(width, filler))

    #Read and store each review
    reviews=[]
    with open('tiny_movie_reviews_dataset.txt') as f:
        for line in f.readlines():
          reviews+=[line]
          
    #Replace quotes to avoid further problems     
    for i in range(20):
      reviews[i]=reviews[i].replace('"', '')
      reviews[i]=reviews[i].replace('\'', '')

    #Load sentiment analysis model
    from transformers import pipeline
    sentiment_pipeline = pipeline("sentiment-analysis",model='distilbert-base-uncased-finetuned-sst-2-english')

    #Run the model and print the evaluation for each review 
    for i in range(20):
      feels=sentiment_pipeline(reviews[i])
      feeling =feels[0]
      print(feeling['label'])      

def part2():
    
    """-------- Task 2: Named Entity Recognition (NER)------------------"""
    # Print Task number 
    width, filler = 62, '='
    print (' Task 2: Named Entity Recognition '.center(width, filler))

    #Import required functions from flair
    from flair.datasets import ColumnCorpus
    from flair.models import SequenceTagger
    from flair.trainers import ModelTrainer
    from flair.visual.training_curves import Plotter
    
    #Store twitter dataset to meet training requirements
    columns_order = {0 : 'data', 1 : 'ner'}
    directory = ''
    corpus = ColumnCorpus(directory,columns_order,
                          train_file='twitter_train',
                          dev_file='twitter_dev',
                          test_file='twitter_test')

    #Ask for training limit 
    N_EXAMPLES_TO_TRAIN = 20 #Number of epochs
    batch_size=150 

    #Load new labels form dataset
    tag_dictionary = corpus.make_label_dictionary(label_type = "ner")
    '''Note: change to 'make_tag_dictionary(tag_type=tag_type)' 
             if using a flair constructed dataset              '''

    #Load model
    tagger = SequenceTagger.load("flair/ner-english")

    #Set trainer and train
    trainer = ModelTrainer(tagger, corpus)
    trainer.train('resources/taggers/ner-english',
                  max_epochs=N_EXAMPLES_TO_TRAIN,
                  monitor_test = True,
                  mini_batch_size=batch_size)

    #Create loss training curve plot
    plot= Plotter()
    plot.plot_training_curves('resources/taggers/ner-english/loss.tsv')

    #Plot loss graphic
    from tensorflow import keras
    import matplotlib.pyplot as plt
    base_image_path='resources/taggers/ner-english/training.png'
    plt.figure(figsize = (12,8))
    plt.axis("off")
    plt.imshow(keras.utils.load_img(base_image_path),aspect='auto')

    #Plot loss graphic - alternative option 
    '''from tensorflow import keras
    import matplotlib.pyplot as plt
    base_image_path='resources/taggers/ner-english/training.png'
    plt.figure(figsize = (12,8))
    plt.axis("off")
    plt.imshow(keras.utils.load_img(base_image_path),aspect='auto')'''

def part3():

    """-------- Task 3: Translation -------------------"""
    # Print Task number 
    width, filler = 62, '='
    print (' Task 3: Translation '.center(width, filler))

    #Read and store the original text in spanish and english
    spanish_text=[] 
    with open('traduccion-espanol.txt') as f:
        for line in f.readlines():
          spanish_text+=[line]

    english_text=[] 
    with open('traduccion-ingles.txt') as f:
        for line in f.readlines():
          english_text+=[line]

    #Split english text to meet evaluation requirements
    english_list=[]
    for i in range(len(english_text)):
      english_list+=english_text[i].split()

    #Load and run Google translation model
    from googletrans import Translator
    g_translator = Translator()
    google_translation= g_translator.translate(spanish_text, dest='en',src='es')

    #Split and store google translation to meet evaluation requirements
    g_translated_text=[]
    for i in range(len(google_translation)):
      g_translated_text.append(str(google_translation[i].text).split())

    g_translated_list=[]
    for i in range(len(google_translation)):
      g_translated_list+=g_translated_text[i]

    #Load necessary requirements for Huggingface "Helsinki-NLP" translation model
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    helsinki_translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")

    #Load and run Helsinki translation model
    from transformers import pipeline
    model_checkpoint = "Helsinki-NLP/opus-mt-es-en"
    translator = pipeline("translation", model=model_checkpoint)
    helsinki_translation= translator(spanish_text)

    #Split and store helsinki translation to meet evaluation requirements
    h_translated_text=[]
    for i in range(len(spanish_text)):
      h_translated_text.append(str(list(helsinki_translation[i].values())).split())

    h_translated_list=[]
    for i in range(len(helsinki_translation)):
      h_translated_list+=h_translated_text[i]

    #Load BLEU scorer, and print the results for each translation
    import  nltk.translate.bleu_score as bleu
    smoother = bleu.SmoothingFunction()
    helsinki_score=bleu.sentence_bleu(h_translated_list, english_list,smoothing_function=smoother.method1)
    google_score=bleu.sentence_bleu(g_translated_list, english_list,smoothing_function=smoother.method1)
    print('BLEU score for Helsinki-NLP: {}'.format(helsinki_score))
    print('BLEU score for Google: {}'.format(google_score))


def main():
    part1()
    part2()
    part3()

if __name__ == '__main__':
    main()
