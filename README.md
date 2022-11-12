# NLP-Module-Project
This is the project for Module 3 refering to Natural Language Processing 
for the advanced artificial intelligence for data science concentration 
at Tec de Monterrey Campus Estado de México

## Description
The project has 3 main parts refering to different tasks:
1. Sentiment analysis over a set of movie reviews 
2. Train of an NER model with a twitter dataset
3. Comparing the BLEU score for 2 diferent translations over the same text

For the first task I used the sentiment analysis pipeline on the transformers library with the default model,
the clasification for each of the reviews is either POSITIVE or NEGATIVE. 

On the second one, I used flair to further train its ner-english model over a twitter dataset.<br />
At the end of the training the loss curve obtained is the following:
![image](https://user-images.githubusercontent.com/101147406/201423135-3ee62d15-85a5-4fc8-936e-31172d992857.png)

NOTE: This is just a reference image, results ought to change between trainings. 

Finally for the translation I used "googletrans" from Google and "Helsinki-NLP/opus-mt-es-en" available at Huggingface, the translation was made from spanish to english, an evaluated using the english original text. 

### Installing
Make sure you complete each of the next steps:
1. Clone this repository, or download all the files 
2. Install anacoda
3. Run the following command : ```conda create --name <env_name> --file requirements.txt```
4. If you are not using anaconda install the following libraries:<br />
   a. pip install -q transformers<br />
   b. pip install transformers[sentencepiece]<br />
   c. pip install googletrans==3.1.0a0<br />
   d. pip install flair<br />

### Executing program
For execution you just have to run the "run.py" file
Before doing that check that you have changed in the code the variables:<br />
 a. N_EXAMPLES_TO_TRAIN , this will limit the number of epochs to run<br />
 b. batch_size (do not change to more than 700, cause it would not work due to the size of the dataset)<br />
Also, be sure that the data files are stored in the same work space as the code, otherwise change the path the are in on the code.

If you have problems displaying the plot for the second task, there is an alternative version on the code, just uncomment it.
(You will require 'keras' and 'matplotlib' for this option)

### Data files description
1. 'tiny_movie_reviews_dataset.txt' 20 different movie reviews (sentiment analysis part) 
2. 'twitter_test'  3850 words from tweets used for testing  (NER part)
3. 'twitter_train' 2394 words from tweets used for training (NER part)
4. 'twitter_dev' 1000 words from tweets used for validation (NER part)
5. 'traduccion_espanol' Spanish version of the 100 first lines of the European Parliament Proceedings Parallel Corpus 1996-2011 (translation part)
6. 'traduccion_ingles' English version of the 100 first lines of the European Parliament Proceedings Parallel Corpus 1996-2011 (translation part)

### Author
Diana Cañibe Valle
A01749422
