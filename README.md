# NLP-Module-Project
This is the project for Module 3 refering to Natural Language Processing 
for the advanced artificial intelligence for data science concentration 
at Tec de Monterrey Campus Estado de México

## Description
The project has 3 main parts refering to diferent tasks:
1. Sentiment analysis over a set of movie reviews 
2. Train of an NER model with a twitter dataset
3. Comparing the BLEU score for 2 diferent translations over the same text

For the first task I used the sentiment analysis pipeline on the transformers library with the default model,
the clasification for each of the reviews is either POSITIVE or NEGATIVE. 

On the second one, I used flair to further train its ner-english model over a twitter dataset. 
At the end of the training the loss curve obtained is the following:
![image](https://user-images.githubusercontent.com/101147406/201423135-3ee62d15-85a5-4fc8-936e-31172d992857.png)

NOTE: This is just a reference image, results ought to change between trainings. 

Finally for the translation I used "googletrans" and "Helsinki-NLP/opus-mt-es-en", the translation was made from a 
text in spanish to english, an evaluated using the english original text. 

### Installing
Make sure you complete each of the next steps:
1. Install anacoda
2. Clone this repository, or download all the files 
2. Run the following command : ```conda create --name <env_name> --file requirements.txt```

### Executing program
For execution you just have to run the "run.py" file
Before doing that check that you have changed in the code the variables:
 a.N_EXAMPLES_TO_TRAIN 

### Author
Diana Cañibe
