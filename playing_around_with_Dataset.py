import sklearn as sk
import pandas as pd
import numpy as np

#----------------------Prepairing Data----------------------#
# Load the dataset
beers = pd.read_csv('E:/Programmering/TNM108-Projekt/Datasets/beer_reviews_cleaned.csv')

# Beer categories without duplicates (104 categories)
my_beercategories = beers['beer_style'].unique()

#----------------------Functions----------------------#
#Demand user input Aroma, appearance, palate, and taste
#Return the beer with the closest scores
#Find the beer with the closest score for each category
def collect_userData(aroma, appearance, palate, taste, style):
    # Find the beer with the closest score for each category
    aroma_beer = beers.iloc[(beers['review_aroma'] - aroma).abs().argsort()[:1]]
    appearance_beer = beers.iloc[(beers['review_appearance'] - appearance).abs().argsort()[:1]]
    palate_beer = beers.iloc[(beers['review_palate'] - palate).abs().argsort()[:1]]
    taste_beer = beers.iloc[(beers['review_taste'] - taste).abs().argsort()[:1]]


    #print("User input: " + style)
    found_style = "No styles found"
    found_stylez = []
    # Find the beer style where user input is at all featured
    for i in range(len(my_beercategories)):
        #If style is found within a string add it to found_style
        if(str.find(my_beercategories[i], style)!=-1):
            #print("Found style: " + my_beercategories[i])
            found_stylez.append(my_beercategories[i]) 

    if(len(found_stylez)>0):
        found_style = found_stylez
    # Return the beer with the closest score for each category
    return aroma_beer, appearance_beer, palate_beer, taste_beer, found_style

# Print the beer style of a unique beer
# If the beer has an index, print the beer style
# If the beer has no index, print nothing
def print_beer(index):
    for i in range(len(beers)):
        if(i==index):
            #Print out everything in the row with index i
            #Rename column names
            beers.columns = ['Brewery Name', 'Review Time', 'Review Platform', 'Overall Review', 'Aroma', 'Appearance', 'Palate', 'Taste', 'Average Review Score', 'Beer Style', 'Beer Name', 'Alcohol by Volume']
            print(beers.loc[i])
            #print(beers['beer_style'][i+1])
    


#----------------------Main----------------------#
#print(beers.head().to_string(index=False, justify='center') + "\n")

#print(beers.columns) # Print the column names (categories)

#print(beers.head().to_string(index=False, justify='center') + "\n")
#print(my_beercategories) # All beer categories with duplicates

#print(my_beercategories)
#print(len(my_beercategories))

#print_beer(248550)

arom,appearance,palate,taste,style = collect_userData(int(input("Enter Aroma (0-5): ")), int(input("Enter Appearance (0-5): ")), int(input("Enter Palate (0-5): ")), int(input("Enter Taste (0-5): ")), str.lower(input("Enter Beer Style: ")))

print(style) #prints all styles with the related term (ex Belgian)


'''
collectedBeers = [arom, appearance, palate, taste]
#Standardize the output
for i in range(len(collectedBeers)):
    print(collectedBeers[i].to_string(index=False,  justify='center' ,col_space=10) + "\n")

'''
#Retrieve closest beer of each style

#----------------------TF-IDF----------------------#
# Z is 5 beer styles

Z = beers

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)
print(tfidf_matrix)

from sklearn.metrics.pairwise import cosine_similarity
cos_similarity = cosine_similarity(tfidf_matrix[], tfidf_matrix)

#Print the beer style with the highest similarity
#print_beer_style(cos_similarity)

#-----------------------------------------------From lab 4------------------------------------------------#
#print(cos_similarity)

d1 = "The sky is blue."
d2 = "The sun is bright."
d3 = "The sun in the sky is bright."
d4 = "We can see the shining sun, the bright sun."
Z = (d1, d2, d3, d4)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
#print(vectorizer)

my_stop_words={"the", "is"}
my_vocabulary={'blue':0, 'sun':1, 'bright':2, 'sky':3}
vectorizer = CountVectorizer(stop_words=my_stop_words, vocabulary=my_vocabulary)

#print(vectorizer.vocabulary)
#print(vectorizer.stop_words)

smatrix = vectorizer.transform(Z)
#print(smatrix)

matrix = smatrix.todense()
#print(matrix)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(norm="l2")
tfidf_transformer.fit(smatrix)

# Print idf values
feature_names = vectorizer.get_feature_names_out()
import pandas as pd
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=feature_names, columns=["idf_weights"])

# Sort ascending
df_idf = df_idf.sort_values(by=["idf_weights"])
#print(df_idf)

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(smatrix)

# get tfidf vector for first document
first_document_vector = tf_idf_vector[0] # first document "The sky is blue."
# print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df = df.sort_values(by=["tfidf"], ascending=False)
#print(df)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)
#print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity
cos_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
#print(cos_similarity)
