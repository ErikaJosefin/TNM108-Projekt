import sklearn
import pandas as pd
import numpy as np


#----------------------Prepairing Data----------------------#
# Load the Beer_reviews dataset
beerdir = 'E:/Programmering/TNM108-Projekt/Datasets/beer_reviews.csv'

beers = pd.read_csv(beerdir)

#To lowercase
beers['beer_style'] = beers['beer_style'].str.lower()
#Sort beers by overall review, ascending
beers.sort_values(by='review_overall', ascending=False, inplace=True)
#Extract the beer categories
my_beercategories = [beers['beer_style']] 
# Beer categories without duplicates (104 cars)
my_beercategories = beers['beer_style'].unique()   
#Add a new column with the average review scoring
beers['avg_score'] = beers[['review_aroma', 'review_appearance', 'review_palate',  'review_taste']].mean(axis=1).round(1)
#Swap places for column beer_style and review_taste
beers = beers[['brewery_name','review_time', 'review_profilename','review_overall', 'review_aroma', 'review_appearance', 'review_palate',  'review_taste', 'avg_score','beer_style','beer_name', 'beer_abv']]
#Convert review_time to datetime
beers['review_time'] = pd.to_datetime(beers['review_time'], unit='s')


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

print(style)

#From lab 4
# Z is 5 beer styles
'''
Z = my_beercategories

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)
#print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity
cos_similarity = cosine_similarity(tfidf_matrix[style], tfidf_matrix)

#Print the beer style with the highest similarity
#print_beer_style(cos_similarity)
print(cos_similarity)

'''