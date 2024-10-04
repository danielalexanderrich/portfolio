import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os

plt.style.use('ggplot')
os.chdir('C:\\Users\\Dan\\OneDrive\\Desktop\\data sets')
df = pd.read_csv('Reviews.csv')
##look at counts of reviews by stars
ax = df['Score'].value_counts().sort_index().plot(
    kind='bar', title='Reviews by Stars', figsize=(10, 5)
)
ax.set_xlabel('Review Stars')
plt.show(block=True)

##use VADERS to perform sentiment analysis on sample of reviews (for the sake of time)
sia = SentimentIntensityAnalyzer()
sampled_df = df.sample(n=1000, random_state=42) #downsample
res = {}
for i in range(len(sampled_df)):
    row = sampled_df.iloc[i]
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)
    if i % 100 == 0:
        print(i)
vaders = pd.DataFrame(res).T ##transposed data frame of the results dictionary -- dict has associated pos, neu, neg, and compound scores for each review ID in sample
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(sampled_df, how='left') ##merge original sampled df with vaders scores and plot below

mean_compound = vaders.groupby('Score')['compound'].mean().reset_index()
# Create the bar plot
plt.bar(mean_compound['Score'], mean_compound['compound'])
plt.title('Compound Score by Amazon Star Review')
plt.xlabel('Score')
plt.ylabel('Compound')
plt.xticks(rotation=45)  # Rotate x-axis labels if necessary
plt.show(block=True)

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show(block=True)



###future work will include 


