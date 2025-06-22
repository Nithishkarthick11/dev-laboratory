# 1. Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import string
from collections import Counter

# 2. Load the Dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]  # Keep only necessary columns
df.columns = ['label', 'message']  # Rename columns

# 3. Understand Dataset Structure
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())
print("\nBasic Stats:")
print(df.describe())

# 4. Check for Null/Missing Values
print("\nMissing Values:\n", df.isnull().sum())

# 5. Analyze the Class Distribution (Spam vs Ham)
print("\nClass Distribution:")
print(df['label'].value_counts())
sns.countplot(x='label', data=df)
plt.title('Spam vs Ham Distribution')
plt.show()

# 6. Visualize Data Distribution Using Plots
df['message_length'] = df['message'].apply(len)

plt.figure(figsize=(10,6))
sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True)
plt.title('Distribution of Message Length by Class')
plt.xlabel('Message Length')
plt.ylabel('Frequency')
plt.show()

# 7. Perform Text-Specific Analysis

# Word Frequency Function
def get_words(messages):
    all_words = []
    for msg in messages:
        words = msg.lower().translate(str.maketrans('', '', string.punctuation)).split()
        all_words += words
    return Counter(all_words)

# Get top words in spam and ham messages
spam_words = get_words(df[df['label']=='spam']['message'])
ham_words = get_words(df[df['label']=='ham']['message'])

print("\nTop 10 Words in Spam Messages:")
print(spam_words.most_common(10))

print("\nTop 10 Words in Ham Messages:")
print(ham_words.most_common(10))

# Word Cloud for Spam and Ham
spam_text = " ".join(df[df['label'] == 'spam']['message'])
ham_text = " ".join(df[df['label'] == 'ham']['message'])

plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
plt.title("Spam Word Cloud")
plt.imshow(WordCloud(background_color='white').generate(spam_text), interpolation='bilinear')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Ham Word Cloud")
plt.imshow(WordCloud(background_color='white').generate(ham_text), interpolation='bilinear')
plt.axis('off')

plt.show()

# 8. Check Correlations
print("\nAverage Message Length by Class:")
print(df.groupby('label')['message_length'].mean())

plt.figure(figsize=(8,6))
sns.boxplot(x='label', y='message_length', data=df)
plt.title('Message Length by Label')
plt.show()

# 9. Conclude Insights
print("\n--- Insights ---")
print("- Spam messages are generally longer than ham messages.")
print("- Common spam words include 'free', 'win', 'claim', etc.")
print("- Word clouds highlight key differences in vocabulary between spam and ham.")
print("- Message length may serve as a useful feature for spam detection.")

