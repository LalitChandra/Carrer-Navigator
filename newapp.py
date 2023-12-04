from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from matplotlib.gridspec import GridSpec
import re
import nltk
from nltk.corpus import stopwords
import string

app = Flask(__name__)

# Load the dataset
resumeDataSet = pd.read_csv('UpdatedResumeDataSet.csv', encoding='utf-8')
resumeDataSet['cleaned_resume'] = ''

# Count plot for Category distribution
plt.figure(figsize=(20, 5))
plt.xticks(rotation=90)
ax = sns.countplot(x="Category", data=resumeDataSet)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
plt.grid()

# Pie chart for Category distribution
plt.figure(1, figsize=(22, 22))
the_grid = GridSpec(2, 2)

targetCounts = resumeDataSet['Category'].value_counts()
targetLabels = resumeDataSet['Category'].unique()

cmap = plt.get_cmap('coolwarm')
plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')

source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True)

# Function to clean resume text
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)
    resumeText = re.sub('RT|cc', ' ', resumeText)
    resumeText = re.sub('#\S+', '', resumeText)
    resumeText = re.sub('@\S+', '  ', resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)
    return resumeText

# Apply cleanResume function to the 'Resume' column
resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))

# Tokenization and word frequency analysis
nltk.download('stopwords')
nltk.download('punkt')
oneSetOfStopWords = set(stopwords.words('english')+['``', "''"])
totalWords = []
Sentences = resumeDataSet['Resume'].values
cleanedSentences = ""
for records in Sentences:
    cleanedText = cleanResume(records)
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)

wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)

# Label encoding for 'Category' column
var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])

# Train-test split and TF-IDF Vectorization
requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english')
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=42, test_size=0.2,
                                                    shuffle=True, stratify=requiredTarget)

# Train KNeighbors Classifier
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

# Web app route
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

# Web app route
@app.route('/index')
def index():
    return render_template('index.html')
    
# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get skills from the form
#         new_resume_text = request.form['skills']

#         # Clean the new resume text
#         cleaned_new_resume = cleanResume(new_resume_text)

#         # Transform the cleaned text into features using the trained TfidfVectorizer
#         new_resume_features = word_vectorizer.transform([cleaned_new_resume])

#         # Predict the category for the new resume using the trained classifier
#         predicted_category = clf.predict(new_resume_features)

#         # Decode the predicted category using the LabelEncoder
#         predicted_category_decoded = le.inverse_transform(predicted_category)

#         return render_template('result.html', predicted_category=predicted_category_decoded[0])


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get skills from the form, assuming skills are comma-separated
        user_skills = request.form['skills']

        # Check if the input is empty or contains only spaces
        if not user_skills.strip():
            return render_template('result.html', predicted_category="Invalid skills are not acceptable. Please provide appropriate skills.")

        # Clean the new resume text
        cleaned_new_resume = cleanResume(user_skills)

        # Check if the cleaned text is empty or contains only non-alphabetic characters
        if not cleaned_new_resume.strip() or not any(char.isalpha() for char in cleaned_new_resume):
            return render_template('result.html', predicted_category="Invalid skills are not acceptable. Please provide appropriate skills.")

        # Tokenize the user-provided skills
        user_skills_tokens = [skill.strip() for skill in cleaned_new_resume.split()]

        # Check if at least one skill is a technical skill
        technical_skills_present = any(skill in totalWords for skill in user_skills_tokens)

        if not technical_skills_present:
            return render_template('result.html', predicted_category="Invalid skills are not acceptable. Please provide appropriate skills.")

        # Transform the cleaned text into features using the trained TfidfVectorizer
        new_resume_features = word_vectorizer.transform([cleaned_new_resume])

        # Predict the category for the new resume using the trained classifier
        predicted_category = clf.predict(new_resume_features)

        # Decode the predicted category using the LabelEncoder
        predicted_category_decoded = le.inverse_transform(predicted_category)

        return render_template('result.html', predicted_category=predicted_category_decoded[0])


if __name__ == '__main__':
    app.run(debug=True)
