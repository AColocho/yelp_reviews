import streamlit as st
from classification_model import Model
model = Model()

def evaluate_text(text):
    pred = model.predict([text])
    if pred[0] == 0:
        return 'This review will not get stars'
    else:
        return 'This review will get stars'

'''
# Classification Demo
In the following text area, you can write or paster text. The model will predict if the review will get stars on yelp or not.
'''
data = st.text_area(label='Review')
st.text(evaluate_text(data))

'''
# Background
The dataset contained over 7 million Yelp reviews, and it suffered from major class imbalances. The following chart represents a 10,000 review sample.
'''
st.image('app/photos/chart1.png')

'''
Originally, this project seeked to classify Yelp reviews in one of four buckets which were stars, useful, cool, or funny. However, the approach was
reframed to **Can we predict if a review will receive stars?**.

# Approach
3 million reviews were processed and model in the following way:
- Punctuation was removed.
- Stop words were removed.
- Reviews were lemmatized.
- Reviews was vectorize using term frequency inverse document frequency.
- Complement Naive Bayes algorithm was used.

# Results
- F1 Score: .83
- Recall: .74
- Precision: .93
- Accuracy: .74

# Future Work
- Text generator.
- Review summarization
'''