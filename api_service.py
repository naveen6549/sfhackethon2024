from flask import Flask, jsonify, request
import spacy
from textblob import TextBlob
import re

app = Flask(__name__)

def is_likely_email(text):
    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    return bool(email_pattern.match(text))

def extract_product_entities(doc):
    product_entities = set()
    common_words_to_exclude = {"details"}

    for token in doc:
        lower_token = token.text.lower()
        for keyword in ["marketing", "ai", "sales", "service", "marketing", "commerce", "data", "tableau", "mulesoft", "slack", "platform", "net", "zero", "small", "business", "partners", "success"]:
            if keyword in lower_token and lower_token not in common_words_to_exclude and not is_likely_email(lower_token):
                product_entities.add(lower_token)

    return list(product_entities)

def analyze_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ != 'ORG']
    product_entities = extract_product_entities(doc)
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    return entities, product_entities, sentiment

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')

    entities, product_entities, sentiment = analyze_text(text)

    response = {
        "Text": text,
        "Named Entities (excluding ORG)": entities,
        "Salesforce Product Entities": product_entities,
        "Sentiment": sentiment
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
