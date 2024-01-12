import spacy
from textblob import TextBlob
import re

def is_likely_email(text):
    # Check if the text is likely to be an email address based on format
    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    return bool(email_pattern.match(text))

def extract_product_entities(doc):
    product_entities = set()
    common_words_to_exclude = {"details"}  # Add more common words as needed

    for token in doc:
        lower_token = token.text.lower()
        for keyword in ["marketing", "ai", "sales", "service", "marketing", "commerce", "data", "tableau", "mulesoft", "slack", "platform", "net", "zero", "small", "business", "partners", "success"]:
            if keyword in lower_token and lower_token not in common_words_to_exclude and not is_likely_email(lower_token):
                product_entities.add(lower_token)

    return list(product_entities)

def analyze_text(text):
    # Load the English model
    nlp = spacy.load("en_core_web_sm")

    # Process the text with spaCy NLP pipeline
    doc = nlp(text)

    # Extract named entities excluding organizations
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ != 'ORG']

    # Extract product-related entities with email exclusion
    product_entities = extract_product_entities(doc)

    # Perform sentiment analysis using TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    return entities, product_entities, sentiment

if __name__ == "__main__":
    # Example text for analysis
    sample_text = """
    Subject: Assistance Needed - Case #12345

    Dear Naveen Singh, 
    I hope this email finds you well. I am experiencing an issue with Salesforce Partners Portal, and I would appreciate your assistance in resolving it. Below are the details of the problem:

    Product: Partners Portal and service cloud & einstein.
    Issue: The feature 'X' is not working as expected.
    Error Message: [Include any error messages if applicable]

    Please let me know if you need any additional information from my end. I have attached screenshots for your reference.

    Thank you for your prompt attention to this matter.

    Best regards,
    John Doe
    john.doe@email.com
    """

    # Perform NLP analysis
    entities, product_entities, sentiment = analyze_text(sample_text)

    # Display the results
    print("Text:", sample_text)
    print("Named Entities (excluding ORG):", entities)
    print("Salesforce Product Entities:", product_entities)
    print("Sentiment:", sentiment)
