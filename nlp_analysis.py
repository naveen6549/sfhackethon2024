import spacy

def analyze_text(text):
    # Load the English model
    nlp = spacy.load("en_core_web_sm")

    # Process the text with spaCy NLP pipeline
    doc = nlp(text)

    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return entities

if __name__ == "__main__":
    # Example text for analysis
    sample_text = """
Subject: Assistance Needed - Case #12345

Dear Support Team,

I hope this email finds you well. I am experiencing an issue with your product, and I would appreciate your assistance in resolving it. Below are the details of the problem:

Product: XYZ Widget
Issue: The widget is not functioning as expected.
Error Message: [Include any error messages if applicable]

Please let me know if you need any additional information from my end. I have attached screenshots for your reference.

Thank you for your prompt attention to this matter.

Best regards,
John Doe
john.doe@email.com
"""


    # Perform NLP analysis
    result = analyze_text(sample_text)

    # Display the results
    print("Text:", sample_text)
    print("Named Entities:", result)
