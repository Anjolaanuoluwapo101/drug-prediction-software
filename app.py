from flask import Flask, jsonify
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline

app = Flask(__name__)

# Load the tokenizer and model for BioBERT
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = BertForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Set up the Named Entity Recognition (NER) pipeline
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer)

def merge_entities(ner_results):
    """
    Merges subwords that belong to the same entity.
    """
    merged_entities = []
    current_entity = {"word": "", "entity": "", "score": 0.0}
    
    for idx, entity in enumerate(ner_results):
        # Check if the token is a continuation of the previous one (subword starts with ##)
        if entity['word'].startswith("##"):
            current_entity["word"] += entity['word'][2:]  # Append subword without ##
            current_entity["score"] = max(current_entity["score"], entity["score"])  # Keep highest confidence score
        else:
            # If there is a previous entity, add it to the list
            if current_entity["word"]:
                merged_entities.append(current_entity)

            # Start a new entity
            current_entity = {
                "word": entity['word'],
                "entity": entity['entity'],
                "score": entity['score']
            }

    # Add the last entity if exists
    if current_entity["word"]:
        merged_entities.append(current_entity)

    return merged_entities

@app.route("/")
def hello_world():
    # Input text to analyze
    text = "The patient was prescribed Paracetamol and Amoxicillin to treat a bacterial infection."

    # Use the NER pipeline to identify drug names and other biomedical entities
    ner_results = nlp_ner(text)

    # Merge subwords to form complete entities
    merged_entities = merge_entities(ner_results)

    # Prepare a list to store the formatted results
    entities = [
        {"Entity": entity['word'], "Label": entity['entity'], "Confidence": f"{entity['score']:.2f}"}
        for entity in merged_entities
    ]

    # Return the results as JSON
    return jsonify(entities)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
