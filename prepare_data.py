import argparse
import re
import string
from nltk.tokenize import sent_tokenize

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation and non-word characters
    text = re.sub(f"[{string.punctuation}]", " ", text)
    
    # Replace all non-space whitespace with a single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Try reading with a different encoding if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess text file for Word2Vec training.')
    parser.add_argument('input_file', type=str, help='Path to the input text file.')
    parser.add_argument('output_file', type=str, help='Path to the output preprocessed text file.')

    # Parse arguments
    args = parser.parse_args()

    # Download punctuation
    nltk.download('punkt')

    # Read and preprocess the text
    text = read_text_file(args.input_file)

    # Split text into sentences
    sentences = sent_tokenize(text)

    # Preprocess each sentence and write to output file
    with open(args.output_file, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            processed_sentence = preprocess_text(sentence)
            file.write(processed_sentence + '\n')

    print(f"Preprocessed text saved to {args.output_file}")

if __name__ == '__main__':
    main()

