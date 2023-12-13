import argparse
from gensim.models import Word2Vec

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train a Word2Vec model.')
    parser.add_argument('dataset', type=str, help='Path to the text dataset file.')
    parser.add_argument('--vector_size', type=int, default=300, choices=[100, 300], help='Size of word vectors (100 or 300). Default is 300.')
    parser.add_argument('--window', type=int, default=5, help='Maximum distance between the current and predicted word. Default is 5.')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads to train the model. Default is 4.')
    parser.add_argument('--min_count', type=int, default=1, help='Minimum frequency count of words to be considered. Default is 1.')

    # Parse arguments
    args = parser.parse_args()

    # Read dataset
    with open(args.dataset, 'r') as file:
        sentences = [line.strip().split() for line in file]

    # Train the model
    model = Word2Vec(sentences, vector_size=args.vector_size, window=args.window, min_count=args.min_count, workers=args.workers)

    # Save the model
    model_save_path = args.dataset.replace('.txt', '') + '_model.vec'
    model.wv.save_word2vec_format(model_save_path, binary=False)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()

