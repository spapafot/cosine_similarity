import sys
import argparse
import spacy
from scipy import spatial

# Add any spacy model with pretrained vectors
nlp = spacy.load("el_core_news_lg")
# Lambda to calculate the cosine similarity
cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)


def find_similar_words(word):
    try:
        assert nlp.vocab[word].has_vector
    except AssertionError:
        print("Not in vocab")
        sys.exit()
    finally:
        word_vector = nlp.vocab[word].vector

    computed = []

    # Search to make sure it is a word, it has a vector and is not a number
    for i in nlp.vocab:
        if i.has_vector & i.is_alpha:
            similarity = cosine_similarity(word_vector, i.vector)
            computed.append([i.text, similarity])

    # sort and filter according to the length of the word
    computed = sorted(computed, key=lambda x: -x[1])
    computed = list(filter(lambda x: len(x[0]) > 4, computed))

    top_five = [i[0] for i in computed[:5]]
    print(top_five)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find similar words by cosine similarity')
    parser.add_argument('-w', type=str, required=True, help="add any word")
    args = parser.parse_args()
    find_similar_words(args.w)



