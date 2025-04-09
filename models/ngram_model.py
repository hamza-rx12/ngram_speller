import math
import nltk
import numpy as np
from collections import defaultdict, Counter


class ngram_model:
    def __init__(self, data_file, ngram_size=2):
        self.ngram_size = ngram_size
        self.data_file = data_file
        self.word_counts = None
        self.k = 0.01
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.train()

    ##############################
    ####### Data preparator ######
    ##############################
    def prepare_data(self):
        sentences = self.load_data(self.data_file)
        sentences = self.start_end_tokens(sentences, self.ngram_size)
        sentences = [sent.split() for sent in sentences]
        sentences = self.tokenize_sentences(sentences)
        return sentences

    ##############################
    ####### Model trainer ########
    ##############################
    def train(self):
        tok_sents = self.prepare_data()
        for sent in tok_sents:
            for i in range(len(sent) - self.ngram_size + 1):
                # ngram
                ngram = tuple(sent[i : i + self.ngram_size])
                self.ngram_counts[ngram] += 1
                # context
                context = tuple(sent[i : i + self.ngram_size - 1])
                self.context_counts[context] += 1
            # last one (context)
            context = tuple(sent[1 - self.ngram_size :])
            self.context_counts[context] += 1

    ################################
    ## Sentence log probabibility ##
    ################################
    def predict_ngram(self, sentence):
        sentence = self.start_end_tokens([sentence], self.ngram_size)
        sentence = [sent.split() for sent in sentence][0]
        sentence = [word if self.word_counts[word] > 1 else "UNK" for word in sentence]
        log_prob = 0
        n = self.ngram_size
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i : i + n])  # the dictionary takes tuples
            context = tuple(sentence[i : i + n - 1])
            ngram_count = self.ngram_counts[ngram]
            context_count = self.context_counts[context]
            # print(ngram, context, ngram_count, context_count)
            log_prob += math.log(
                (ngram_count + self.k)
                / (context_count + self.k * len(self.ngram_counts))
            )
        # print(log_prob)
        return log_prob

    ##################################
    ######### Perplexity #############
    ##################################
    def text_perplexity(self, test_file):
        total_log_prob = 0.0
        total_tokens = 0

        with open(test_file, "r") as file:
            test_sentences = file.read().lower().split("\n")

        for sentence in test_sentences:
            total_log_prob += self.predict_ngram(sentence, self.ngram_size)
            total_tokens += len(sentence.split()) + 1  # +1 for the end token

        avg_log_prob = total_log_prob / total_tokens
        perplexity = math.exp(-avg_log_prob)
        # print(perplexity)
        return perplexity

    ####################################
    ######## Text Generation ###########
    ####################################
    def generate_text(self, max_len=100):
        current_context = ["<s>"] * (self.ngram_size - 1)
        generated_text = []

        for _ in range(max_len):
            context_tuple = tuple(current_context)
            # getting possible next words based on the current context
            possible_next_words = [
                ngram[-1] for ngram in self.ngram_counts if ngram[:-1] == context_tuple
            ]
            if not possible_next_words:
                break

            # calculating the probabilities of the possible next words
            probabilities = np.array(
                [
                    (self.ngram_counts[context_tuple + (word,)] + self.k)
                    / (
                        self.context_counts[context_tuple]
                        + self.k * len(self.ngram_counts)
                    )
                    for word in possible_next_words
                ]
            )
            probabilities /= probabilities.sum()  # Normalize to get probabilities

            # using np.random.choice to choose the next word based on the probabilities
            next_word = np.random.choice(possible_next_words, p=probabilities)
            if next_word == "</s>":
                break

            # appending the next word to the generated text and updating the current context
            generated_text.append(next_word)
            current_context = current_context[1:] + [next_word]

        return " ".join(generated_text)

    ######################################
    ########## Autocompletion ############
    ######################################
    def autoComplete(self, text, ngram_size=2):
        # preparing the input text for prediction
        tokens = text.lower().split()
        current_context = ["<s>"] * (ngram_size - 1) + tokens
        context_tuple = tuple(current_context[-(ngram_size - 1) :])

        # getting possible next words based on the current context
        possible_next_words = [
            ngram[-1] for ngram in self.ngram_counts if ngram[:-1] == context_tuple
        ]

        # if no possible next words, return None
        if not possible_next_words:
            return None

        # calculating the probabilities of the possible next words
        probabilities = np.array(
            [
                (self.ngram_counts[context_tuple + (word,)] + self.k)
                / (self.context_counts[context_tuple] + self.k * len(self.ngram_counts))
                for word in possible_next_words
            ]
        )
        probabilities /= probabilities.sum()  # Normalize to get probabilities

        # using np.argmax to get the index of the most probable next word
        next_word = possible_next_words[np.argmax(probabilities)]
        return next_word

    ###############################
    ############ Tools ############
    ###############################

    def start_end_tokens(self, sentences, ngram_size):
        return ["<s> " * (ngram_size - 1) + sent + " </s>" for sent in sentences]

    ################
    def tokenize_sentences(self, sentences):
        self.word_counts = Counter(word for sent in sentences for word in sent)
        tok_sents = [
            [word if self.word_counts[word] > 1 else "<UNK>" for word in sent]
            for sent in sentences
        ]
        return tok_sents

    ################
    def load_data(self, data_file):
        with open(data_file, "r") as file:
            text = file.read().lower()
            return nltk.sent_tokenize(text)


##########################################################
################### TESTS ################################
##########################################################
if __name__ == "__main__":
    data_file = "data/big_data.txt"
    ml = ngram_model(data_file, ngram_size=10)
    # print(ml.ngram_counts)
    ml.predict_ngram("hello my name is hamza ")
    print(ml.generate_text())
