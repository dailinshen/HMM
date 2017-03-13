import math
import nltk
import time

# Skeleton Code
# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}

    # --------------------------Unigram Calculation-----------------------
    for i in xrange(len(training_corpus)):
        current_s = START_SYMBOL + " " + training_corpus[i] + " " + STOP_SYMBOL
        words = current_s.split()
        for each_w in words:
            if each_w != START_SYMBOL:
                if tuple([each_w]) in unigram_p:
                    unigram_p[tuple([each_w])] += 1
                else:
                    unigram_p[tuple([each_w])] = 1

    log_No_words = math.log(sum(unigram_p.values()), 2)

    unigram_p[tuple([START_SYMBOL])] = len(training_corpus)
    uni_count_d = unigram_p.copy()

    unigram_p.update((key, math.log(value, 2) - log_No_words) for key, value in unigram_p.items())

    # print unigram_p[('captain',)]
    # print unigram_p[('near',)]

    # --------------------------Bigram Calculation-----------------------
    for each_s in training_corpus:
        each_s = START_SYMBOL + " " + each_s + " " + STOP_SYMBOL
        pair_words = each_s.split()
        for j in xrange(1, len(pair_words)):
            if tuple([pair_words[j - 1], pair_words[j]]) in bigram_p:
                bigram_p[tuple([pair_words[j - 1], pair_words[j]])] += 1
            else:
                bigram_p[tuple([pair_words[j - 1], pair_words[j]])] = 1

    bigram_p[tuple([START_SYMBOL, START_SYMBOL])] = len(training_corpus)
    bi_count_d = bigram_p.copy()

    bigram_p.update((key, math.log(value, 2) - math.log(uni_count_d[tuple([key[0]])], 2)) for key, value in bigram_p.items())

    # print bigram_p[('and','religion')]
    # print bigram_p[('near','the')]

    # --------------------------Trigram Calculation-----------------------
    for k in xrange(len(training_corpus)):
        current_s = START_SYMBOL + " " + START_SYMBOL + " " + training_corpus[k] + " " + STOP_SYMBOL
        tri_words = current_s.split()
        for x in xrange(2, len(tri_words)):
            if tuple([tri_words[x - 2], tri_words[x - 1], tri_words[x]]) in trigram_p:
                trigram_p[tuple([tri_words[x - 2], tri_words[x - 1], tri_words[x]])] += 1
            else:
                trigram_p[tuple([tri_words[x - 2], tri_words[x - 1], tri_words[x]])] = 1

    trigram_p.update((key, math.log(value, 2) - math.log(bi_count_d[tuple(key[:2])], 2)) for key, value in trigram_p.items())

    # print trigram_p[('and','not','a')]
    # print trigram_p[('near', 'the', 'ecliptic')]

    return unigram_p, bigram_p, trigram_p


# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc.
def score(ngram_p, n, corpus):
    scores = []
    for each_s in corpus:
        s_score = 0
        concentration = each_s.replace('\r\n', STOP_SYMBOL).split()
        if n == 1:
            for item in concentration:
                if (item,) not in ngram_p:
                    s_score = MINUS_INFINITY_SENTENCE_LOG_PROB
                    break
                s_score += ngram_p[(item,)]
        elif n == 2:
            for item in list(nltk.bigrams([START_SYMBOL] + concentration)):
                if item not in ngram_p:
                    s_score = MINUS_INFINITY_SENTENCE_LOG_PROB
                    break
                s_score += ngram_p[item]
        elif n == 3:
            for item in list(nltk.trigrams([START_SYMBOL, START_SYMBOL] + concentration)):
                if item not in ngram_p:
                    s_score = MINUS_INFINITY_SENTENCE_LOG_PROB
                s_score += ngram_p[item]
        scores.append(s_score)
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

#TODO: IMPLEMENT THIS FUNCTION
# Calculcates the perplexity of a language model
# scores_file: one of the A2 output files of scores
# sentences_file: the file of sentences that were scores (in this case: data/Brown_train.txt)
# This function returns a float, perplexity, the total perplexity of the corpus
def calc_perplexity(scores_file, sentences_file):

    score_infile = open(scores_file, "r")
    scores = score_infile.readlines()
    score_infile.close()
    sentence_infile = open(sentences_file, 'r')
    sentences = sentence_infile.readlines()
    sentence_infile.close()

    subsum, M = 0, 0
    for each_score in scores:
        subsum += float(each_score)
    for each_s in sentences:
        M += len(each_s.split()) + 1

    perplexity = 2 ** (- subsum / M)
    return perplexity

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    coefficience = 1.0 / 3
    for each_s in corpus:
        possibility = 0
        tri_words = (START_SYMBOL + " " + START_SYMBOL + " " + each_s + " " + STOP_SYMBOL).split()
        for index in range(2, len(tri_words)):
            if (tri_words[index - 2], tri_words[index - 1], tri_words[index]) not in trigrams:
                possibility = MINUS_INFINITY_SENTENCE_LOG_PROB;
                break
            possibility += math.log((2 ** trigrams[(tri_words[index - 2], tri_words[index - 1], tri_words[index])] + 2 ** bigrams[(tri_words[index - 1], tri_words[index])] + 2 ** unigrams[(tri_words[index],)]) * coefficience, 2)
        scores.append(possibility)

    return scores


DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()


