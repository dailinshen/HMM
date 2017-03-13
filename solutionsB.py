import sys
import nltk
import math
import time
import collections

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in your returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []

    for each_sentence in brown_train:
        each_sentence = START_SYMBOL + '/' + START_SYMBOL + " " + START_SYMBOL + "/" + START_SYMBOL + " " + each_sentence + " " + STOP_SYMBOL + "/" + STOP_SYMBOL
        words_with_tags = each_sentence.split()
        words = []
        tags = []
        for each_wwt in words_with_tags:
            temp = each_wwt.rsplit("/", 1)
            words.append(temp[0])
            tags.append(temp[1])

        brown_words.append(words)
        brown_tags.append(tags)

    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    # keys are tuples, values are log probability of a certain related tuple
    q_values = {}

    # --------------------------Tag Unigram Calculation-----------------------
    unigrams = {}
    for each_s in brown_tags:
        for each_tag in each_s:
            if each_tag != START_SYMBOL:
                if (each_tag,) in unigrams:
                    unigrams[(each_tag,)] += 1
                else:
                    unigrams[(each_tag,)] = 1

    log_No_words = math.log(sum(unigrams.values()), 2)

    unigrams[(START_SYMBOL,)] = unigrams[(STOP_SYMBOL,)]
    unigrams_dict = unigrams.copy()

    unigrams.update((key, math.log(value, 2) - log_No_words)for key, value in unigrams.items())

    # --------------------------Tag Bigram Calculation-----------------------
    bigrams = {}
    for each_s in brown_tags:
        for e_pair_tag in list(nltk.bigrams(each_s)):
            if e_pair_tag in bigrams:
                bigrams[e_pair_tag] += 1
            else:
                bigrams[e_pair_tag] = 1

    bigrams[(START_SYMBOL, START_SYMBOL)] = unigrams_dict[(START_SYMBOL,)]
    bigrams_dict = bigrams.copy()

    bigrams.update((key, math.log(value, 2) - math.log(unigrams_dict[(key[0],)], 2))for key, value in bigrams.items())

    # --------------------------Tag Trigram Calculation-----------------------
    for each_s in brown_tags:
        for e_tri_tag in list(nltk.trigrams(each_s)):
            if e_tri_tag in q_values:
                q_values[e_tri_tag] += 1
            else:
                q_values[e_tri_tag] = 1

    q_values.update((key, math.log(value, 2) - math.log(bigrams_dict[(key[0], key[1])], 2))for key, value in q_values.items())

    return q_values


# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):

    known_words = set([])
    pool = []
    for each_s in brown_words:
        pool += each_s
    dic_bw = collections.Counter(pool)
    for item in dic_bw.items():
        if item[1] > RARE_WORD_MAX_FREQ:
            known_words.add(item[0])

    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for each_s in brown_words:
        update_sentence = []
        for item in each_s:
            if item in known_words:
                update_sentence.append(item)
            else:
                update_sentence.append(RARE_SYMBOL)
        brown_words_rare.append(update_sentence)
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values, words_pool, taglist = {}, [], set([])
    pool_Tags = []
    for item in brown_tags:
        pool_Tags += item
    tag_dictionary = collections.Counter(pool_Tags)
    for each_tag in pool_Tags:
        taglist.add(each_tag)
    for each_slice in brown_words_rare:
        words_pool += each_slice
    for i in xrange(len(pool_Tags)):
        if (words_pool[i], pool_Tags[i]) in e_values:
            e_values[(words_pool[i], pool_Tags[i])] += 1
        else:
            e_values[(words_pool[i], pool_Tags[i])] = 1

    e_values.update((key, math.log(value, 2) - math.log(tag_dictionary[key[1]], 2))for key, value in e_values.items())

    return e_values, taglist


# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
def forward(brown_dev_words,taglist, known_words, q_values, e_values):
    probs = []

    for sentence in brown_dev_words:
        each_s = sentence[:]

        # Change the rare word into RARE_SYMBOL when doing computational stuff
        for index in xrange(len(each_s)):
            if each_s[index] not in known_words:
                each_s[index] = RARE_SYMBOL

        each_s[:] = [START_SYMBOL, START_SYMBOL] + each_s + [STOP_SYMBOL]

        forward_dict = {(0, START_SYMBOL, START_SYMBOL): 0}

        # Recursion
        for index in range(1, len(each_s) - 2):
            for b in taglist:
                if (each_s[index], b) in e_values:
                    for c in taglist:
                        if (each_s[index + 1], c) in e_values:
                            forward_dict[(index, b, c)] = 0
                            subsum = 0
                            for a in taglist:
                                if (index - 1, a, b) not in forward_dict or (each_s[index - 1], a) not in e_values:
                                    continue
                                input_sum = forward_dict[(index - 1, a, b)] + (LOG_PROB_OF_ZERO if (a, b, c) not in q_values else q_values[(a, b, c)]) + e_values[(each_s[index + 1], c)]
                                subsum += 2 ** input_sum
                            if subsum:
                                forward_dict[(index, b, c)] = math.log(subsum, 2)
                            else:
                                forward_dict[(index, b, c)] = LOG_PROB_OF_ZERO
        # Terminate
        testtest = 0
        for T_1 in taglist:
            for T in taglist:
                if (len(each_s) - 3, T_1, T) in forward_dict:
                    temp = forward_dict[(len(each_s) - 3, T_1, T)] + (LOG_PROB_OF_ZERO if (T_1, T, STOP_SYMBOL) not in q_values else q_values[(T_1, T, STOP_SYMBOL)])
                    testtest += 2 ** temp
                    forward_dict[(len(each_s) - 3, T_1, T)] = math.log(testtest, 2)
                    output_value = math.log(testtest, 2)

        # Output
        probs.append(str(output_value) + "\n")

    return probs


# This function takes the output of forward() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    for sentence in brown_dev_words:
        each_s = sentence[:]
        original_s = sentence[:]
        # Change the rare word into RARE_SYMBOL when doing computational stuff
        for index in xrange(len(each_s)):
            if each_s[index] not in known_words:
                each_s[index] = RARE_SYMBOL

        viterbi_path = [None] * len(each_s)

        each_s[:] = [START_SYMBOL, START_SYMBOL] + each_s + [STOP_SYMBOL]

        backpointer = {}
        viterbi_dict = {(0, START_SYMBOL, START_SYMBOL):0}

        # Recursion
        for index in range(1, len(each_s) - 2):
            for b in taglist:
                if (each_s[index], b) in e_values:
                    for c in taglist:
                        if (each_s[index + 1], c) in e_values:
                            max_record = -float("inf")
                            viterbi_dict[(index, b, c)] = max_record
                            for a in taglist:
                                if (index - 1, a, b) not in viterbi_dict or (each_s[index - 1], a) not in e_values:
                                    continue
                                temp = viterbi_dict[(index - 1, a, b)] + (LOG_PROB_OF_ZERO if (a, b, c) not in q_values else q_values[(a, b, c)]) + e_values[(each_s[index + 1], c)]
                                if temp > max_record:
                                    max_record = temp
                                    best_tag = a
                            viterbi_dict[(index, b, c)] = max_record
                            backpointer[(index, b, c)] = best_tag

        # Terminate
        terminate_max_record = -float("inf")
        for T_1 in taglist:
            for T in taglist:
                if (len(each_s) - 3, T_1, T) in viterbi_dict:
                    temp = viterbi_dict[(len(each_s) - 3, T_1, T)] + (LOG_PROB_OF_ZERO if (T_1, T, STOP_SYMBOL) not in q_values else q_values[(T_1, T, STOP_SYMBOL)])
                    if temp > terminate_max_record:
                        terminate_max_record = temp
                        viterbi_path[-1] = T
                        viterbi_path[-2] = T_1

        for index in range(len(each_s) - 6, -1, -1):
            viterbi_path[index] = backpointer[(index + 3, viterbi_path[index + 1], viterbi_path[index + 2])]

        # Output
        output_sentence = ""
        for index in xrange(len(viterbi_path)):
            output_sentence += original_s[index] + '/' + viterbi_path[index] + ' '
        tagged.append(output_sentence + "\n")

    return tagged


# This function takes the output of viterbi() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens.
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [zip(brown_words[i], brown_tags[i]) for i in xrange(len(brown_words))]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    default_tagger = nltk.DefaultTagger("NOUN")
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
    for each_s in brown_dev_words:
        temp = trigram_tagger.tag(each_s)
        output_sentence = ""
        for item in temp:
            output_sentence += item[0] + "/" + item[1] + " "
        tagged.append(output_sentence + "\n")
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q7_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 6)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # question 5
    forward_probs = forward(brown_dev_words,taglist, known_words, q_values, e_values)
    q5_output(forward_probs, OUTPUT_PATH + 'B5.txt')

    # do viterbi on brown_dev_words (question 6)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 6 output
    q6_output(viterbi_tagged, OUTPUT_PATH + 'B6.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 7 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B7.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()



