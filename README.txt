UNI: ds3420
Name: Dailin Shen

———————————————Assignment Part A - Language Model———————————————

(1)

UNIGRAM near -12.4560686964

BIGRAM near the -1.56187888761

TRIGRAM near the ecliptic -5.39231742278

(2)

UNIGRAM
The perplexity is 1052.4865859

BIGRAM
The perplexity is 53.8984761198

TRIGRAM
The perplexity is 5.7106793082

(3)

The perplexity is 12.5516094886

(4)

[A]: From the observations, the perplexity of models with linear interpolation is better than that of unigram and bigram. Generally, one expects higher performance of models with linear interpolation. However, it did no better than trigram in the given case. One guess is that in the given case, we assigned lambdas equally, which in practice, are learned from a held-out corpus and the lambdas for trigrams are usually higher.

(5)

Sample1_scored
The perplexity is 1.54761961801

Sample2_scored
The perplexity is 7.32048000699

[Argument]: Sample1.txt is an excerpt of Brown_train.txt while Sample2.txt is not. This is because the perplexity of Sample1_scored.txt is lower, indicating its texts are more similar to the brown dataset.

———————————————Assignment Part B - Part-of-Speech Tagging————————————————

(2)

TRIGRAM CONJ ADV NOUN -4.46650366731

TRIGRAM DET NUM NOUN -0.713200128516

TRIGRAM NOUN PRT CONJ -6.38503274104

(4)

* * 0.0

midnight NOUN -13.1814628813

Place VERB -15.4538814891

primary ADJ -10.0668014957

STOP STOP 0.0

_RARE_ VERB -3.17732085089

_RARE_ X -0.546359661497

(5)

The probability for the first sentence in log space is: -198.906734204

(6)

Percent correct tags: 93.3249946254

(7)

Percent correct tags: 87.9985146677


———————————————Running Time———————————————

Part A time: 16.551129 sec
Part B time: 49.70847 sec

