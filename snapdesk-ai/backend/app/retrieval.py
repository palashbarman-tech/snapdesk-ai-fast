import math
import re
from collections import Counter

STOP = set(
    """a an the and or of to in on for with is are was were be been being it its this that these those as at by from
    but not no if then than so such can could should would will shall may might do does did done has have had i you he
    she we they them his her our your their my me what which who whom how when where why also about into over under
    है हैं था थे थी का की के में और से को पर यह वह एक भी तो कि जो""".split()
)
SPLIT = re.compile(r"[\s.,;:!?()\[\]{}\"'“”‘’/\\|<>=+*&^%$#@~`।॥—–-]+")
SENTENCES = re.compile(r"(?<=[.!?।])\s+|\n{2,}")


def tokenize(text):
    return [t for t in SPLIT.split(text.lower()) if len(t) > 1 and t not in STOP]


def split_sentences(text):
    return [s.strip() for s in SENTENCES.split(text) if s and s.strip()]


class BM25:
    def __init__(self, token_lists, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.counts = [Counter(tokens) for tokens in token_lists]
        self.lengths = [len(tokens) for tokens in token_lists]
        self.average = sum(self.lengths) / max(len(token_lists), 1) or 1
        frequency = Counter()
        for tokens in token_lists:
            frequency.update(set(tokens))
        total = len(token_lists)
        self.idf = {t: math.log(1 + (total - n + 0.5) / (n + 0.5)) for t, n in frequency.items()}

    def scores(self, query_tokens):
        result = []
        for counts, length in zip(self.counts, self.lengths):
            score = 0.0
            for token in query_tokens:
                if token not in counts:
                    continue
                tf = counts[token]
                norm = tf + self.k1 * (1 - self.b + self.b * length / self.average)
                score += self.idf[token] * tf * (self.k1 + 1) / norm
            result.append(score)
        return result
