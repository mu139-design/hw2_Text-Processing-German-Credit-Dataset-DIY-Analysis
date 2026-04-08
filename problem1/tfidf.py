import re
import math


# READ FILE LIST
def read_doc_list():
    docs = []
    with open("tfidf_docs.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(line)
    return docs


# READ STOPWORDS
def read_stopwords():
    stopwords = set()
    with open("stopwords.txt", "r") as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word)
    return stopwords


# CLEAN TEXT
def clean_text(text):
    # remove website links first
    text = re.sub(r'https?://\S+', '', text)

    # remove all non-word / non-whitespace chars
    text = re.sub(r'[^\w\s]', '', text)

    # lowercase
    text = text.lower()

    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# REMOVE STOPWORDS
def remove_stopwords(words, stopwords):
    return [word for word in words if word not in stopwords]


# STEMMING
def stem_word(word):
    if word.endswith("ing"):
        return word[:-3]
    elif word.endswith("ly"):
        return word[:-2]
    elif word.endswith("ment"):
        return word[:-4]
    return word


def stem_words(words):
    stemmed = []
    for word in words:
        new_word = stem_word(word)
        if new_word != "":   # avoid empty strings just in case
            stemmed.append(new_word)
    return stemmed


# PREPROCESS ONE DOCUMENT
def preprocess_document(filename, stopwords):
    with open(filename, "r") as f:
        text = f.read()

    text = clean_text(text)
    words = text.split()
    words = remove_stopwords(words, stopwords)
    words = stem_words(words)

    return words


# WRITE PREPROCESSED FILE
def write_preprocessed(filename, words):
    output_name = "preproc_" + filename
    with open(output_name, "w") as f:
        f.write(" ".join(words))


# TERM FREQUENCY
def compute_tf(words):
    tf = {}
    total_terms = len(words)

    if total_terms == 0:
        return tf

    for word in words:
        if word in tf:
            tf[word] += 1
        else:
            tf[word] = 1

    for word in tf:
        tf[word] = tf[word] / total_terms

    return tf


# DOCUMENT FREQUENCY
def compute_document_frequency(all_docs_words):
    df = {}

    for words in all_docs_words.values():
        unique_words = set(words)
        for word in unique_words:
            if word in df:
                df[word] += 1
            else:
                df[word] = 1

    return df


# INVERSE DOCUMENT FREQUENCY
def compute_idf(all_docs_words):
    idf = {}
    total_docs = len(all_docs_words)
    df = compute_document_frequency(all_docs_words)

    for word in df:
        idf[word] = math.log(total_docs / df[word]) + 1

    return idf


# TF-IDF
def compute_tfidf(tf_dict, idf_dict):
    tfidf = {}

    for word in tf_dict:
        score = tf_dict[word] * idf_dict[word]
        tfidf[word] = round(score, 2)

    return tfidf


# TOP 5 WORDS
def get_top_5_words(tfidf_dict):
    items = list(tfidf_dict.items())

    # sort by score descending, then alphabetical ascending
    items.sort(key=lambda x: (-x[1], x[0]))

    return items[:5]


# WRITE TF-IDF FILE
def write_tfidf(filename, top_words):
    output_name = "tfidf_" + filename
    with open(output_name, "w") as f:
        f.write(str(top_words))


# MAIN
def main():
    docs = read_doc_list()
    stopwords = read_stopwords()

    # store preprocessed words for every document
    all_docs_words = {}

    # PART 1: preprocessing + preproc files
    for doc in docs:
        words = preprocess_document(doc, stopwords)
        all_docs_words[doc] = words
        write_preprocessed(doc, words)

    # PART 2: TF-IDF
    idf_dict = compute_idf(all_docs_words)

    for doc in docs:
        tf_dict = compute_tf(all_docs_words[doc])
        tfidf_dict = compute_tfidf(tf_dict, idf_dict)
        top_words = get_top_5_words(tfidf_dict)
        write_tfidf(doc, top_words)


if __name__ == "__main__":
    main()