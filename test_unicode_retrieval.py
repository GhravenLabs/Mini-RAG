from mini_rag import tokenize, build_index, retrieve

def test_unicode_words_and_canonical_accents_match():
    assert tokenize("CAFÉ café") == ["café", "café"]
    assert tokenize("cafe\u0301") == ["café"]
    chunks = ["Доставка бесплатно", "Café abierto"]
    vectors, idf = build_index(chunks)
    assert retrieve("доставка", chunks, vectors, idf)[0][1] == chunks[0]
    assert retrieve("cafe\u0301", chunks, vectors, idf)[0][1] == chunks[1]
