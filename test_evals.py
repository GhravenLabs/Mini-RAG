import evals


def test_expected_keywords_match_whole_words_and_phrases_case_insensitively():
    passages = [(0.9, "Customer PAYMENT terms are net 30.")]

    assert evals._hit_rank(passages, ["payment terms"]) == 1
    assert evals._hit_rank(passages, ["PAYMENT"]) == 1


def test_expected_keyword_does_not_match_inside_another_word():
    passages = [(0.9, "The invoice remains in repayment status.")]

    assert evals._hit_rank(passages, ["pay"]) == 0
    assert evals._hit_rank(passages, ["payment"]) == 0
