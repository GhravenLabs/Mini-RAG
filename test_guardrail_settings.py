import pytest

import finance_rag


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -0.1, 1.1, True])
def test_invalid_score_threshold_rejected(score):
    index = finance_rag.build_index('Invoices require approval before payment.')
    with pytest.raises(ValueError, match='min_score'):
        finance_rag.answer('invoices payment', index, min_score=score)


@pytest.mark.parametrize('count', [-1, 0.5, True])
def test_invalid_evidence_threshold_rejected(count):
    index = finance_rag.build_index('Invoices require approval before payment.')
    with pytest.raises(ValueError, match='min_evidence_terms'):
        finance_rag.answer('invoices payment', index, min_evidence_terms=count)


def test_zero_thresholds_allow_matches_but_not_absent_evidence():
    index = finance_rag.build_index('Invoices require approval before payment.')
    assert finance_rag.answer('invoices', index, min_score=0, min_evidence_terms=0)['grounded']
    assert not finance_rag.answer('vacation', index, min_score=0, min_evidence_terms=0)['grounded']
