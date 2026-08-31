import finance_rag


DOC = """
## Payment terms

Standard customer invoices are due on net 30 payment terms.

## Expense claims

Expense claims require a receipt and must be submitted within 30 days.
"""


def test_answer_reports_evidence_terms_for_grounded_question():
    index = finance_rag.build_index(DOC)

    result = finance_rag.answer("What are the standard payment terms?", index)

    assert result["grounded"] is True
    assert "payment" in result["evidence_terms"]
    assert "terms" in result["evidence_terms"]
    assert result["evidence_coverage"] > 0


def test_answer_refuses_when_retrieval_lacks_enough_question_evidence():
    index = finance_rag.build_index(DOC)

    result = finance_rag.answer(
        "What vacation policy applies?",
        index,
        min_score=0.0,
        min_evidence_terms=2,
    )

    assert result["grounded"] is False
    assert len(result["evidence_terms"]) < 2
