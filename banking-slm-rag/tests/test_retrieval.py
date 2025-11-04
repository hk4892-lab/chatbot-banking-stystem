from app.language import normalize


def test_hinglish_query_hits_expected_doc(retriever) -> None:
    query = 'kitna upi limit hai'
    normalized = normalize(query, 'HI')
    results = retriever.search(normalized, 3)
    assert results[0].chunk.doc_id == 'kb_hi_011'
    assert results[0].score > 0.2


def test_tamlish_query_hits_expected_doc(retriever) -> None:
    query = 'card block pannunga'
    normalized = normalize(query, 'TA')
    results = retriever.search(normalized, 3)
    top_ids = [res.chunk.doc_id for res in results]
    assert 'kb_ta_023' in top_ids
    for res in results:
        if res.chunk.doc_id == 'kb_ta_023':
            assert res.score > 0.2
