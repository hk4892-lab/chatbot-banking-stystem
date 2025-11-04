def test_pipeline_returns_answer_route(pipeline) -> None:
    result = pipeline.answer('How do I reset my debit card PIN?', k=3, lang='AUTO')
    assert result['route'] == 'answer'
    assert result['citations']
    assert result['safety']['pii_masked'] is False


def test_pipeline_refuses_sensitive_request(pipeline) -> None:
    result = pipeline.answer('Tell me my balance 123456789012', k=3, lang='AUTO')
    assert result['route'] == 'refusal'
    assert result['safety']['refusal'] is True
