def test_health_endpoint(api_client) -> None:
    response = api_client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_chat_turn_schema(api_client) -> None:
    payload = {'text': 'How to get account statement?', 'k': 3, 'lang': 'AUTO', 'temperature': 0.0}
    response = api_client.post('/chat/turn', json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 'reply' in body
    assert isinstance(body['citations'], list)
    assert body['latency_ms'] >= 0
    if body['route'] == 'answer':
        assert body['citations']
