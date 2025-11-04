# Contributing to Banking SLM-RAG Chatbot

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork and clone**:
   ```bash
   git clone https://github.com/yourusername/banking-slm-rag.git
   cd banking-slm-rag
   ```

2. **Set up development environment**:
   ```bash
   make venv
   source venv/bin/activate
   make index
   ```

3. **Run tests**:
   ```bash
   make test
   ```

## 🔧 Development Workflow

### Making Changes

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation

3. **Test your changes**:
   ```bash
   make test
   make lint
   ```

4. **Commit**:
   ```bash
   git add .
   git commit -m "feat: Add your feature description"
   ```

### Commit Message Format

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding tests
- `refactor:` Code refactoring
- `perf:` Performance improvements

Examples:
- `feat: Add Bengali language support`
- `fix: Correct PII redaction for international phone numbers`
- `docs: Update deployment guide`

## 📝 Code Style

### Python

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use docstrings for functions and classes

Example:
```python
def my_function(param: str, count: int = 5) -> list[str]:
    """
    Brief description.
    
    Args:
        param: Description of param
        count: Description of count
        
    Returns:
        Description of return value
    """
    pass
```

### Linting

```bash
# Auto-format with ruff
ruff check --fix app/

# Type checking
mypy app/ --ignore-missing-imports
```

## 🧪 Testing

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names

Example:
```python
def test_redact_phone_number():
    """Test phone number redaction."""
    text = "Call 9876543210"
    masked, mapping = redact(text)
    assert "PII_PHONE_1" in masked
    assert mapping["PII_PHONE_1"] == "9876543210"
```

### Running Tests

```bash
# All tests
make test

# Specific file
pytest tests/test_redaction.py -v

# With coverage
pytest --cov=app tests/
```

## 📊 Adding Knowledge Base Content

1. **Add chunks** to `data/kb_chunks.sample.jsonl`:
   ```json
   {
     "doc_id": "kb_en_015",
     "title": "Your Topic",
     "content": "Detailed information...",
     "lang": "EN",
     "tags": ["tag1", "tag2"],
     "updated_on": "2025-01-15"
   }
   ```

2. **Rebuild index**:
   ```bash
   make index
   ```

3. **Add evaluation queries** to `eval/prompts.sample.csv`:
   ```csv
   query,expected_doc_id,category
   Your test query?,kb_en_015,your_category
   ```

## 🌐 Adding Languages

1. **Update `app/language.py`**:
   - Add Unicode range detection
   - Add code-mix synonyms

2. **Add KB chunks** in new language

3. **Add tests** in `tests/test_language.py`

4. **Update documentation**

## 🐛 Reporting Issues

### Bug Reports

Include:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Python version)
- Relevant logs/error messages

### Feature Requests

Include:
- Clear description of the feature
- Use case / motivation
- Proposed implementation (if any)

## 🔍 Code Review Process

1. All changes require review
2. Tests must pass
3. Code must follow style guidelines
4. Documentation must be updated
5. No breaking changes without discussion

## 📚 Documentation

When adding features:
- Update README.md if user-facing
- Add docstrings to functions
- Update relevant docs in `index/README.md`, `train/README.md`, etc.
- Add examples

## ⚖️ License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 💬 Questions?

Open an issue for questions about:
- Architecture decisions
- Implementation approaches
- Feature suggestions

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!
