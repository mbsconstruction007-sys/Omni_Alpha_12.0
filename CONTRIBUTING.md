# Contributing to Omni Alpha 5.0

Thank you for your interest in contributing to Omni Alpha 5.0! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment (venv or conda)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/mbsconstruction007-sys/omni_alpha_5.0.git
   cd omni_alpha_5.0
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
   ```

## 📋 Development Workflow

### Branch Strategy
- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - Feature development branches
- `hotfix/*` - Critical bug fixes

### Commit Convention
We follow conventional commits:
```
type(scope): description

feat: add new feature
fix: bug fix
docs: documentation changes
style: formatting changes
refactor: code refactoring
test: add or update tests
chore: maintenance tasks
```

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   python check_step4_endpoints.py
   python check_step7_webhook.py
   python check_step8_advice.py
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Use the PR template
   - Provide clear description
   - Link related issues

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python check_step4_endpoints.py
python check_step7_webhook.py
python check_step8_advice.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage
- Maintain test coverage above 80%
- Write unit tests for new features
- Include integration tests for API endpoints

## 📝 Code Style

### Python Style Guide
- Follow PEP 8
- Use type hints
- Write docstrings for functions and classes
- Keep functions small and focused

### Example Code Style
```python
def calculate_progress(completed_steps: int, total_steps: int) -> float:
    """
    Calculate the progress percentage of analysis steps.
    
    Args:
        completed_steps: Number of completed steps
        total_steps: Total number of steps
        
    Returns:
        Progress percentage as float
    """
    if total_steps == 0:
        return 0.0
    return (completed_steps / total_steps) * 100
```

## 🐛 Bug Reports

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Screenshots if applicable

## 💡 Feature Requests

For feature requests:
- Check existing issues first
- Provide clear use case
- Explain the expected behavior
- Consider implementation complexity

## 📚 Documentation

### API Documentation
- Update API documentation for new endpoints
- Include request/response examples
- Document error codes and messages

### Code Documentation
- Write clear docstrings
- Add inline comments for complex logic
- Update README for new features

## 🔒 Security

- Never commit sensitive information
- Use environment variables for configuration
- Follow security best practices
- Report security issues privately

## 📞 Getting Help

- Check existing issues and discussions
- Join our community discussions
- Contact maintainers for urgent issues

## 🎯 Project Roadmap

### Current Phase (Steps 1-45) ✅
- Core API implementation
- Dashboard interface
- Testing framework
- Basic deployment

### Next Phase (Steps 46-60) 🔄
- User authentication
- Data persistence
- Advanced analytics
- Production deployment

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Omni Alpha 5.0! 🚀
