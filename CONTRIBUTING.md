# Contributing to Omni Alpha 12.0

Thank you for your interest in contributing to Omni Alpha 12.0! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Git
- Docker (optional)
- GitHub account

### Setup
1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/Omni_Alpha_12.0.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Copy environment file: `cp .env.example .env`
7. Edit `.env` with your configuration

## 📋 Development Workflow

### Branch Strategy
- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - Feature development branches
- `hotfix/*` - Critical bug fixes
- `release/*` - Release preparation branches

### Creating a Feature
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Write tests for your changes
4. Run tests: `pytest tests/`
5. Run linting: `flake8 backend/` and `black backend/`
6. Commit your changes: `git commit -m "feat: add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

### Commit Message Format
Use conventional commits format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_step1.py

# Run with coverage
pytest --cov=backend tests/

# Run integration tests
pytest tests/integration/
```

### Writing Tests
- Write unit tests for all new functionality
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Mock external dependencies

## 📝 Code Style

### Python Style
- Follow PEP 8
- Use type hints
- Write docstrings for functions and classes
- Keep functions small and focused

### Linting
```bash
# Check code style
flake8 backend/

# Format code
black backend/

# Type checking
mypy backend/
```

## 🏗️ Architecture Guidelines

### Project Structure
```
backend/
├── app/
│   ├── core/           # Core functionality
│   ├── strategies/     # Trading strategies
│   ├── execution/      # Order execution
│   ├── risk/          # Risk management
│   ├── ai_brain/      # AI components
│   ├── institutional/ # Institutional features
│   └── ecosystem/     # Global ecosystem
├── api/               # API endpoints
└── main.py           # Application entry point
```

### Design Principles
- Single Responsibility Principle
- Dependency Injection
- Async/Await for I/O operations
- Error handling and logging
- Configuration management

## 🐛 Bug Reports

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

## 💡 Feature Requests

For feature requests, please:
- Check existing issues first
- Provide clear use case
- Explain the expected behavior
- Consider implementation complexity

## 📚 Documentation

### API Documentation
- Use FastAPI automatic documentation
- Add docstrings to all endpoints
- Include request/response examples
- Document error codes

### Code Documentation
- Write clear docstrings
- Use type hints
- Add inline comments for complex logic
- Update README for major changes

## 🔒 Security

### Security Guidelines
- Never commit secrets or API keys
- Use environment variables for configuration
- Validate all inputs
- Follow secure coding practices
- Report security issues privately

### Reporting Security Issues
Email security issues to: security@omnialpha.com

## 🚀 Release Process

### Version Numbering
- Major.Minor.Patch (e.g., 12.0.1)
- Major: Breaking changes
- Minor: New features
- Patch: Bug fixes

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version bumped
- [ ] Changelog updated
- [ ] Release notes written
- [ ] Tagged and released

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the golden rule

### Getting Help
- Check documentation first
- Search existing issues
- Ask questions in discussions
- Join our community chat

## 📞 Contact

- GitHub: @mbsconstruction007-sys
- Repository: Omni_Alpha_12.0
- Issues: Use GitHub Issues
- Discussions: Use GitHub Discussions

## 📄 License

By contributing to Omni Alpha 12.0, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Omni Alpha 12.0! 🚀
