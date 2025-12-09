# Contributing to SynDoc_Wild_v1

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow

## How to Contribute

### Reporting Bugs

1. Check if the bug is already reported in Issues
2. Use the bug report template
3. Provide:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Your environment (OS, Python version, etc.)

### Suggesting Features

1. Check if the feature is already suggested in Issues
2. Use the feature request template
3. Describe:
   - The problem it solves
   - The proposed solution
   - Why it would be useful

### Submitting Code Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes following the code style guidelines
4. Test your changes thoroughly
5. Commit with clear messages: `git commit -m "Add feature: description"`
6. Push to your fork: `git push origin feature/your-feature`
7. Open a Pull Request with a clear description

## Code Style Guidelines

### Python

- Follow PEP 8 guidelines
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use type hints where possible
- Write docstrings for all functions

Example:
```python
def extract_images_from_pdf(pdf_path: Path, output_folder: Path, index: int) -> int:
    """
    Extract images from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Output folder for images
        index: Sequential index for naming
        
    Returns:
        Number of images extracted
    """
    pass
```

### Comments

- Use clear, concise comments
- Explain WHY, not WHAT
- Keep comments up-to-date

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Doc-Dataset-Generate.git
cd Doc-Dataset-Generate

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pylint pytest black

# Format code
black *.py

# Run tests (if available)
pytest
```

## Commit Message Guidelines

Use clear, descriptive commit messages:

```
Format: <type>(<scope>): <subject>

<body>

<footer>

Types:
- feat: A new feature
- fix: A bug fix
- refactor: Code refactoring
- perf: Performance improvement
- docs: Documentation changes
- test: Adding or updating tests
- chore: Build process, dependencies, etc.

Examples:
- feat(dataset): Add GPU acceleration support
- fix(generator): Fix random document repetition issue
- perf(threading): Increase worker pool to 16
- docs: Update README with new features
```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG if applicable
5. Provide clear PR description

## Areas for Contribution

- [ ] Bug fixes
- [ ] Performance improvements
- [ ] Additional shadow types and textures
- [ ] GPU acceleration (CUDA support)
- [ ] Additional degradation models
- [ ] Documentation improvements
- [ ] Test coverage
- [ ] UI/CLI improvements

## Testing

Before submitting, test your changes:

```bash
# Test basic functionality
python extract_pdf_images.py --help
python dataset_generator.py --help

# Test with small dataset
python dataset_generator.py --limit 5 --iterations 1

# Verify output
python verify_mask_accuracy_v2.py
```

## Questions?

- Check existing documentation in README.md
- Review existing issues and discussions
- Open a new discussion if needed

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to make this project better! 🎉
