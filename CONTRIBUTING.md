
# Contributing to RCT-Reviewer

First off, thank you for taking the time to contribute to **RCT-Reviewer**! Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

This project is a modernized, standalone version of RobotReviewer, and we welcome efforts to improve its accuracy, usability, and extensensible.

## Code of Conduct

This project and everyone participating in it is governed by the [RCT-Reviewer Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the repository maintainer.

## 🛠️ How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you are creating a bug report, please include as many details as possible:

*   **Use a clear and descriptive title**.
*   **Describe the exact steps to reproduce the problem**.
*   **Provide specific examples** (e.g., a specific PDF file that causes a crash).
*   **Describe the behavior you observed and the behavior you expected**.
*   **Include details about your environment**: OS, Python version (remember this project targets **Python 3.12**), and dependency versions.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

*   **Use a clear and descriptive title**.
*   **Provide a step-by-step description of the suggested enhancement**.
*   **Describe the current behavior** and explain which behavior you expected to see instead.
*   **Explain why this enhancement would be useful** to most RCT-Reviewer users.

### Pull Requests

1.  **Fork the repository** and create your branch from `main`.
2.  **Ensure the test suite passes** (if applicable).
3.  **Make sure your code lints** (follows PEP 8 standards).
4.  **Issue that pull request**!

## 🐍 Development Environment Setup

To set up a local development environment, please follow the installation instructions in the [README.md](README.md).

**Crucial Note:** Ensure you are using **Python 3.12**. Using a different version may cause compatibility issues with the ML models or dependencies.

### Prerequisites

*   Python 3.12
*   Git & Git LFS (if working with local model weights)

### Local Setup

```bash
# Clone your fork
git clone https://github.com/your-username/RCT-Reviewer.git
cd RCT-Reviewer

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download NLP model
python -m spacy download en_core_web_sm

# Pull model weights (only needed for app.py / app1.py)
git lfs pull
```

## 📁 Project Structure

Please try to maintain the existing project structure:

*   `rct_reviewer/`: Main application code (UI, processing logic).
*   `data/`: Model weights (handled by Git LFS).
*   `tests/`: Unit tests.

---

Please make sure you do not commit model data files in pull requests!

Thank you again for your contribution! ❤️
