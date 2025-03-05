#!/bin/bash
cd "$(dirname "$0")"

# Define help function
function show_help {
  echo "Expense Tracker - Options:"
  echo "  --review: Enable interactive review of low-confidence transactions"
  echo "  --skip-normalization: Skip LLM-based description normalization (faster)"
  echo "  --skip-llm-dedup: Skip LLM-based deduplication (faster)"
  echo "  --skip-rule-gen: Skip automatic rule generation"
  echo "  --fast: Run in fast mode (skips all LLM features)"
  echo "  --help: Show this help message"
  echo ""
  echo "Example: ./expense_tracker.command --review"
}

# Check for help flag
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  show_help
  read -p "Press Enter to close..."
  exit 0
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
  echo "Warning: .env file not found. Please create one from .env.example"
  echo "cp .env.example .env"
  echo "Then edit .env to add your OpenAI API key"
  read -p "Press Enter to close..."
  exit 1
fi

# Run the expense tracker with any provided arguments
/opt/homebrew/opt/python@3.11/bin/python3.11 expense_tracker.py "$@"

read -p "Press Enter to close..."