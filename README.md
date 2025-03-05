# Expense Tracker

An AI-powered Python tool that consolidates financial transactions from multiple bank exports (PDFs and CSVs) into a categorized expense report. Drop your exported files into the `inputs/` directory, run the script, and get an intelligent expense analysis with automatic categorization.

## Features

- **Bank Support**: Processes statements from Bank of America (BoA), Apple Card, Citi, and Stanford Federal Credit Union (SFCU).
- **File Types**: Handles PDFs (BoA, Apple, Citi) and CSVs (SFCU, BoA).
- **AI Categorization**: Uses GPT-4o to intelligently categorize transactions based on description patterns.
- **Description Normalization**: Standardizes transaction descriptions for better matching and categorization.
- **Smart Deduplication**: Identifies and removes duplicate transactions even when they appear different.
- **Custom Rules**: Learns from your manual categorizations to create rules for future transactions.
- **Interactive Review**: Enables review and correction of low-confidence categorizations.
- **Progress Tracking**: Shows real-time progress with ETA estimates and detailed statistics.
- **Checkpointing**: Automatically saves progress to recover from interruptions.
- **Robust Error Handling**: Detailed logging and fallback mechanisms for LLM failures.

## Project Structure

```
expense_tracker/
├── inputs/                   # Drop your bank export files here
│   ├── eStmt_2025-02-09.pdf  # BoA PDF
│   ├── Apple Card Statement.pdf  # Apple Card PDF
│   ├── February 04.pdf       # Citi PDF
│   └── AccountHistory.csv    # SFCU CSV
├── outputs/                  # Generated output files
│   ├── combined_expenses.csv    # All transactions with categories
│   └── deduplicated_expenses.csv # Final deduplicated output
├── cache/                    # Caching directory for LLM responses
├── expense_tracker.py        # Main Python script
├── expense_tracker.command   # Double-clickable executable (Mac)
├── categories.json           # Custom categorization rules
├── expense_tracker.log       # Detailed log file
└── README.md                 # This file
```

## Prerequisites

- **Python 3.8+**: Required for modern language features.
- **OpenAI API Key**: Required for AI categorization (set as environment variable or in script).
- **Libraries**: Install via pip:

  ```bash
  python3 -m pip install pdfplumber pandas openai tenacity python-dotenv --user
  ```

- **Mac OS**: The .command file is Mac-specific. For other OS, run python3 expense_tracker.py manually.

## Environment Setup

1. Copy the `.env.example` file to a new file named `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

3. The application will automatically load environment variables from this file.

## Privacy Note

This project is designed with privacy in mind:
- All files in the `inputs/` and `outputs/` directories are excluded from git via `.gitignore`
- Your financial data will never be committed to the repository
- Your OpenAI API key in the `.env` file is also excluded from git

## Usage

1. **Export Files**:
   - Download statements from your bank websites.
   - Place them in `inputs/` directory without renaming.

2. **Run the Script**:
   - Double-click `expense_tracker.command` in Finder, or
   - From Terminal: `cd ~/dev/expense_tracker && python3 expense_tracker.py`

3. **Command Line Options**:
   ```
   python3 expense_tracker.py [options]
   
   Options:
     --review             Enable interactive review of low-confidence transactions
     --skip-normalization Skip LLM-based description normalization (faster)
     --skip-llm-dedup     Skip LLM-based deduplication (faster)
     --skip-rule-gen      Skip automatic rule generation
     --fast               Run in fast mode (skips all LLM features)
     --verbose            Show detailed progress information
     --view-logs          View recent log entries
     --log-lines N        Number of log lines to view (default: 50)
     --log-level LEVEL    Filter logs by level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
     --log-contains TEXT  Filter logs by content
   ```

4. **Check Output**:
   - `outputs/combined_expenses.csv`: All transactions with categories
   - `outputs/deduplicated_expenses.csv`: Final deduplicated output

## Output Format

The final `deduplicated_expenses.csv` includes:

- **Date**: Transaction date (MM/DD/YYYY)
- **Description**: Original transaction description
- **NormalizedDesc**: Standardized description for better matching
- **Amount**: Transaction amount
- **Source**: Bank name (BoA, Apple, Citi, SFCU)
- **Category**: AI-determined expense category
- **Confidence**: AI confidence score (0-1)
- **Reasoning**: AI reasoning for the categorization
- **IsDuplicate**: Flag for identifying duplicates
- **DedupeReasoning**: Explanation for duplication decisions

## Example Output:

```
Date,Description,NormalizedDesc,Amount,Source,Category,Confidence,Reasoning
01/14/2025,BA ELECTRONIC PAYMENT,Bank of America Payment,-664.05,BoA,Payments,1.0,Matched hardcoded rule
02/05/2025,NETFLX.COM 121 Abright Way,Netflix,12.19,Apple,Entertainment,0.95,Subscription service for streaming content
01/03/2025,UBER *EATS 8005928996 CA,Uber Eats,41.91,Citi,Food,0.92,Food delivery service
03/03/2025,ACH Debit VENMO - PAYMENT,Venmo Payment,850.00,SFCU,Rent,0.87,Regular monthly payment amount matches rent pattern
```

## Custom Categories

The system uses predefined categories that can be extended:
- Entertainment
- Food
- Payments
- Shopping
- Travel
- Fees
- Subscriptions
- Insurance
- Investments
- Rent
- Uncategorized (default for uncertain transactions)

To add custom rules, edit the `categories.json` file or use the interactive review feature.

## Troubleshooting

- **View Logs**: `python3 expense_tracker.py --view-logs` to see detailed logs
- **Debug JSON Issues**: Check the log file for detailed JSON parsing errors
- **Performance Issues**: Use `--fast` mode to skip LLM-intensive features
- **Interrupted Process**: Just run the script again - it will pick up where it left off

## Performance Notes

- First run will be slower as AI responses are cached for future use
- Subsequent runs will be faster as normalized descriptions and categories are reused
- For large statement batches, expect ~1-2 seconds per transaction for first-time processing