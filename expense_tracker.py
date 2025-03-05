import os
import pdfplumber
import pandas as pd
import json
import argparse
import hashlib
import pickle
import time
import signal
import sys
import logging
from datetime import datetime
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_log, after_log
from dotenv import load_dotenv

# Set up logging
LOG_FILE = "expense_tracker.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create a logger for this module
logger = logging.getLogger("expense_tracker")

# Add console handler to show important logs in the terminal too
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
console_formatter = logging.Formatter("%(levelname)s: %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

logger.info("Starting Expense Tracker")

# Set up OpenAI API
# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = "YOUR_OPENAI_API_KEY"  # Replace with your key or set as env var
client = OpenAI(api_key=openai_api_key)
logger.info("OpenAI client initialized")

# Directory setup
INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
CACHE_DIR = "cache"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "combined_expenses.csv")
DEDUPE_CSV = os.path.join(OUTPUT_DIR, "deduplicated_expenses.csv")
CACHE_FILE = os.path.join(CACHE_DIR, "llm_cache.pkl")
NORMALIZE_CACHE_FILE = os.path.join(CACHE_DIR, "normalize_cache.pkl")
CATEGORIZE_CACHE_FILE = os.path.join(CACHE_DIR, "categorize_cache.pkl")

# In-memory caches
LLM_CACHE = {}  # Generic LLM request cache
NORMALIZE_CACHE = {}  # Description normalization cache
CATEGORIZE_CACHE = {}  # Transaction categorization cache

# Ensure directories exist
for directory in [OUTPUT_DIR, CACHE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def safe_json_parse(content, fallback={"category": "Uncategorized", "confidence": 0.0, "reasoning": "Invalid JSON"}, context=None):
    """
    Safely parse JSON from LLM responses with robust error handling and detailed logging.
    
    Args:
        content: String content from LLM that should contain JSON
        fallback: Default value to return if parsing fails
        context: Optional context information for logging (e.g., transaction description)
        
    Returns:
        Parsed JSON object or fallback value
    """
    context_str = f" for {context}" if context else ""
    
    try:
        # Log the original content for debugging
        logger.debug(f"Attempting to parse JSON{context_str}. Raw content: {content[:200]}...")
        
        # Strip code fences or extra text
        original_content = content
        content = content.strip().replace("```json", "").replace("```", "")
        
        # Extract JSON if embedded in text
        json_extracted = False
        if not content.startswith("{") and not content.startswith("["):
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            
            # Try array extraction if object extraction fails
            if start_idx < 0 or end_idx <= start_idx:
                start_idx = content.find("[")
                end_idx = content.rfind("]")
                
            if start_idx >= 0 and end_idx > start_idx:
                extracted_content = content
                content = content[start_idx:end_idx+1]
                json_extracted = True
                logger.debug(f"Extracted JSON from text{context_str}: {content[:100]}...")
        
        # Log extraction details if applicable
        if json_extracted:
            logger.debug(f"JSON extraction details{context_str}:")
            logger.debug(f"- Original length: {len(original_content)}")
            logger.debug(f"- Extracted from index {start_idx} to {end_idx}")
            logger.debug(f"- Extracted content length: {len(content)}")
        
        # Try to parse the JSON
        parsed = json.loads(content)
        
        # Log success
        logger.debug(f"Successfully parsed JSON{context_str}: {parsed}")
        return parsed
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing failed{context_str}: {e}")
        logger.warning(f"Problematic content{context_str}: {content[:200]}...")
        
        # Log detailed error information
        if context:
            logger.error(f"JSON parse error details for {context}:")
        else:
            logger.error("JSON parse error details:")
            
        logger.error(f"- Error message: {str(e)}")
        logger.error(f"- Error position: line {e.lineno}, column {e.colno}")
        logger.error(f"- Error document: '{e.doc}'")
        logger.error(f"- Full content: {content}")
        
        # Return fallback value
        return fallback

# Configure retry logger
retry_logger = logging.getLogger("tenacity")

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_log(retry_logger, logging.WARNING),
    after=after_log(retry_logger, logging.INFO)
)
def call_llm(prompt, model="gpt-4o", max_tokens=100, temperature=0.2, context=None):
    """
    Call LLM with automatic retry logic for failed requests and detailed logging.
    
    Args:
        prompt: The prompt to send to the LLM
        model: The model to use (default: gpt-4o)
        max_tokens: Maximum tokens in the response
        temperature: Temperature setting (0-1)
        context: Optional context information for logging
        
    Returns:
        OpenAI API response
    """
    context_str = f" for {context}" if context else ""
    
    # Log basic information about the request
    logger.info(f"Calling LLM{context_str}: model={model}, max_tokens={max_tokens}, temperature={temperature}")
    
    # Log a truncated version of the prompt for debugging
    truncated_prompt = prompt[:200] + "..." if len(prompt) > 200 else prompt
    logger.debug(f"LLM prompt{context_str}: {truncated_prompt}")
    
    try:
        # Start timing the request
        start_time = time.time()
        
        # Make the API call
        response = client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, 
            temperature=temperature
        )
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log success
        logger.info(f"LLM call succeeded{context_str} in {duration:.2f}s")
        
        # Log response details
        truncated_response = response.choices[0].message.content[:200] + "..." if len(response.choices[0].message.content) > 200 else response.choices[0].message.content
        logger.debug(f"LLM response{context_str}: {truncated_response}")
        
        return response
        
    except Exception as e:
        # Log the error
        logger.error(f"LLM call failed{context_str}: {str(e)}")
        
        # Add prompt details for debugging
        logger.error(f"Failed LLM prompt{context_str}: {prompt}")
        
        # Re-raise the exception to trigger retry
        raise
        
# Load existing caches if available
def load_cache(cache_file, default=None):
    """Load cache from disk if it exists."""
    if default is None:
        default = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache from {cache_file}: {e}")
    return default

def save_cache(cache_data, cache_file):
    """Save cache to disk."""
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"Error saving cache to {cache_file}: {e}")

# Load caches
NORMALIZE_CACHE = load_cache(NORMALIZE_CACHE_FILE)
CATEGORIZE_CACHE = load_cache(CATEGORIZE_CACHE_FILE)

# Hardcoded category rules
CATEGORY_RULES = {
    "Entertainment": ["netflx", "steamgames", "openai", "claude.ai", "cursor", "medium", "x corp", "uber *one", 
                      "hbo max", "disney+", "spotify", "hulu", "paramount+", "apple tv", "prime video", "cinema"],
    "Food": ["uber *eats", "starbucks", "tacos", "restaurant", "rest ", "coffee", "cafe", "doss av", 
             "doordash", "grubhub", "mcdonald", "chipotle", "taco bell", "grocery", "safeway", "trader joe",
             "whole foods", "sushi", "thai", "mexican", "chinese food"],
    "Payments": ["autopay", "bill pay", "online banking", "bank transfer", "wire transfer"],  # More specific
    "Shopping": ["amazon", "fred s place", "plantation", "texttra", "target", "walmart", "costco", "best buy", 
                "ebay", "etsy", "ikea", "home depot", "clothing", "shoes", "apparel"],
    "Travel": ["uber *trip", "hotel", "airbnb", "airline", "flight", "delta air", "united air", "southwest air", 
              "car rental", "hertz", "enterprise", "train", "amtrak", "expedia", "booking.com"],
    "Fees": ["fee", "foreign transaction", "atm fee", "overdraft", "service charge", "annual fee", "late fee",
            "finance charge", "interest charge", "penalty fee"],
    "Subscriptions": ["chatgpt subscr", "medium monthly", "subscription", "monthly", "yearly", "recurring", 
                     "membership", "access fee", "premium", "plus plan"],
    "Insurance": ["csaa ig", "aaa paymnt", "insurance", "premium", "coverage", "policy", "geico", "state farm",
                 "progressive", "allstate", "health insurance", "auto insurance", "life insurance"],
    "Investments": ["robinhood", "coinbase", "wealthfront", "vanguard", "fidelity", "schwab", "etrade", 
                   "td ameritrade", "crypto", "bitcoin", "ethereum", "401k", "ira", "investment"],
    "Rent": ["venmo - payment", "rent", "mortgage", "apartment", "property", "landlord", "housing", "lease", 
            "zelle", "real estate", "management", "leasing", "home payment", "condo fee"]  # Expanded rent category
}

# Load custom rules from JSON
try:
    with open(os.path.join(os.path.dirname(__file__), "categories.json"), "r") as f:
        CUSTOM_RULES = json.load(f)["rules"]
except FileNotFoundError:
    CUSTOM_RULES = []

def normalize_description(desc):
    """Simplify description for duplicate matching."""
    desc = str(desc).lower().strip()
    for keyword in ["ach debit", "ach credit", "online", "payment", "*"]:
        desc = desc.replace(keyword, "")
    return " ".join(desc.split())

def normalize_with_llm(description):
    """Use LLM to standardize transaction descriptions into a concise, consistent format, with caching."""
    # Create a cache key
    cache_key = hashlib.md5(description.encode()).hexdigest()
    
    # Check if we have this description in cache
    if cache_key in NORMALIZE_CACHE:
        # Track cache hits for reporting
        NORMALIZE_CACHE[cache_key]['hits'] += 1
        return NORMALIZE_CACHE[cache_key]['result']
    
    # Not in cache, so normalize with LLM
    prompt = f"""
    Standardize this transaction description into a concise, consistent format:
    - Remove unnecessary details (dates, transaction IDs, etc.)
    - Standardize vendor names (e.g., "UBER *EATS 8005928996 CA" → "Uber Eats")
    - Keep only relevant identifying information
    - Use title case for proper nouns, lowercase for others
    - Maximum 5 words
    
    Input: '{description}'
    Return only the normalized description without quotes or explanations.
    """
    try:
        # Use our retry-enabled function
        response = call_llm(
            prompt,
            model="gpt-4o",
            max_tokens=50,
            temperature=0.1
        )
        result = response.choices[0].message.content.strip()
        
        # Cache the result
        NORMALIZE_CACHE[cache_key] = {
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'original': description,
            'hits': 1
        }
        
        # Periodically save cache to disk (every 10 new entries)
        if len(NORMALIZE_CACHE) % 10 == 0:
            save_cache(NORMALIZE_CACHE, NORMALIZE_CACHE_FILE)
            
        return result
    except Exception as e:
        print(f"Error normalizing description '{description}': {e}")
        # Fall back to basic normalization if LLM fails
        basic_result = normalize_description(description)
        
        # Cache the fallback result too
        NORMALIZE_CACHE[cache_key] = {
            'result': basic_result,
            'timestamp': datetime.now().isoformat(),
            'original': description,
            'hits': 1,
            'error': str(e)
        }
        
        return basic_result

def normalize_descriptions_batch(descriptions):
    """Process multiple descriptions in a single LLM call with detailed logging."""
    # Log the batch processing request
    logger.info(f"Normalizing batch of {len(descriptions)} descriptions")
    
    # First check cache for each description
    results = {}
    descriptions_to_process = []
    indices_to_process = []
    
    # Track metrics for logging
    cache_hits = 0
    cache_misses = 0
    
    for i, desc in enumerate(descriptions):
        # Create a cache key
        cache_key = hashlib.md5(desc.encode()).hexdigest()
        
        # Check if we have this description in cache
        if cache_key in NORMALIZE_CACHE:
            # Track cache hits for reporting
            NORMALIZE_CACHE[cache_key]['hits'] += 1
            results[i] = NORMALIZE_CACHE[cache_key]['result']
            cache_hits += 1
            
            # Log cache hit at debug level
            logger.debug(f"Cache hit for description: {desc[:30]}... -> {results[i]}")
        else:
            descriptions_to_process.append(desc)
            indices_to_process.append(i)
            cache_misses += 1
    
    # Log cache statistics
    if len(descriptions) > 0:
        hit_rate = (cache_hits / len(descriptions)) * 100
        logger.info(f"Normalization cache: {cache_hits} hits, {cache_misses} misses ({hit_rate:.1f}% hit rate)")
    
    # If there are descriptions not in cache, process them in a batch
    if descriptions_to_process:
        logger.info(f"Processing {len(descriptions_to_process)} descriptions in batch")
        prompt = f"""
        Standardize each of these transaction descriptions into concise, consistent formats.
        For each description:
        - Remove unnecessary details (dates, transaction IDs, etc.)
        - Standardize vendor names
        - Keep only relevant identifying information
        - Use title case for proper nouns, lowercase for others
        - Maximum 5 words for each
        
        CRITICAL: You MUST return EXACTLY {len(descriptions_to_process)} lines, one normalized description per line.
        - NO JSON formatting
        - NO numbers or bullets
        - NO explanations or commentary
        - Just the plain text of each normalized description
        
        Input descriptions:
        {json.dumps(descriptions_to_process)}
        
        YOUR RESPONSE FORMAT MUST BE EXACTLY {len(descriptions_to_process)} LINES, ONE NORMALIZED DESCRIPTION PER LINE.
        """
        
        # Create a context identifier for logs
        context_id = f"norm_batch:{len(descriptions_to_process)}"
        
        try:
            logger.info(f"Calling LLM for batch normalization of {len(descriptions_to_process)} descriptions")
            
            # Log a sample of descriptions being processed
            if len(descriptions_to_process) > 0:
                sample_size = min(3, len(descriptions_to_process))
                sample_descriptions = descriptions_to_process[:sample_size]
                logger.debug(f"Sample descriptions to normalize: {sample_descriptions}")
            
            # Use our retry-enabled function with context
            response = call_llm(
                prompt,
                model="gpt-4o",
                max_tokens=300,
                temperature=0.1,
                context=context_id
            )
            
            # Use simple line-by-line format instead of JSON
            content = response.choices[0].message.content.strip()
            
            # Log the raw response for debugging
            logger.debug(f"Batch normalization raw response {context_id}: {content[:200]}...")
            
            # Split by lines and clean up
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Log the parsed lines
            logger.debug(f"Parsed {len(lines)} lines from batch normalization response")
            
            # Ensure exact match to input size
            if len(lines) != len(descriptions_to_process):
                mismatch_msg = f"Batch mismatch: Expected {len(descriptions_to_process)}, got {len(lines)}. Adjusting."
                logger.warning(mismatch_msg)
                print(mismatch_msg)
                
                if len(lines) < len(descriptions_to_process):
                    # Fill in missing entries with basic normalization
                    missing_count = len(descriptions_to_process) - len(lines)
                    logger.warning(f"Missing {missing_count} descriptions in batch response. Using fallback normalization.")
                    print(f"Adding {missing_count} normalized descriptions using fallback method")
                    
                    # Log the descriptions that need fallback normalization
                    missing_descriptions = descriptions_to_process[len(lines):]
                    logger.debug(f"Descriptions requiring fallback: {missing_descriptions}")
                    
                    # Use basic normalization for missing entries
                    basic_normalized = [normalize_description(desc) for desc in missing_descriptions]
                    lines.extend(basic_normalized)
                    
                    # Log the fallback normalizations
                    logger.debug(f"Fallback normalizations: {basic_normalized}")
                else:
                    # Truncate extra entries
                    extra_count = len(lines) - len(descriptions_to_process)
                    logger.warning(f"Got {extra_count} extra lines in batch response. Truncating.")
                    print(f"Truncating {extra_count} extra descriptions")
                    
                    # Log the extra entries being truncated
                    truncated_lines = lines[len(descriptions_to_process):]
                    logger.debug(f"Truncated lines: {truncated_lines}")
                    
                    lines = lines[:len(descriptions_to_process)]
                    
            normalized_descriptions = lines
            
            # Log success if we have the exact count expected
            if len(normalized_descriptions) == len(descriptions_to_process):
                print(f"Successfully normalized {len(normalized_descriptions)} descriptions in batch")
            
            # Add normalized descriptions to results and cache
            for batch_idx, result_idx in enumerate(indices_to_process):
                normalized_desc = normalized_descriptions[batch_idx]
                results[result_idx] = normalized_desc
                
                # Cache this result
                desc = descriptions_to_process[batch_idx]
                cache_key = hashlib.md5(desc.encode()).hexdigest()
                
                # Note whether this came from exact match or fallback
                was_fallback = batch_idx >= len(lines) and len(lines) < len(descriptions_to_process)
                
                NORMALIZE_CACHE[cache_key] = {
                    'result': normalized_desc,
                    'timestamp': datetime.now().isoformat(),
                    'original': desc,
                    'hits': 1,
                    'batch_processed': True,
                    'fallback_used': was_fallback
                }
            
            # Save cache after batch processing
            save_cache(NORMALIZE_CACHE, NORMALIZE_CACHE_FILE)
            
        except Exception as e:
            # Log the error with detailed information
            error_msg = f"Error in batch description normalization: {str(e)}"
            logger.error(error_msg)
            logger.exception(f"Exception details for batch normalization:")
            
            # Print a shorter message to console
            print(f"Error in batch description normalization: {e}")
            print(f"Falling back to basic normalization for {len(descriptions_to_process)} descriptions")
            
            # Log detailed information about the descriptions that triggered the error
            logger.error(f"Error occurred when processing these descriptions:")
            for i, desc in enumerate(descriptions_to_process):
                logger.error(f"  Description {i}: {desc}")
            
            # Log the fallback strategy
            logger.info(f"Applying basic normalization to {len(descriptions_to_process)} descriptions")
            
            # Fall back to basic normalization for all remaining
            basic_results = []
            for batch_idx, result_idx in enumerate(indices_to_process):
                desc = descriptions_to_process[batch_idx]
                basic_result = normalize_description(desc)
                results[result_idx] = basic_result
                basic_results.append(basic_result)
                
                # Cache the fallback result too
                cache_key = hashlib.md5(desc.encode()).hexdigest()
                NORMALIZE_CACHE[cache_key] = {
                    'result': basic_result,
                    'timestamp': datetime.now().isoformat(),
                    'original': desc,
                    'hits': 1,
                    'error': str(e),
                    'batch_processed': True,
                    'fallback_used': True
                }
            
            # Log the basic normalization results
            logger.debug(f"Basic normalization results: {basic_results}")
            
            # Save cache even when fallback is used
            logger.info(f"Saving normalization cache after fallback processing")
            save_cache(NORMALIZE_CACHE, NORMALIZE_CACHE_FILE)
    
    # Convert results dict to ordered list
    ordered_results = [results[i] for i in range(len(descriptions))]
    return ordered_results

def generate_rule_suggestions(df, threshold=0.7):
    """Generate and save new categorization rules based on patterns in uncategorized transactions."""
    uncategorized = df[df["Category"] == "Uncategorized"]
    if len(uncategorized) < 5:  # Require a minimum cluster size
        return
    prompt = f"""
    Analyze these uncategorized transactions: {uncategorized.to_json(orient='records')}.
    Suggest a new categorization rule in JSON format with:
    - 'description': a keyword or phrase to match
    - 'amount': an optional specific amount (or null)
    - 'day_range': an optional day range (e.g., [1, 5]) or [1, 31] for full month
    - 'category': the suggested category
    Only suggest if you identify a clear pattern with >{threshold} confidence.
    """
    try:
        # Use our retry-enabled function
        response = call_llm(
            prompt,
            model="gpt-4o",
            max_tokens=200,
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        
        # Use our robust JSON parsing function
        suggestion = safe_json_parse(
            content,
            fallback={"category": None}
        )
        if suggestion.get("category") and suggestion["category"] != "Uncategorized":
            CUSTOM_RULES.append(suggestion)
            with open("categories.json", "w") as f:
                json.dump({"rules": CUSTOM_RULES}, f, indent=4)
            print(f"Added new rule: {suggestion}")
    except Exception as e:
        print(f"Error generating rule suggestions: {e}")
        
def suggest_custom_rules():
    """Suggest custom rules to improve categorization of common transactions."""
    print("\n=== Custom Rule Suggestions for Common Payment Transactions ===\n")
    print("Consider adding the following custom rules to categories.json:")
    
    suggestions = [
        {
            "description": "zelle ",  # Space after to avoid partial matches
            "amount": None,
            "day_range": [1, 5],
            "category": "Rent"
        },
        {
            "description": "venmo",
            "amount": None,
            "day_range": [1, 31],
            "category": "Food"
        },
        {
            "description": "paypal",
            "amount": None,
            "day_range": [1, 31],
            "category": "Shopping"
        },
        {
            "description": "cash app",
            "amount": None,
            "day_range": [1, 31],
            "category": "Entertainment"
        }
    ]
    
    print("Copy these rules to categories.json or customize based on your spending patterns:")
    print(json.dumps({"rules": suggestions}, indent=4))
    print("\nFor personalized categorization, fill in specific names/amounts from your payment history.")
    print("Example: {'description': 'zelle fernando', 'amount': 1800.00, 'day_range': [1, 5], 'category': 'Rent'}")
    
    add_rules = input("\nAdd these sample rules to categories.json? (y/n): ").lower().strip()
    if add_rules == 'y':
        # Merge with existing rules
        combined_rules = CUSTOM_RULES + suggestions
        with open("categories.json", "w") as f:
            json.dump({"rules": combined_rules}, f, indent=4)
        print(f"Added {len(suggestions)} new rule templates to categories.json")
    
    return

def interactive_review(df):
    """Allow users to review and correct LLM categorization suggestions."""
    print("\n=== Starting Interactive Review of Low Confidence Transactions ===\n")
    print("This tool will help you review transactions where the AI is uncertain.")
    print("You can accept the AI's suggestion, change the category, or skip to the next transaction.")
    print("Any changes you make will be saved as custom rules for future transactions.\n")
    
    # Option to view rule suggestions
    show_rule_suggestions = input("Would you like to see suggestions for custom payment rules? (y/n): ").lower().strip()
    if show_rule_suggestions == 'y':
        suggest_custom_rules()
    
    # Get valid categories
    valid_categories = list(CATEGORY_RULES.keys()) + ["Uncategorized"]
    
    # Display category options
    print("Available categories:")
    for i, category in enumerate(valid_categories, 1):
        print(f"{i}. {category}")
    print("\n")
    
    # Track changes for saving at the end
    changes_made = False
    
    try:
        # Focus on low confidence transactions
        low_confidence = df[df["Confidence"] < 0.8].copy()
        if len(low_confidence) == 0:
            print("No low-confidence transactions found. All transactions have been categorized with high confidence!")
            return
            
        print(f"Found {len(low_confidence)} transactions with confidence score below 0.8")
        
        for idx, row in low_confidence.iterrows():
            print("\n" + "="*80)
            print(f"Transaction: {row['Description']}")
            print(f"Normalized: {row['NormalizedDesc']}")
            print(f"Amount: ${row['Amount']:.2f}, Date: {row['Date']}, Source: {row['Source']}")
            print(f"Current Category: {row['Category']} (Confidence: {row['Confidence']:.2f})")
            print(f"AI Reasoning: {row['Reasoning']}")
            
            choice = input("\nAccept (y), Change (c), Skip (s), Quit (q): ").lower().strip()
            
            if choice == "q":
                print("Exiting review mode...")
                break
                
            elif choice == "c":
                # Show category options
                print("\nEnter the number or name of the new category:")
                for i, category in enumerate(valid_categories, 1):
                    print(f"{i}. {category}")
                
                category_input = input("\nNew category: ").strip()
                
                # Handle numeric input
                if category_input.isdigit() and 1 <= int(category_input) <= len(valid_categories):
                    new_category = valid_categories[int(category_input) - 1]
                else:
                    # Check if input matches a category name
                    if category_input in valid_categories:
                        new_category = category_input
                    else:
                        print(f"Invalid category. Using 'Uncategorized' instead.")
                        new_category = "Uncategorized"
                
                # Update dataframe
                df.at[idx, "Category"] = new_category
                df.at[idx, "Confidence"] = 1.0
                df.at[idx, "Reasoning"] = "User override during interactive review"
                
                # Create a rule based on this correction
                # Ask if the user wants to create a custom rule
                create_rule = input("\nCreate a custom rule based on this transaction? (y/n): ").lower().strip()
                if create_rule == "y":
                    # Extract key terms from description
                    desc_words = row["Description"].lower().split()
                    key_term = input(f"Enter a keyword from the description to match similar transactions (default: {desc_words[0]}): ").strip()
                    if not key_term:
                        key_term = desc_words[0].lower()
                    
                    # Ask about amount matching
                    amount_specific = input(f"Should this rule only match transactions with amount ${row['Amount']:.2f}? (y/n): ").lower().strip()
                    amount = float(row["Amount"]) if amount_specific == "y" else None
                    
                    # Ask about day range
                    day = pd.to_datetime(row["Date"]).day
                    day_range_input = input(f"Enter day range for the rule (default: {day-2}-{day+2}, or 'any' for full month): ").strip()
                    if day_range_input.lower() == "any":
                        day_range = [1, 31]
                    elif "-" in day_range_input:
                        try:
                            start, end = map(int, day_range_input.split("-"))
                            day_range = [start, end]
                        except:
                            day_range = [max(1, day-2), min(31, day+2)]
                    else:
                        day_range = [max(1, day-2), min(31, day+2)]
                    
                    # Create and save the rule
                    new_rule = {
                        "description": key_term,
                        "amount": amount,
                        "day_range": day_range,
                        "category": new_category
                    }
                    
                    CUSTOM_RULES.append(new_rule)
                    changes_made = True
                    print(f"Added new rule: {new_rule}")
                
                print(f"Updated category to: {new_category}")
                
            elif choice == "y":
                print("Accepted current category.")
            else:
                print("Skipping to next transaction.")
        
        if changes_made:
            with open("categories.json", "w") as f:
                json.dump({"rules": CUSTOM_RULES}, f, indent=4)
            print("\nSaved updated rules to categories.json")
        
        print("\n=== Interactive Review Complete ===\n")
        
        # Save changes to the main dataframe
        df.to_csv(DEDUPE_CSV, index=False)
        print(f"Saved updates to {DEDUPE_CSV}")
        
    except KeyboardInterrupt:
        print("\n\nReview interrupted. Saving changes...")
        if changes_made:
            with open("categories.json", "w") as f:
                json.dump({"rules": CUSTOM_RULES}, f, indent=4)
            print("Saved updated rules to categories.json")
        df.to_csv(DEDUPE_CSV, index=False)
        print(f"Saved updates to {DEDUPE_CSV}")

def generate_low_confidence_rules(df, confidence_threshold=0.6):
    """Generate rules for transactions with low confidence scores."""
    low_confidence = df[(df["Confidence"] < confidence_threshold) & (df["Category"] != "Uncategorized")]
    if len(low_confidence) < 3:  # Require a minimum number of low confidence transactions
        return
    
    prompt = f"""
    Analyze these low-confidence transactions: {low_confidence.to_json(orient='records')}.
    Identify patterns that could be turned into reliable categorization rules.
    For each pattern, create a rule in JSON format with:
    - 'description': a keyword or phrase to match
    - 'amount': an optional specific amount (or null)
    - 'day_range': an optional day range (e.g., [1, 5]) or [1, 31] for full month
    - 'category': the suggested category
    Return an array of rule objects. Only include rules you have high confidence in.
    """
    try:
        # Use our retry-enabled function
        response = call_llm(
            prompt,
            model="gpt-4o",
            max_tokens=400,
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        
        # Use our robust JSON parsing function
        suggestions = safe_json_parse(content, fallback=[])
        if not isinstance(suggestions, list):
            suggestions = [suggestions]
            
        new_rules_added = 0
        for suggestion in suggestions:
            if suggestion.get("category") and suggestion["category"] != "Uncategorized":
                # Check if a similar rule already exists
                similar_rule_exists = False
                for rule in CUSTOM_RULES:
                    if (rule.get("description") == suggestion.get("description") and 
                        rule.get("category") == suggestion.get("category")):
                        similar_rule_exists = True
                        break
                
                if not similar_rule_exists:
                    CUSTOM_RULES.append(suggestion)
                    new_rules_added += 1
                    print(f"Added new rule from low confidence analysis: {suggestion}")
        
        if new_rules_added > 0:
            with open("categories.json", "w") as f:
                json.dump({"rules": CUSTOM_RULES}, f, indent=4)
            print(f"Added {new_rules_added} new rules from low confidence transactions")
    except Exception as e:
        print(f"Error generating low confidence rule suggestions: {e}")
        
def dedupe_with_llm(df):
    """Use LLM to analyze potential duplicates and confirm or refute them with reasoning."""
    print("\nIdentifying potential duplicate transactions...")
    
    # Use the already existing normalized descriptions for deduplication
    df["AbsAmount"] = df["Amount"].abs()
    df["DateAmountKey"] = df["Date"] + df["AbsAmount"].astype(str) + df["NormalizedDesc"]
    df["IsDuplicate"] = df.duplicated(subset=["DateAmountKey"], keep=False)
    
    # Add a column for deduplication reasoning
    df["DedupeReasoning"] = ""
    
    # Get only potential duplicates
    duplicates = df[df["IsDuplicate"]]
    if duplicates.empty:
        print("No potential duplicates found.")
        return df
        
    # Group by duplicate key
    grouped = duplicates.groupby("DateAmountKey")
    
    # Track indices to drop
    indices_to_drop = []
    
    # Prepare for progress tracking
    total_groups = len(grouped)
    print(f"Found {total_groups} groups of potential duplicates to analyze")
    print(f"Total potential duplicate transactions: {len(duplicates)}")
    
    start_time_dedup_analysis = time.time()
    processed_groups = 0
    deduplicated_count = 0
    not_duplicate_count = 0
    
    for key, group in grouped:
        if len(group) > 1:
            # Update progress
            processed_groups += 1
            
            # Display progress every 5 groups or at the end
            if processed_groups % 5 == 0 or processed_groups == total_groups:
                elapsed = time.time() - start_time_dedup_analysis
                rate = processed_groups / elapsed if elapsed > 0 else 0
                
                # Calculate ETA
                if rate > 0:
                    eta_seconds = (total_groups - processed_groups) / rate
                    eta = f"ETA: {eta_seconds:.0f}s" if eta_seconds < 60 else f"ETA: {eta_seconds/60:.1f}m"
                else:
                    eta = "ETA: calculating..."
                    
                suffix = f"({rate:.1f} groups/sec, found {deduplicated_count} dupes) {eta}"
                print_progress_bar(
                    processed_groups, 
                    total_groups, 
                    prefix="Deduplicating:", 
                    suffix=suffix, 
                    length=40
                )
            
            # Only analyze if we have multiple transactions with the same key
            prompt = f"""
            Are these transactions duplicates? Provide a JSON response with 'is_duplicate' (true/false) and 'reasoning'.
            Consider different sources, slight variations in description, and timing when determining if these are truly duplicates.
            Transactions: {group[["Date", "Description", "Amount", "Source"]].to_json(orient='records')}
            """
            # Create a context identifier for logs
            context_id = f"dedupe:[{key[:20]}]"
            
            try:
                logger.info(f"Analyzing potential duplicate group: {key} ({len(group)} transactions)")
                
                # Use our retry-enabled function with context
                response = call_llm(
                    prompt,
                    model="gpt-4o",
                    max_tokens=150,
                    temperature=0.2,
                    context=context_id
                )
                content = response.choices[0].message.content.strip()
                
                # Use our robust JSON parsing function with context
                result = safe_json_parse(
                    content,
                    fallback={
                        "is_duplicate": True,
                        "reasoning": "Fallback: JSON parsing error, assuming duplicate based on exact date/amount/description match"
                    },
                    context=context_id
                )
                reasoning = result.get('reasoning', 'No reasoning provided')
                
                # Log the decision with detailed information
                if result.get("is_duplicate", False):
                    logger.info(f"Duplicate detected {context_id}: {reasoning}")
                    logger.debug(f"Duplicate group {context_id}: {group[['Date', 'Description', 'Amount']].to_dict()}")
                    
                    # Keep first transaction, add reasoning to it
                    first_idx = group.index[0]
                    df.at[first_idx, "DedupeReasoning"] = f"Kept as primary. {reasoning}"
                    
                    # Mark others as duplicates and add to drop list
                    for idx in group.index[1:]:
                        indices_to_drop.append(idx)
                    
                    deduplicated_count += len(group) - 1
                else:
                    logger.info(f"False duplicate {context_id}: {reasoning}")
                    
                    # Not duplicates, update reasoning for all
                    for idx in group.index:
                        df.at[idx, "DedupeReasoning"] = f"Not a duplicate. {reasoning}"
                    
                    # Also update IsDuplicate flag
                    for idx in group.index:
                        df.at[idx, "IsDuplicate"] = False
                        
                    not_duplicate_count += len(group)
            except Exception as e:
                # Log the error with detailed information
                error_msg = f"Error during deduplication for {context_id}: {str(e)}"
                logger.error(error_msg)
                logger.exception(f"Exception details for {context_id}:")
                
                # Print a shorter message to console
                print(f"\nError during deduplication: {e}")
                
                # Fall back to basic deduplication if LLM fails
                logger.warning(f"Falling back to basic deduplication for {context_id}")
                
                first_idx = group.index[0]
                df.at[first_idx, "DedupeReasoning"] = "Kept as primary (fallback to basic deduplication due to error)"
                
                for idx in group.index[1:]:
                    indices_to_drop.append(idx)
                    
                deduplicated_count += len(group) - 1
    
    # Final progress update
    print_progress_bar(total_groups, total_groups, prefix="Deduplicating:", suffix="Complete!", length=40)
    
    # Drop identified duplicates
    if indices_to_drop:
        df = df.drop(indices_to_drop)
        print(f"\nRemoved {len(indices_to_drop)} duplicate transactions using LLM reasoning")
    else:
        print("\nNo duplicates found after LLM analysis")
        
    # Show summary
    print(f"Deduplication summary:")
    print(f"- Analyzed {total_groups} groups of potential duplicates")
    print(f"- Found {deduplicated_count} actual duplicate transactions")
    print(f"- Identified {not_duplicate_count} false positives (similar but not duplicates)")
    
    elapsed_total = time.time() - start_time_dedup_analysis
    print(f"Deduplication analysis completed in {elapsed_total:.1f}s")
    
    return df

def categorize_with_gpt4_advanced(df_row, existing_rules, past_transactions):
    """Categorize a transaction using GPT-4 with caching."""
    # Create a cache key based on critical transaction details
    # We don't include past_transactions in the key as it might change frequently
    # and we want to leverage the cache even if past_transactions changes slightly
    key_dict = {
        'Description': df_row['Description'],
        'Amount': float(df_row['Amount']),
        'Date': df_row['Date'],
        'Source': df_row['Source']
    }
    if 'NormalizedDesc' in df_row:
        key_dict['NormalizedDesc'] = df_row['NormalizedDesc']
        
    # Also include rule fingerprints in the key, so cache invalidates if rules change
    # Use a hash of the rules to keep the key manageable
    rules_hash = hashlib.md5(json.dumps(existing_rules, sort_keys=True).encode()).hexdigest()
    key_dict['rules_hash'] = rules_hash
    
    # Create the final cache key
    cache_key = hashlib.md5(json.dumps(key_dict, sort_keys=True).encode()).hexdigest()
    
    # Check if we have this transaction in cache
    if cache_key in CATEGORIZE_CACHE:
        # Track cache hits for reporting
        CATEGORIZE_CACHE[cache_key]['hits'] += 1
        result = CATEGORIZE_CACHE[cache_key]['result']
        return result[0], result[1], result[2]
    
    # Not in cache, so categorize with LLM
    # Include both original and normalized descriptions in the prompt
    normalized_desc = df_row["NormalizedDesc"] if "NormalizedDesc" in df_row else "Not Available"
    
    prompt = f"""
    Categorize this transaction into one of: {list(CATEGORY_RULES.keys()) + ['Uncategorized']}.
    Transaction: 
    - Date: '{df_row['Date']}'
    - Original Description: '{df_row['Description']}'
    - Normalized Description: '{normalized_desc}'
    - Amount: {df_row['Amount']}
    - Source: '{df_row['Source']}'
    
    Existing Rules: {json.dumps(existing_rules, indent=2)}.
    Past Transactions (sample): {past_transactions.head(5).to_json(orient='records')}.
    
    IMPORTANT CATEGORIZATION GUIDELINES:
    1. 'Payments' category should ONLY be used for generic bank transfers, bill payments, or when purpose is unclear
    2. For transfers with clear purposes, use the appropriate specific category:
       - Rent/mortgage payments → 'Rent'
       - Grocery store transfers → 'Food'
       - Online shopping transfers → 'Shopping'
       - Subscription services → 'Subscriptions'
       - Investment deposits → 'Investments'
    3. Consider both the recipient name and amount patterns:
       - Regular large transfers near start of month likely 'Rent'
       - Transfers to investment platforms are 'Investments'
       - Transfers for food delivery or restaurants are 'Food'
    4. Be specific whenever possible - only use 'Payments' as a last resort
    
    Return a valid JSON object:
    ```json
    {{"category": "string", "confidence": float 0-1, "reasoning": "string < 20 words"}}
    ```
    Ensure the response is valid JSON within json fences.
    """
    
    # Create a context identifier for logs
    context_id = f"txn:[{df_row['Description'][:20]}...{df_row['Amount']}]"
    
    try:
        logger.info(f"Categorizing transaction: {df_row['Description']} (${df_row['Amount']})")
        
        # Use our retry-enabled function with context
        response = call_llm(
            prompt, 
            model="gpt-4o", 
            max_tokens=100, 
            temperature=0.2,
            context=context_id
        )
        content = response.choices[0].message.content.strip()
        
        # Use our robust JSON parsing function with context
        result = safe_json_parse(
            content, 
            fallback={
                "category": "Uncategorized", 
                "confidence": 0.0,
                "reasoning": f"Failed to parse LLM response: {content[:100]}"
            },
            context=context_id
        )
        
        category = result["category"]
        confidence = result["confidence"]
        reasoning = result["reasoning"]
        
        # Log the categorization result
        logger.info(f"Categorized {context_id} as '{category}' with confidence {confidence}")
        logger.debug(f"Reasoning for {context_id}: {reasoning}")
        
        # Cache the result
        CATEGORIZE_CACHE[cache_key] = {
            'result': (category, confidence, reasoning),
            'timestamp': datetime.now().isoformat(),
            'transaction': {
                'Description': df_row['Description'],
                'Amount': float(df_row['Amount']),
                'Date': df_row['Date'],
                'Source': df_row['Source']
            },
            'hits': 1
        }
        
        # Periodically save cache to disk
        if len(CATEGORIZE_CACHE) % 5 == 0:
            save_cache(CATEGORIZE_CACHE, CATEGORIZE_CACHE_FILE)
            
        return category, confidence, reasoning
    except Exception as e:
        # Log the error with detailed information
        error_msg = f"Failed to categorize {context_id}: {str(e)}"
        logger.error(error_msg)
        logger.exception(f"Exception details for {context_id}:")
        
        # Print a shorter message to console
        print(f"Error categorizing transaction: {e}")
        
        # Use a detailed fallback with error information
        fallback_result = ("Uncategorized", 0.0, f"Failed to categorize due to error: {e}")
        
        # Cache the fallback result too
        CATEGORIZE_CACHE[cache_key] = {
            'result': fallback_result,
            'timestamp': datetime.now().isoformat(),
            'transaction': {
                'Description': df_row['Description'],
                'Amount': float(df_row['Amount']),
                'Date': df_row['Date'],
                'Source': df_row['Source']
            },
            'hits': 1,
            'error': str(e)
        }
        
        return fallback_result

# Progress saving and signal handling
def save_progress(df, interrupt=False):
    """Save current progress to files to allow recovery from crashes."""
    # Create temporary file paths to avoid corruption if interrupted during save
    temp_csv = DEDUPE_CSV + ".tmp"
    
    # Save the dataframe
    df.to_csv(temp_csv, index=False)
    
    # Save all caches
    save_cache(NORMALIZE_CACHE, NORMALIZE_CACHE_FILE)
    save_cache(CATEGORIZE_CACHE, CATEGORIZE_CACHE_FILE)
    
    # Rename temporary files to final files (atomic operation)
    if os.path.exists(temp_csv):
        os.rename(temp_csv, DEDUPE_CSV)
    
    print(f"Progress saved to {DEDUPE_CSV} ({len(df)} transactions)")
    
    if interrupt:
        print("Exiting due to interrupt. Progress has been saved.")
        sys.exit(0)

def signal_handler(sig, frame):
    """Handle interrupt signals by saving progress and exiting gracefully."""
    print("\nInterrupt received. Saving progress...")
    # Access the global dataframe
    if 'df' in globals():
        save_progress(globals()['df'], interrupt=True)
    else:
        print("No data to save. Exiting.")
        sys.exit(0)

# Register signal handler for graceful interruption
signal.signal(signal.SIGINT, signal_handler)

# Update categorize_transaction
def categorize_transaction(df_row, past_transactions):
    # Use normalized description if available, otherwise use original description
    desc = df_row["NormalizedDesc"].lower() if "NormalizedDesc" in df_row else df_row["Description"].lower()
    original_desc = df_row["Description"].lower()
    
    # Check custom and hardcoded rules first
    for rule in CUSTOM_RULES:
        # Check both normalized and original description to ensure backward compatibility
        if ((rule["description"] in desc or rule["description"] in original_desc) and 
            df_row["Amount"] == rule["amount"] and 
            pd.to_datetime(df_row["Date"]).day in rule["day_range"]):
            return rule["category"], 1.0, "Matched custom rule"
            
    for category, keywords in CATEGORY_RULES.items():
        # Check both normalized and original description
        if any(keyword in desc for keyword in keywords) or any(keyword in original_desc for keyword in keywords):
            return category, 1.0, "Matched hardcoded rule"
            
    # Fallback to LLM
    category, confidence, reasoning = categorize_with_gpt4_advanced(df_row, {"hardcoded": CATEGORY_RULES, "custom": CUSTOM_RULES}, past_transactions)
    if confidence > 0.8 and category != "Uncategorized":
        print(f"LLM suggested: {category} for '{desc}' (Confidence: {confidence}, Reasoning: {reasoning})")
    return category, confidence, reasoning

def process_pdf(file_path, source_name):
    all_data = []
    if "apple" in source_name.lower():
        source = "Apple"
    elif "estmt" in source_name.lower() or "boa" in source_name.lower():
        source = "BoA"
    elif "sfcu" in source_name.lower():
        source = "SFCU"
    elif "citi" in source_name.lower():
        source = "Citi"
    else:
        source = "Unknown"

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            tables = page.extract_tables()

            if source == "Unknown":
                if "Citi" in text:
                    source = "Citi"
                elif "Bank of America" in text:
                    source = "BoA"

            for table in tables:
                if not table or table[0][0] in ["Transaction", "Trans.", "Date", ""]:
                    continue
                for row in table:
                    if source == "Apple":
                        if len(row) == 3 and "ACH Deposit" in str(row[1]):
                            date = row[0].strip("$")
                            desc = row[1]
                            amount = row[2].replace("-$", "-").replace("$", "")
                            all_data.append([date, desc, amount, source, file_path])
                        elif len(row) == 4 and row[3]:
                            date = row[0].strip("$")
                            desc = row[1]
                            amount = row[3].replace("$", "")
                            all_data.append([date, desc, amount, source, file_path])
                    elif source == "BoA" and len(row) >= 6:
                        post_date = row[1]
                        desc = row[2]
                        amount = row[5].replace("$", "").replace(",", "")
                        if post_date and desc and amount:
                            all_data.append([post_date, desc, amount, source, file_path])
                    elif source == "Citi":
                        if len(row) == 4:
                            post_date = row[1] if row[1] else row[0]
                            desc = row[2]
                            amount = row[3].replace("-$", "-").replace("$", "").replace(",", "")
                            if post_date and desc and amount:
                                all_data.append([post_date, desc, amount, source, file_path])
                        elif len(row) == 2 and "FEE" in str(row[1]):
                            date = row[0]
                            desc = row[1]
                            amount = row[2].replace("$", "") if len(row) > 2 else "0.47"
                            all_data.append([date, desc, amount, source, file_path])
    return all_data

def process_csv(file_path, source_name):
    all_data = []
    if "boa" in source_name.lower() or "estmt" in source_name.lower():
        source = "BoA"
        df = pd.read_csv(file_path, usecols=["Posted Date", "Payee", "Amount"])
        df.columns = ["Date", "Description", "Amount"]
    elif "sfcu" in source_name.lower() or "accounthistory" in source_name.lower():
        source = "SFCU"
        df = pd.read_csv(file_path)
        df["Amount"] = df["Debit"].fillna(0) - df["Credit"].fillna(0)
        df = df[["Post Date", "Description", "Amount"]]
        df.columns = ["Date", "Description", "Amount"]
    elif "citi" in source_name.lower():
        source = "Citi"
        df = pd.read_csv(file_path, usecols=["Posted Date", "Payee", "Amount"])
        df.columns = ["Date", "Description", "Amount"]
    else:
        source = "Unknown"
        return all_data

    df["Source"] = source
    df["File"] = file_path
    return df.values.tolist()

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    Parameters:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\n") (Str)
    """
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print new line on complete
    if iteration == total: 
        print()

def view_logs(num_lines=50, level="ALL", contains=None):
    """View recent log entries with optional filtering."""
    if not os.path.exists(LOG_FILE):
        print(f"Log file {LOG_FILE} does not exist.")
        return
        
    print(f"\nShowing last {num_lines} log entries" + 
          (f" with level {level}" if level != "ALL" else "") +
          (f" containing '{contains}'" if contains else "") +
          ":\n")
    
    # Read the log file
    with open(LOG_FILE, 'r') as f:
        log_lines = f.readlines()
    
    # Apply filters
    filtered_lines = []
    for line in log_lines:
        if level != "ALL" and f" - {level} - " not in line:
            continue
        if contains and contains not in line:
            continue
        filtered_lines.append(line.strip())
    
    # Show last n lines
    lines_to_show = filtered_lines[-num_lines:] if filtered_lines else []
    
    if lines_to_show:
        for line in lines_to_show:
            print(line)
    else:
        print("No matching log entries found.")

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Expense Tracker: Process and categorize financial transactions")
    parser.add_argument("--review", action="store_true", help="Enable interactive review of low-confidence transactions")
    parser.add_argument("--skip-normalization", action="store_true", help="Skip LLM-based description normalization (faster)")
    parser.add_argument("--skip-llm-dedup", action="store_true", help="Skip LLM-based deduplication (faster)")
    parser.add_argument("--skip-rule-gen", action="store_true", help="Skip automatic rule generation")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode (skips all LLM features)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress information")
    parser.add_argument("--view-logs", action="store_true", help="View recent log entries")
    parser.add_argument("--log-lines", type=int, default=50, help="Number of log lines to view")
    parser.add_argument("--log-level", choices=["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        default="ALL", help="Filter logs by level")
    parser.add_argument("--log-contains", type=str, help="Filter logs by content")
    parser.add_argument("--suggest-rules", action="store_true", help="Get suggestions for custom payment rules")
    args = parser.parse_args()
    
    # Handle the view-logs command
    if args.view_logs:
        view_logs(args.log_lines, args.log_level, args.log_contains)
        return
        
    # Handle the suggest-rules command
    if args.suggest_rules:
        suggest_custom_rules()
        return
    
    # If fast mode is enabled, set all skip flags
    if args.fast:
        args.skip_normalization = True
        args.skip_llm_dedup = True
        args.skip_rule_gen = True
        print("Running in fast mode - all LLM features disabled")
    
    # Performance tracking
    start_time_total = time.time()
    all_transactions = []

    # Scan inputs/ directory
    for filename in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, filename)
        source_name = os.path.splitext(filename)[0].lower()

        if filename.endswith(".pdf"):
            print(f"Processing PDF: {filename}")
            transactions = process_pdf(file_path, source_name)
            all_transactions.extend(transactions)
        elif filename.endswith(".csv"):
            print(f"Processing CSV: {filename}")
            transactions = process_csv(file_path, source_name)
            all_transactions.extend(transactions)

    # Create DataFrame
    if not all_transactions:
        print("No files found in inputs/ directory!")
        return

    df = pd.DataFrame(all_transactions, columns=["Date", "Description", "Amount", "Source", "File"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%m/%d/%Y")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    
    # Add normalized descriptions
    if args.skip_normalization:
        print("Skipping LLM-based description normalization (using basic normalization instead)...")
        df["NormalizedDesc"] = df["Description"].apply(normalize_description)
    else:
        print("Normalizing transaction descriptions using LLM (with caching and batching)...")
        
        # Calculate cache hit stats before processing
        cache_hits = sum(1 for desc in df["Description"] if hashlib.md5(desc.encode()).hexdigest() in NORMALIZE_CACHE)
        print(f"Found {cache_hits}/{len(df)} descriptions already in cache")
        
        # Process in larger batches for efficiency
        batch_size = 20
        start_time = time.time()
        last_save_time = time.time()
        save_interval = 60  # Save at least every 60 seconds
        processed = 0
        
        # Initialize progress tracking
        print("\nStarting description normalization...")
        progress_update_interval = 1 if args.verbose else 5  # Update progress bar more frequently in verbose mode
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            descriptions = batch["Description"].tolist()
            
            # Process batch and update dataframe
            normalized_descriptions = normalize_descriptions_batch(descriptions)
            df.loc[batch.index, "NormalizedDesc"] = normalized_descriptions
            
            # Update processed count
            processed += len(batch)
            
            # Save progress periodically
            current_time = time.time()
            if i % 100 == 0 or (current_time - last_save_time) > save_interval:
                # Save the cache
                save_cache(NORMALIZE_CACHE, NORMALIZE_CACHE_FILE)
                # Save intermediate results
                df.to_csv(OUTPUT_CSV, index=False)
                last_save_time = current_time
                
                if not args.verbose:  # Only show this message in non-verbose mode
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"Saved progress: {processed}/{len(df)} descriptions normalized ({rate:.1f} desc/sec)")
            
            # Update progress bar
            if i % (batch_size * progress_update_interval) == 0 or processed >= len(df):
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                cache_hit_rate = (cache_hits / processed) * 100 if processed > 0 else 0
                
                # Calculate ETA
                if rate > 0:
                    eta_seconds = (len(df) - processed) / rate
                    eta = f"ETA: {eta_seconds:.0f}s" if eta_seconds < 60 else f"ETA: {eta_seconds/60:.1f}m"
                else:
                    eta = "ETA: calculating..."
                    
                suffix = f"({rate:.1f} desc/sec, {cache_hit_rate:.1f}% cache hits) {eta}"
                print_progress_bar(processed, len(df), prefix="Normalizing:", suffix=suffix, length=40)
                
                if args.verbose and i > 0:
                    print(f"\nNormalized {processed}/{len(df)} descriptions... ({rate:.1f} desc/sec)")
                    print(f"Cache hits: {cache_hits} ({cache_hit_rate:.1f}%), Cache size: {len(NORMALIZE_CACHE)}")
        
        # Calculate final stats
        elapsed = time.time() - start_time
        rate = len(df) / elapsed if elapsed > 0 else 0
        cache_entries = len(NORMALIZE_CACHE)
        print(f"Completed description normalization for {len(df)} transactions in {elapsed:.1f}s ({rate:.1f} desc/sec)")
        print(f"Cache now contains {cache_entries} unique descriptions")

    # Add categories with confidence and reasoning
    start_time_cat = time.time()
    print("Categorizing transactions...")
    
    # Track cache hits for reporting
    cat_cache_hits = 0
    
    # Check if we can resume from an existing file
    if os.path.exists(DEDUPE_CSV):
        print(f"Found existing progress file: {DEDUPE_CSV}")
        resume_choice = input("Resume from existing progress? (y/n): ").lower().strip()
        if resume_choice == 'y':
            try:
                saved_df = pd.read_csv(DEDUPE_CSV)
                # Check if this is a valid saved file with required columns
                required_cols = ["Date", "Description", "NormalizedDesc", "Amount", "Category", "Confidence"]
                if all(col in saved_df.columns for col in required_cols):
                    # Match transactions with the current loaded set
                    # Create a key for each transaction that should uniquely identify it
                    df["TxnKey"] = df["Date"] + ":" + df["Description"] + ":" + df["Amount"].astype(str)
                    saved_df["TxnKey"] = saved_df["Date"] + ":" + saved_df["Description"] + ":" + saved_df["Amount"].astype(str)
                    
                    # Find which transactions are already processed
                    processed_keys = set(saved_df["TxnKey"])
                    df["IsProcessed"] = df["TxnKey"].isin(processed_keys)
                    
                    # Count how many we can skip
                    to_skip = df["IsProcessed"].sum()
                    to_process = len(df) - to_skip
                    
                    print(f"Found {to_skip} already processed transactions. {to_process} remain to be processed.")
                    
                    # For already processed transactions, copy the values from saved_df
                    for idx, row in df[df["IsProcessed"]].iterrows():
                        saved_row = saved_df[saved_df["TxnKey"] == row["TxnKey"]].iloc[0]
                        df.at[idx, "Category"] = saved_row["Category"]
                        df.at[idx, "Confidence"] = saved_row["Confidence"]
                        df.at[idx, "Reasoning"] = saved_row["Reasoning"]
                        if "NormalizedDesc" in saved_row:
                            df.at[idx, "NormalizedDesc"] = saved_row["NormalizedDesc"]
                        
                    print("Successfully restored previous categorization progress.")
                else:
                    print("Saved file doesn't have required columns. Starting fresh.")
            except Exception as e:
                print(f"Error loading previous progress: {e}. Starting fresh.")
    
    # Process each row and update the dataframe directly (with periodic saves)
    print("\nStarting transaction categorization...")
    
    # First, make sure all rows have the required columns
    if "Category" not in df.columns:
        df["Category"] = None
    if "Confidence" not in df.columns:
        df["Confidence"] = None
    if "Reasoning" not in df.columns:
        df["Reasoning"] = None
    
    # Count how many transactions need processing - only after ensuring columns exist
    already_categorized = df[pd.notna(df["Category"]) & pd.notna(df["Confidence"])].shape[0]
    to_process = len(df) - already_categorized
    
    results_count = 0
    cat_cache_hits = 0
    last_save_time = time.time()
    save_interval = 60  # Save at least every 60 seconds
    progress_update_interval = 1 if args.verbose else 5  # Update progress bar more frequently in verbose mode
    
    start_time_categorize = time.time()
    
    # Log column information
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    logger.info(f"Found {already_categorized} already categorized transactions. {to_process} remain to be processed.")
    
    # Show initial progress including already categorized transactions
    if already_categorized > 0:
        print(f"Found {already_categorized} already categorized transactions. {to_process} remain to be processed.")
        
    if to_process == 0:
        print("All transactions are already categorized. Skipping categorization step.")
    else:
        print(f"Categorizing {to_process} transactions...")
    
    # Process rows that aren't already categorized
    for idx, row in df.iterrows():
        # Skip already processed transactions
        if pd.notna(row.get("Category")) and pd.notna(row.get("Confidence")):
            continue
            
        # Try to fetch from cache first for quick reporting
        key_dict = {
            'Description': row['Description'],
            'Amount': float(row['Amount']),
            'Date': row['Date'],
            'Source': row['Source']
        }
        if 'NormalizedDesc' in row:
            key_dict['NormalizedDesc'] = row['NormalizedDesc']
            
        rules_hash = hashlib.md5(json.dumps({"hardcoded": CATEGORY_RULES, "custom": CUSTOM_RULES}, sort_keys=True).encode()).hexdigest()
        key_dict['rules_hash'] = rules_hash
        
        cache_key = hashlib.md5(json.dumps(key_dict, sort_keys=True).encode()).hexdigest()
        
        if cache_key in CATEGORIZE_CACHE:
            cat_cache_hits += 1
        
        # Apply categorization and update dataframe directly
        category, confidence, reasoning = categorize_transaction(row, df)
        df.at[idx, "Category"] = category
        df.at[idx, "Confidence"] = confidence
        df.at[idx, "Reasoning"] = reasoning
        
        results_count += 1
        
        # Save progress periodically (every 10 transactions and at least every 60 seconds)
        current_time = time.time()
        if results_count % 10 == 0 or (current_time - last_save_time) > save_interval:
            save_progress(df)
            last_save_time = current_time
            
            if not args.verbose:  # Only show this message in non-verbose mode
                elapsed = time.time() - start_time_categorize
                rate = results_count / elapsed if elapsed > 0 else 0
                print(f"Saved progress: {results_count}/{to_process} transactions categorized ({rate:.1f} txn/sec)")
        
        # Update progress bar
        if to_process > 0 and (results_count % progress_update_interval == 0 or results_count >= to_process):
            elapsed = time.time() - start_time_categorize
            rate = results_count / elapsed if elapsed > 0 else 0
            cache_hit_rate = (cat_cache_hits / results_count) * 100 if results_count > 0 else 0
            
            # Calculate ETA
            if rate > 0:
                eta_seconds = (to_process - results_count) / rate
                eta = f"ETA: {eta_seconds:.0f}s" if eta_seconds < 60 else f"ETA: {eta_seconds/60:.1f}m"
            else:
                eta = "ETA: calculating..."
                
            suffix = f"({rate:.1f} txn/sec, {cache_hit_rate:.1f}% cache hits) {eta}"
            print_progress_bar(results_count, to_process, prefix="Categorizing:", suffix=suffix, length=40)
            
            if args.verbose and results_count > 0 and results_count % 10 == 0:
                print(f"\nProcessed {results_count}/{to_process} transactions ({cat_cache_hits} from cache)")
                # Show category distribution
                categories = df["Category"].value_counts().to_dict()
                print("Category distribution so far:")
                for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                    if pd.notna(cat):
                        print(f"  {cat}: {count}")
    
    # Show final summary if we processed any transactions
    if results_count > 0:
        print("\nCategory distribution:")
        categories = df["Category"].value_counts().to_dict()
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            if pd.notna(cat):
                print(f"  {cat}: {count}")
                
        confidence_avg = df["Confidence"].mean()
        print(f"Average confidence: {confidence_avg:.2f}")

    # Report performance for categorization
    elapsed_cat = time.time() - start_time_cat
    print(f"Categorized {len(df)} transactions in {elapsed_cat:.1f}s ({cat_cache_hits} from cache)")
    
    # Save the cache
    save_cache(CATEGORIZE_CACHE, CATEGORIZE_CACHE_FILE)

    # Save all transactions
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Combined expenses saved to {OUTPUT_CSV} with {len(df)} transactions!")

    # Handle duplicates
    start_time_dedup = time.time()
    
    # Check if a deduped file already exists and if we want to skip deduplication
    skip_dedup = False
    if os.path.exists(DEDUPE_CSV):
        print(f"Found existing deduplicated file: {DEDUPE_CSV}")
        # If we've made it this far and there's a dedupe file, we probably want to resume
        resume_dedupe = input("Skip deduplication and use existing file? (y/n): ").lower().strip()
        if resume_dedupe == 'y':
            try:
                deduped_df = pd.read_csv(DEDUPE_CSV)
                # Verify it has the required columns
                required_cols = ["Date", "Description", "Amount", "Category", "IsDuplicate"]
                if all(col in deduped_df.columns for col in required_cols):
                    print(f"Using existing deduplicated file with {len(deduped_df)} transactions")
                    skip_dedup = True
                else:
                    print("Existing file doesn't have required columns. Running deduplication...")
            except Exception as e:
                print(f"Error loading deduplicated file: {e}. Running deduplication...")
    
    if not skip_dedup:
        if args.skip_llm_dedup:
            print("Using basic deduplication (skipping LLM-based reasoning)...")
            # Use basic deduplication
            df["AbsAmount"] = df["Amount"].abs()
            df["DateAmountKey"] = df["Date"] + df["AbsAmount"].astype(str) + df["NormalizedDesc"]
            df["IsDuplicate"] = df.duplicated(subset=["DateAmountKey"], keep=False)
            df["DedupeReasoning"] = "Basic deduplication"
            deduped_df = df.drop_duplicates(subset=["DateAmountKey"], keep="first")
            print(f"Removed {len(df) - len(deduped_df)} duplicate transactions using basic deduplication")
        else:
            print("Deduplicating transactions using LLM reasoning...")
            deduped_df = dedupe_with_llm(df)
        
    elapsed_dedup = time.time() - start_time_dedup
    print(f"Deduplication completed in {elapsed_dedup:.1f}s")

    # Save deduplicated output using our save_progress function
    deduped_df = deduped_df[["Date", "Description", "NormalizedDesc", "Amount", "Source", "File", "Category", "Confidence", "Reasoning", "IsDuplicate", "DedupeReasoning"]]
    
    # Final save
    save_progress(deduped_df)
    print(f"Deduplicated expenses saved to {DEDUPE_CSV} with {len(deduped_df)} unique transactions!")
    
    # Generate new rule suggestions if not skipped
    if not args.skip_rule_gen:
        print("Analyzing transactions for rule generation...")
        generate_rule_suggestions(deduped_df)
        
        print("Analyzing low confidence transactions for rule generation...")
        generate_low_confidence_rules(deduped_df)
    else:
        print("Skipping automatic rule generation...")
        
    # Launch interactive review if requested
    if args.review:
        interactive_review(deduped_df)
        
    # Final performance stats
    elapsed_total = time.time() - start_time_total
    
    # Log the final performance statistics
    logger.info("="*50)
    logger.info("EXPENSE TRACKER COMPLETED")
    logger.info("="*50)
    logger.info(f"Total processing time: {elapsed_total:.1f}s")
    logger.info(f"- Normalized {len(df)} descriptions")
    logger.info(f"- Categorized {len(df)} transactions")
    logger.info(f"- Removed {len(df) - len(deduped_df)} duplicates")
    logger.info(f"- Final transaction count: {len(deduped_df)}")
    
    # Log category distribution
    categories = deduped_df["Category"].value_counts().to_dict()
    logger.info("Category distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        if pd.notna(cat):
            percentage = (count/len(deduped_df))*100
            logger.info(f"  {cat}: {count} ({percentage:.1f}%)")
    
    # Log cache statistics
    normalize_cache_size = len(NORMALIZE_CACHE)
    categorize_cache_size = len(CATEGORIZE_CACHE)
    logger.info(f"Cache statistics:")
    logger.info(f"- Normalization cache: {normalize_cache_size} entries")
    logger.info(f"- Categorization cache: {categorize_cache_size} entries")
    
    # Print summary to console
    print("\n" + "="*50)
    print(f"EXPENSE TRACKER SUMMARY")
    print("="*50)
    
    # Transaction statistics
    print("\nTransaction Statistics:")
    print(f"- Total transactions processed: {len(df)}")
    print(f"- Unique transactions after deduplication: {len(deduped_df)}")
    print(f"- Duplicates removed: {len(df) - len(deduped_df)}")
    
    # Category statistics
    categories = deduped_df["Category"].value_counts().to_dict()
    category_percentages = {cat: (count/len(deduped_df))*100 for cat, count in categories.items()}
    
    print("\nCategory Distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        if pd.notna(cat):
            percentage = category_percentages[cat]
            print(f"  {cat}: {count} ({percentage:.1f}%)")
    
    # Confidence statistics
    confidence_avg = deduped_df["Confidence"].mean()
    confidence_low = deduped_df[deduped_df["Confidence"] < 0.7].shape[0]
    confidence_pct = (confidence_low / len(deduped_df)) * 100 if len(deduped_df) > 0 else 0
    
    print("\nConfidence Statistics:")
    print(f"- Average confidence score: {confidence_avg:.2f}")
    print(f"- Low confidence transactions (<0.7): {confidence_low} ({confidence_pct:.1f}%)")
    
    # Performance statistics
    print("\nPerformance Statistics:")
    print(f"- Total processing time: {elapsed_total:.1f}s")
    print(f"- Description normalization: {elapsed:.1f}s")
    print(f"- Transaction categorization: {elapsed_cat:.1f}s")
    print(f"- Deduplication: {elapsed_dedup:.1f}s")
    
    # Cache statistics
    cache_stats = {
        "normalize_cache_size": len(NORMALIZE_CACHE),
        "categorize_cache_size": len(CATEGORIZE_CACHE),
        "normalize_cache_hits": cache_hits,
        "categorize_cache_hits": cat_cache_hits
    }
    
    # Calculate hit rates
    normalize_hit_rate = (cache_hits / len(df)) * 100 if len(df) > 0 else 0
    categorize_hit_rate = (cat_cache_hits / len(df)) * 100 if len(df) > 0 else 0
    
    print("\nCache Statistics:")
    print(f"- Normalization cache: {cache_stats['normalize_cache_size']} entries")
    print(f"  {cache_stats['normalize_cache_hits']} hits ({normalize_hit_rate:.1f}% hit rate)")
    print(f"- Categorization cache: {cache_stats['categorize_cache_size']} entries")
    print(f"  {cache_stats['categorize_cache_hits']} hits ({categorize_hit_rate:.1f}% hit rate)")
    
    print("\n" + "="*50)
    print(f"Processing complete! Results saved to {DEDUPE_CSV}")
    print("="*50)

if __name__ == "__main__":
    main()