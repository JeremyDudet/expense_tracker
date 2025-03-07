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
import re
from datetime import datetime
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_log, after_log
from dotenv import load_dotenv
from fuzzywuzzy import fuzz

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

# API usage tracking
API_USAGE = {
    "total_calls": 0,
    "gpt4_calls": 0,
    "gpt35_calls": 0,
    "normalization_calls": 0,
    "categorization_calls": 0,
    "rule_generation_calls": 0,
    "deduplication_calls": 0,
    "token_estimate": 0,
    "cost_estimate": 0.0
}

# List to store user corrections for rule generation
CORRECTIONS = []  # Format: {"transaction": {...}, "original_category": "...", "new_category": "..."}

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
    
    # Use cheaper model for simpler tasks based on context
    # Note: We're not accessing args directly here to avoid the NameError
    if context and "norm_batch" in str(context):
        # Use 3.5 for normalization tasks by default
        if model == "gpt-4o":
            model = "gpt-3.5-turbo"
            logger.info(f"Downgrading to {model} for normalization task")
    
    # Log basic information about the request
    logger.info(f"Calling LLM{context_str}: model={model}, max_tokens={max_tokens}, temperature={temperature}")
    
    # Log a truncated version of the prompt for debugging
    truncated_prompt = prompt[:200] + "..." if len(prompt) > 200 else prompt
    logger.debug(f"LLM prompt{context_str}: {truncated_prompt}")
    
    # Check LLM cache for this exact prompt
    cache_key = hashlib.md5((model + prompt).encode()).hexdigest()
    if cache_key in LLM_CACHE:
        logger.info(f"LLM cache hit{context_str} for {model}")
        LLM_CACHE[cache_key]['hits'] += 1
        return LLM_CACHE[cache_key]['response']
    
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
        
        # Update API usage statistics
        API_USAGE["total_calls"] += 1
        
        # Track model-specific usage
        if "gpt-4" in model:
            API_USAGE["gpt4_calls"] += 1
        elif "gpt-3.5" in model:
            API_USAGE["gpt35_calls"] += 1
            
        # Track purpose-specific usage
        if context:
            if "norm" in str(context):
                API_USAGE["normalization_calls"] += 1
            elif "txn:" in str(context):
                API_USAGE["categorization_calls"] += 1
            elif "rule" in str(context):
                API_USAGE["rule_generation_calls"] += 1
            elif "dedupe" in str(context):
                API_USAGE["deduplication_calls"] += 1
        # Show feedback on API usage (but don't rely on args to avoid NameError)
        # We'll just log this info instead of using the arg flag
        logger.info(f"API Call: {model} - Total: {API_USAGE['total_calls']} (GPT-4: {API_USAGE['gpt4_calls']}, GPT-3.5: {API_USAGE['gpt35_calls']})")
                
        # Rough token estimation (to be improved with actual token counts)
        prompt_tokens = len(prompt) / 3  # Very rough approximation
        response_tokens = len(response.choices[0].message.content) / 2
        total_tokens = prompt_tokens + response_tokens
        API_USAGE["token_estimate"] += total_tokens
        
        # Very rough cost estimation
        if "gpt-4" in model:
            # Approximately $0.01 per 1K tokens for GPT-4
            API_USAGE["cost_estimate"] += total_tokens * 0.00001
        elif "gpt-3.5" in model:
            # Approximately $0.001 per 1K tokens for GPT-3.5
            API_USAGE["cost_estimate"] += total_tokens * 0.000001
        
        # Log success
        logger.info(f"LLM call succeeded{context_str} in {duration:.2f}s")
        
        # Log response details
        truncated_response = response.choices[0].message.content[:200] + "..." if len(response.choices[0].message.content) > 200 else response.choices[0].message.content
        logger.debug(f"LLM response{context_str}: {truncated_response}")
        
        # Cache the response
        LLM_CACHE[cache_key] = {
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'prompt': prompt,
            'hits': 1
        }
        
        return response
        
    except Exception as e:
        # Log the error
        logger.error(f"LLM call failed{context_str}: {str(e)}")
        
        # Add prompt details for debugging
        logger.error(f"Failed LLM prompt{context_str}: {prompt}")
        
        # Re-raise the exception to trigger retry
        raise
        
# Define correction cache file path
CORRECTIONS_CACHE_FILE = os.path.join(CACHE_DIR, "corrections_cache.pkl")

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
CORRECTIONS = load_cache(CORRECTIONS_CACHE_FILE, default=[])

# Hardcoded category rules
CATEGORY_RULES = {
    "Entertainment": ["netflx", "steamgames", "openai", "claude.ai", "cursor", "medium", "x corp", "uber *one", 
                      "hbo max", "disney+", "spotify", "hulu", "paramount+", "apple tv", "prime video", "cinema",
                      "netflix", "spotify", "youtube", "twitch", "discord", "nintendo", "playstation", "xbox",
                      "movie", "theater", "concert", "festival", "game", "gaming", "steam"],
    "Food": ["uber *eats", "starbucks", "tacos", "restaurant", "rest ", "coffee", "cafe", "doss av", 
             "doordash", "grubhub", "mcdonald", "chipotle", "taco bell", "grocery", "safeway", "trader joe",
             "whole foods", "sushi", "thai", "mexican", "chinese food", "delivery", "instacart", "postmates",
             "burger", "pizza", "sandwich", "deli", "market", "bakery", "breakfast", "lunch", "dinner"],
    "Payments": ["autopay", "bill pay", "online banking", "bank transfer", "wire transfer", "payment",
                "ach", "bill", "monthly payment", "automatic payment", "electronic payment", "check payment",
                "money transfer", "funds transfer", "transaction", "withdrawal saving", "transfer to", "transfer from"],
    "Shopping": ["amazon", "fred s place", "plantation", "texttra", "target", "walmart", "costco", "best buy", 
                "ebay", "etsy", "ikea", "home depot", "clothing", "shoes", "apparel", "online purchase",
                "retail", "store", "shop", "marketplace", "mall", "outlet", "purchase", "order", "merchant"],
    "Travel": ["uber *trip", "hotel", "airbnb", "airline", "flight", "delta air", "united air", "southwest air", 
              "car rental", "hertz", "enterprise", "train", "amtrak", "expedia", "booking.com", "trip",
              "vacation", "resort", "motel", "lodging", "airfare", "transportation", "travel", "airport", "taxi", 
              "lyft", "uber trip", "uber (trip)"],
    "Fees": ["fee", "foreign transaction", "atm fee", "overdraft", "service charge", "annual fee", "late fee",
            "finance charge", "interest charge", "penalty fee", "membership fee", "account fee", "maintenance fee",
            "processing fee", "transaction fee", "bank fee", "card fee", "surcharge", "charge"],
    "Subscriptions": ["chatgpt subscr", "medium monthly", "subscription", "monthly", "yearly", "recurring", 
                     "membership", "access fee", "premium", "plus plan", "member", "subscribe", "renewal",
                     "auto pay", "service", "plan", "monthly service", "monthly fee", "annual", "annual plan"],
    "Insurance": ["csaa ig", "aaa paymnt", "insurance", "premium", "coverage", "policy", "geico", "state farm",
                 "progressive", "allstate", "health insurance", "auto insurance", "life insurance", "medical",
                 "dental", "vision", "policy payment", "protection plan", "warranty", "security", "assurance"],
    "Investments": ["robinhood", "coinbase", "wealthfront", "vanguard", "fidelity", "schwab", "etrade", 
                   "td ameritrade", "crypto", "bitcoin", "ethereum", "401k", "ira", "investment", "stock",
                   "mutual fund", "etf", "bond", "securities", "brokerage", "portfolio", "capital", "dividend",
                   "retirement", "savings", "trading", "asset"],
    "Rent": ["venmo - payment", "rent", "mortgage", "apartment", "property", "landlord", "housing", "lease", 
            "zelle", "real estate", "management", "leasing", "home payment", "condo fee", "housing payment",
            "rent payment", "apartment fee", "property management", "residence", "accommodation"],
    "Income": ["payroll", "direct deposit", "salary", "wage", "income", "kforce", "paycheck", "payment from",
               "payment received", "cash app payment", "venmo payment", "deposit", "tax refund", "refund",
               "reimbursement", "rebate", "cashback", "bonus", "commission"]
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

def fuzzy_match(rule_text, transaction_text, threshold=80):
    """
    Perform fuzzy matching between rule keyword and transaction description.
    
    Args:
        rule_text: The rule keyword text to look for
        transaction_text: The transaction description to search in
        threshold: Minimum score (0-100) required for a match
        
    Returns:
        Boolean indicating if the match meets the threshold
    """
    # Convert inputs to strings and lowercase
    rule_text = str(rule_text).lower().strip()
    transaction_text = str(transaction_text).lower().strip()
    
    # Skip empty strings
    if not rule_text or not transaction_text:
        return False
        
    # First try exact matching (faster)
    if rule_text in transaction_text:
        return True
        
    # Use partial_ratio for substring-like matching
    # This checks if rule_text is approximately a substring of transaction_text
    score = fuzz.partial_ratio(rule_text, transaction_text)
    
    # For very short rule texts (1-2 chars), require higher threshold
    if len(rule_text) <= 2:
        adjusted_threshold = max(threshold, 90)  # Require at least 90% for very short strings
        return score >= adjusted_threshold
        
    return score >= threshold

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
    """Process multiple descriptions in a single LLM call with detailed logging and batch size control."""
    # Get the global max_batch_size if it exists, otherwise use default
    max_batch_size = 50  # Default batch size
    # Try to get the batch size from the global scope
    if 'max_batch_size' in globals():
        max_batch_size = globals()['max_batch_size']
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
    
    # If there are descriptions not in cache, process them in batches of max_batch_size
    if descriptions_to_process:
        # Process in smaller batches
        for start in range(0, len(descriptions_to_process), max_batch_size):
            batch = descriptions_to_process[start:start + max_batch_size]
            batch_indices = indices_to_process[start:start + max_batch_size]
            
            logger.info(f"Processing batch {start//max_batch_size + 1} with {len(batch)} descriptions (max batch size: {max_batch_size})")
            
            prompt = f"""
            Standardize {len(batch)} descriptions into concise formats (max 5 words each).
            Return exactly {len(batch)} lines, one per description, no JSON or explanations.
            Input: {json.dumps(batch)}
            """
            
            # Create a context identifier for logs
            context_id = f"norm_batch:{start}-{start+len(batch)}"
            
            try:
                logger.info(f"Calling LLM for batch normalization of {len(batch)} descriptions")
                
                # Log a sample of descriptions being processed
                if len(batch) > 0:
                    sample_size = min(3, len(batch))
                    sample_descriptions = batch[:sample_size]
                    logger.debug(f"Sample descriptions to normalize: {sample_descriptions}")
                
                # Use our retry-enabled function with context
                response = call_llm(
                    prompt,
                    model="gpt-3.5-turbo",  # Using cheaper model for simple normalization
                    max_tokens=150,
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
                if len(lines) != len(batch):
                    mismatch_msg = f"Batch mismatch: Expected {len(batch)}, got {len(lines)}. Adjusting."
                    logger.warning(mismatch_msg)
                    
                    if len(lines) < len(batch):
                        # Fill in missing entries with basic normalization
                        missing_count = len(batch) - len(lines)
                        logger.warning(f"Missing {missing_count} descriptions in batch response. Using fallback normalization.")
                        
                        # Log the descriptions that need fallback normalization
                        missing_descriptions = batch[len(lines):]
                        logger.debug(f"Descriptions requiring fallback: {missing_descriptions}")
                        
                        # Use basic normalization for missing entries
                        basic_normalized = [normalize_description(desc) for desc in missing_descriptions]
                        lines.extend(basic_normalized)
                        
                        # Log the fallback normalizations
                        logger.debug(f"Fallback normalizations: {basic_normalized}")
                    else:
                        # Truncate extra entries
                        extra_count = len(lines) - len(batch)
                        logger.warning(f"Got {extra_count} extra lines in batch response. Truncating.")
                        
                        # Log the extra entries being truncated
                        truncated_lines = lines[len(batch):]
                        logger.debug(f"Truncated lines: {truncated_lines}")
                        
                        lines = lines[:len(batch)]
                
                # Add normalized descriptions to results and cache
                for idx, (batch_idx, result_idx) in enumerate(zip(range(len(batch)), batch_indices)):
                    normalized_desc = lines[idx] if idx < len(lines) else normalize_description(batch[batch_idx])
                    results[result_idx] = normalized_desc
                    
                    # Cache this result
                    desc = batch[batch_idx]
                    cache_key = hashlib.md5(desc.encode()).hexdigest()
                    
                    # Note whether this came from exact match or fallback
                    was_fallback = idx >= len(lines)
                    
                    NORMALIZE_CACHE[cache_key] = {
                        'result': normalized_desc,
                        'timestamp': datetime.now().isoformat(),
                        'original': desc,
                        'hits': 1,
                        'batch_processed': True,
                        'fallback_used': was_fallback
                    }
                
                # Save cache after processing each batch
                if start % (max_batch_size * 2) == 0:
                    save_cache(NORMALIZE_CACHE, NORMALIZE_CACHE_FILE)
                
            except Exception as e:
                # Log the error with detailed information
                error_msg = f"Error in batch description normalization: {str(e)}"
                logger.error(error_msg)
                logger.exception(f"Exception details for batch normalization:")
                
                # Print a shorter message to console
                print(f"Error in batch description normalization: {e}")
                print(f"Falling back to basic normalization for this batch")
                
                # Fall back to basic normalization for all remaining in this batch
                for batch_idx, result_idx in enumerate(batch_indices):
                    desc = batch[batch_idx]
                    basic_result = normalize_description(desc)
                    results[result_idx] = basic_result
                    
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
                
                # Save cache even when fallback is used
                save_cache(NORMALIZE_CACHE, NORMALIZE_CACHE_FILE)
    
    # Convert results dict to ordered list
    ordered_results = [results[i] for i in range(len(descriptions))]
    return ordered_results

def generate_rule_suggestions(df, threshold=0.7):
    """Generate and save new categorization rules based on uncategorized and corrected transactions."""
    uncategorized = df[df["Category"] == "Uncategorized"]
    if len(uncategorized) + len(CORRECTIONS) < 3:
        print(f"Not enough data ({len(uncategorized)} uncategorized, {len(CORRECTIONS)} corrections) to generate rules.")
        return False
        
    # Log information about uncategorized transactions and corrections
    logger.info(f"Generating rules for {len(uncategorized)} uncategorized and {len(CORRECTIONS)} corrected transactions")
    
    # Display summary of uncategorized transactions to the user
    print(f"\nAnalyzing {len(uncategorized)} uncategorized and {len(CORRECTIONS)} corrected transactions...")
    
    # Group similar uncategorized transactions
    if 'NormalizedDesc' in uncategorized.columns:
        pattern_groups = uncategorized.groupby('NormalizedDesc').size().reset_index(name='count')
        pattern_groups = pattern_groups.sort_values('count', ascending=False)
        
        if len(pattern_groups) > 0:
            print("\nMost common uncategorized transaction patterns:")
            for idx, row in pattern_groups.head(5).iterrows():
                print(f"  {row['NormalizedDesc']}: {row['count']} transactions")
    
    # Prepare corrections as a DataFrame-like JSON for consistency
    corrections_json = json.dumps(CORRECTIONS, default=str) if CORRECTIONS else "[]"
    
    prompt = f"""
    Analyze these transactions to suggest categorization rules:
    - Uncategorized transactions: {uncategorized.to_json(orient='records')}
    - Corrected transactions: {corrections_json}

    For corrected transactions, pay special attention to the 'original_category' and 'new_category' fields
    that show how users manually recategorized transactions.
    
    Use existing categories: {list(CATEGORY_RULES.keys())}
    
    Return a JSON ARRAY of rule suggestions in this EXACT format:
    [
      {
        "description": "keyword from transaction description",
        "amount": null,  // null if variable, or specific amount like 25.99
        "day_range": [1, 5],  // start day and end day as integers
        "category": "Category",  // category from the list above
        "confidence": 0.9,  // confidence as decimal between 0 and 1
        "reasoning": "Brief explanation"
      },
      // Add more suggestions here
    ]
    
    IMPORTANT: 
    - Use PROPER JSON syntax with double quotes for keys
    - Make sure the confidence value is a number (not a string)
    - Make sure day_range is a proper array with two integers
    - Include complete closing brackets and braces
    - Only include rules with confidence > {threshold}
    - Aim for at least 3 suggestions, prioritizing frequent patterns
    """
    try:
        # Use our retry-enabled function
        response = call_llm(
            prompt,
            model="gpt-4o",
            max_tokens=200,
            temperature=0.2,
            context="rule_suggestions_with_corrections"
        )
        content = response.choices[0].message.content.strip()
        
        # The LLM might return a JSON array or individual JSON objects
        # Try wrapping the content in brackets if it doesn't start with [
        if not content.strip().startswith('['):
            # Check if it looks like JSON objects separated by commas
            if '{' in content and '},' in content:
                content = f"[{content}]"
                logger.info("Wrapped content in array brackets to fix JSON format")
            
        # Use our robust JSON parsing function - now expecting an array
        try:
            # First, we'll log the complete content to better understand what we're parsing
            logger.info(f"Original JSON content length: {len(content)}")
            logger.debug(f"Content to parse: {content[:500]}...")
            
            # Clean up the content for better parsing chance
            # Sometimes LLMs return double-wrapped arrays or add extra brackets
            cleaned_content = content.strip()
            if cleaned_content.startswith('[[') and cleaned_content.endswith(']]'):
                # Fix double-wrapped arrays
                logger.info("Detected double-wrapped array, fixing format")
                cleaned_content = cleaned_content[1:-1]
                
            # Try additional unwrapping if it's still malformed
            if cleaned_content.startswith('[') and not cleaned_content.endswith(']'):
                logger.info("Detected unclosed array, attempting to fix")
                cleaned_content = cleaned_content + ']'
                
            # Fix truncated content (common issue with rule suggestions)
            if "confidence" in cleaned_content and cleaned_content.endswith('0'):
                logger.info("Detected truncated JSON, attempting to fix")
                # Add closing brackets to potentially fix truncated JSON
                if cleaned_content.count('{') > cleaned_content.count('}'):
                    missing_braces = cleaned_content.count('{') - cleaned_content.count('}')
                    cleaned_content = cleaned_content + '}'*missing_braces
                if cleaned_content.count('[') > cleaned_content.count(']'):
                    missing_brackets = cleaned_content.count('[') - cleaned_content.count(']')
                    cleaned_content = cleaned_content + ']'*missing_brackets
                    
            # Use our robust JSON parsing with the cleaned content
            suggestions = safe_json_parse(
                cleaned_content,
                fallback=[],
                context="rule suggestions"
            )
            
            # If still empty, try an alternative approach - parse each suggestion separately
            if not suggestions and '{' in content:
                logger.info("First parsing attempt failed, trying to extract individual JSON objects")
                suggestions = []
                # Find all JSON objects in the content
                starts = [m.start() for m in re.finditer(r'{\s*"', content)]
                for i, start in enumerate(starts):
                    # Find the end of this object
                    end = content.find('}", ', start)
                    if end < 0:
                        end = content.find('}', start)
                    if end > 0:
                        # Extract and parse the individual object
                        obj_str = content[start:end+1]
                        try:
                            obj = json.loads(obj_str)
                            suggestions.append(obj)
                            logger.info(f"Successfully extracted object {i+1}")
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse extracted object {i+1}: {obj_str[:50]}...")
            
        except Exception as e:
            logger.error(f"Error preprocessing JSON content: {str(e)}")
            suggestions = []
        
        # Make sure we have a list of suggestions
        if not isinstance(suggestions, list):
            # Single object returned
            suggestions = [suggestions]
            
        # Log what we parsed
        logger.info(f"Parsed {len(suggestions)} rule suggestions")
        
        # Filter out suggestions without a category or with 'Uncategorized'
        valid_suggestions = []
        for suggestion in suggestions:
            # Skip invalid suggestions
            if not isinstance(suggestion, dict):
                logger.warning(f"Skipping non-dict suggestion: {suggestion}")
                continue
                
            if suggestion.get("category") and suggestion["category"] != "Uncategorized":
                # Skip suggestions that are missing required fields
                if "description" not in suggestion:
                    logger.warning(f"Skipping suggestion without description: {suggestion}")
                    continue
                    
                # Format the suggestion to include confidence and reasoning if missing
                if "confidence" not in suggestion:
                    suggestion["confidence"] = threshold + 0.1
                if "reasoning" not in suggestion:
                    suggestion["reasoning"] = "Pattern identified in transactions"
                    
                # Store just the fields needed for rules
                rule = {
                    "description": suggestion["description"],
                    "amount": suggestion.get("amount"),
                    "day_range": suggestion.get("day_range", [1, 31]),
                    "category": suggestion["category"]
                }
                
                valid_suggestions.append(rule)
                CUSTOM_RULES.append(rule)
                
                # Log and show the suggestion with its reasoning
                print(f"Added new rule: '{rule['description']}' → {rule['category']} ({suggestion.get('confidence', 0.7):.1f})")
                if "reasoning" in suggestion:
                    print(f"  Reasoning: {suggestion['reasoning']}")
                
        # Save the updated rules
        if valid_suggestions:
            with open("categories.json", "w") as f:
                json.dump({"rules": CUSTOM_RULES}, f, indent=4)
            print(f"\nAdded {len(valid_suggestions)} new rules to categories.json")
            
            # Ask if user wants to re-categorize now
            recategorize = input("\nRe-categorize transactions with new rules? ").lower().strip()
            return any(yes_word in recategorize for yes_word in ['y', 'yes', 'yeah', 'sure', 'ok', 'please', 'do it'])
        else:
            print("No valid rule suggestions found.")
            return False
    except Exception as e:
        print(f"Error generating rule suggestions: {e}")
        return False
        
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
    
    add_rules = input("\nAdd these sample rules to categories.json? ").lower().strip()
    if any(yes_word in add_rules for yes_word in ['y', 'yes', 'yeah', 'sure', 'ok', 'please', 'do it']):
        # Merge with existing rules
        combined_rules = CUSTOM_RULES + suggestions
        with open("categories.json", "w") as f:
            json.dump({"rules": combined_rules}, f, indent=4)
        print(f"Added {len(suggestions)} new rule templates to categories.json")
    
    return

def interactive_review(df):
    """Allow users to review and correct LLM categorization suggestions with natural language support."""
    print("\n=== Starting Interactive Review of Low Confidence Transactions ===\n")
    print("This tool will help you review transactions where the AI is uncertain.")
    print("You can respond in natural language to accept, change, skip, or quit.")
    print("Examples:")
    print("  - 'Yes' or 'looks good' to accept the current category")
    print("  - 'Change to Food' or 'this should be Entertainment' to change the category")
    print("  - 'Skip' or 'next' to move to the next transaction")
    print("  - 'Quit' or 'exit' to stop the review")
    print("Any changes you make will be saved as custom rules for future transactions.\n")
    
    # Option to view rule suggestions
    show_rule_suggestions = input("Would you like to see suggestions for custom payment rules? (y/n): ").lower().strip()
    if any(answer in show_rule_suggestions for answer in ['y', 'yes', 'yeah', 'sure', 'ok']):
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
        # Focus on low confidence transactions, sorted by confidence (lowest first)
        low_confidence = df[df["Confidence"] < 0.8].copy()
        if len(low_confidence) == 0:
            print("No low-confidence transactions found. All transactions have been categorized with high confidence!")
            return
        
        # Sort by confidence (ascending) first, then by Category to group "Uncategorized" together
        review_df = low_confidence.sort_values(by=["Confidence", "Category"])
        
        # Show both total count and uncategorized count
        uncategorized_count = sum(review_df["Category"] == "Uncategorized")
        print(f"Found {len(review_df)} transactions with confidence score below 0.8")
        print(f"Of these, {uncategorized_count} are marked as 'Uncategorized' and need your input")
        print("Transactions are sorted by confidence level, showing least confident first.\n")
        
        for idx, row in review_df.iterrows():
            print("\n" + "="*80)
            print(f"Transaction: {row['Description']}")
            print(f"Normalized: {row['NormalizedDesc']}")
            print(f"Amount: ${row['Amount']:.2f}, Date: {row['Date']}, Source: {row['Source']}")
            print(f"Current Category: {row['Category']} (Confidence: {row['Confidence']:.2f})")
            
            # For previously uncategorized transactions, add clearer instruction
            if row['Category'] == 'Uncategorized':
                print("This transaction needs categorization.")
                
                # If the reasoning contains the original category suggestion, extract and display it
                if 'Original reasoning' in row['Reasoning']:
                    # Try to extract the original AI suggestion if available
                    match = re.search(r"Low confidence \([0-9.]+\) for '([^']+)'", row['Reasoning'])
                    if match:
                        suggested_category = match.group(1)
                        print(f"AI suggested '{suggested_category}' with low confidence")
            
            print(f"AI Reasoning: {row['Reasoning']}")
            
            # Use natural language prompt and parse the response
            user_response = input("\nWhat would you like to do with this transaction? ").lower().strip()
            
            # Parse the user's natural language response
            if any(quit_word in user_response for quit_word in ['quit', 'exit', 'stop', 'end', 'done', 'q']):
                print("Exiting review mode...")
                break
                
            elif any(change_word in user_response for change_word in ['change', 'edit', 'update', 'correct', 'modify', 'should be', 'change to', 'make it']):
                # Try to extract category from the response
                extracted_category = None
                
                # Check if any category name is mentioned in the response
                # First look for explicit patterns like "change to Food" or "should be Entertainment"
                for pattern in ["change to ", "should be ", "make it ", "change it to ", "set as ", "set to ", "categorize as "]:
                    if pattern in user_response:
                        # Get the text after the pattern
                        text_after = user_response.split(pattern)[1].strip()
                        # Find the best category match
                        for category in valid_categories:
                            if category.lower() in text_after.lower():
                                extracted_category = category
                                break
                        if extracted_category:
                            break
                
                # If not found with patterns, look for any category name in the response
                if not extracted_category:
                    for category in valid_categories:
                        if category.lower() in user_response:
                            extracted_category = category
                            break
                
                # If we couldn't extract a category, ask explicitly
                if not extracted_category:
                    print("\nEnter the number or name of the new category:")
                    for i, category in enumerate(valid_categories, 1):
                        print(f"{i}. {category}")
                    
                    category_input = input("\nNew category: ").strip()
                    
                    # Handle numeric input
                    if category_input.isdigit() and 1 <= int(category_input) <= len(valid_categories):
                        extracted_category = valid_categories[int(category_input) - 1]
                    else:
                        # Check if input matches a category name (case-insensitive)
                        category_match = next((cat for cat in valid_categories if cat.lower() == category_input.lower()), None)
                        if category_match:
                            extracted_category = category_match
                        else:
                            print(f"Invalid category. Using 'Uncategorized' instead.")
                            extracted_category = "Uncategorized"
                
                # Store the original category
                original_category = row["Category"]
                
                # Update dataframe with the new category
                df.at[idx, "Category"] = extracted_category
                df.at[idx, "Confidence"] = 1.0
                df.at[idx, "Reasoning"] = "User override during interactive review"
                
                # Record the correction in the global CORRECTIONS list
                correction = {
                    "transaction": row.to_dict(),
                    "original_category": original_category,
                    "new_category": extracted_category
                }
                CORRECTIONS.append(correction)
                logger.info(f"User correction recorded: '{row['Description']}' from {original_category} to {extracted_category}")
                
                # Ask about creating a rule
                create_rule_response = input("\nWould you like to create a rule for similar transactions in the future? ").lower().strip()
                if any(yes_word in create_rule_response for yes_word in ['y', 'yes', 'yeah', 'sure', 'ok', 'create', 'rule']):
                    # Extract key terms from description
                    desc_words = row["Description"].lower().split()
                    key_term_prompt = f"Enter a keyword from the description to match similar transactions (default: {desc_words[0]}): "
                    key_term_response = input(key_term_prompt).strip()
                    key_term = key_term_response if key_term_response else desc_words[0].lower()
                    
                    # Ask about amount matching
                    amount_prompt = f"Should this rule only match transactions with amount ${row['Amount']:.2f}? "
                    amount_response = input(amount_prompt).lower().strip()
                    amount = float(row["Amount"]) if any(yes_word in amount_response for yes_word in ['y', 'yes', 'yeah', 'sure', 'ok']) else None
                    
                    # Ask about day range
                    day = pd.to_datetime(row["Date"]).day
                    day_range_prompt = f"Enter day range for the rule (default: {day-2}-{day+2}, or 'any' for full month): "
                    day_range_response = input(day_range_prompt).strip()
                    
                    if any(word in day_range_response.lower() for word in ['any', 'all', 'full', 'month', 'every']):
                        day_range = [1, 31]
                    elif "-" in day_range_response:
                        try:
                            start, end = map(int, day_range_response.split("-"))
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
                        "category": extracted_category
                    }
                    
                    CUSTOM_RULES.append(new_rule)
                    changes_made = True
                    print(f"Added new rule: {new_rule}")
                
                print(f"Updated category to: {extracted_category}")
                
            # Skip logic
            elif any(skip_word in user_response for skip_word in ['skip', 'next', 'pass', 'continue', 's']):
                print("Skipping to next transaction.")
            
            # Accept logic - anything that sounds positive and isn't covered by other commands
            elif any(accept_word in user_response for accept_word in [
                'y', 'yes', 'accept', 'good', 'correct', 'agree', 'approve', 'right', 
                'fine', 'ok', 'okay', 'sure', 'looks good', 'keep', 'makes sense'
            ]):
                print("Accepted current category.")
            
            # Default case if we couldn't parse the command
            else:
                print("I'm not sure what you want to do. Skipping to the next transaction.")
                print("Tip: You can say 'yes' to accept, 'change to Food' to change category, 'skip' to move on, or 'quit' to exit.")
        
        if changes_made:
            with open("categories.json", "w") as f:
                json.dump({"rules": CUSTOM_RULES}, f, indent=4)
            print("\nSaved updated rules to categories.json")
        
        # Save corrections for future rule generation
        if CORRECTIONS:
            save_cache(CORRECTIONS, CORRECTIONS_CACHE_FILE)
            print(f"Saved {len(CORRECTIONS)} user corrections for future rule generation")
        
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
        
        # Save corrections even in case of interruption
        if CORRECTIONS:
            save_cache(CORRECTIONS, CORRECTIONS_CACHE_FILE)
            print(f"Saved {len(CORRECTIONS)} user corrections for future rule generation")
            
        df.to_csv(DEDUPE_CSV, index=False)
        print(f"Saved updates to {DEDUPE_CSV}")

def generate_rules_from_corrections():
    """Generate rules based on user corrections collected during interactive review."""
    global CORRECTIONS
    
    if not CORRECTIONS:
        print("No user corrections available to generate rules from.")
        return False
    
    print(f"\nAnalyzing {len(CORRECTIONS)} user corrections to generate intelligent rules...")
    
    # Group corrections by new category to identify patterns
    by_category = {}
    for correction in CORRECTIONS:
        new_category = correction["new_category"]
        if new_category not in by_category:
            by_category[new_category] = []
        by_category[new_category].append(correction)
    
    # Show summary of corrections
    print("\nCorrections by category:")
    for category, corrections in by_category.items():
        print(f"  {category}: {len(corrections)} corrections")
    
    # Prepare the prompt with detailed information about the corrections
    prompt = f"""
    Analyze these user corrections where the user manually changed AI-suggested categories:
    {json.dumps(CORRECTIONS, default=str)}
    
    IMPORTANT INSTRUCTIONS:
    1. Identify common patterns in transactions that the user consistently recategorizes
    2. Look for keywords in descriptions that could be used as rules
    3. Pay attention to amount patterns (e.g., consistent amounts for rent)
    4. Consider date patterns (e.g., recurring expenses at specific days of month)
    5. Focus especially on corrections where the AI repeatedly made the same mistake
    
    Return a JSON ARRAY of rule suggestions in this EXACT format:
    [
      {
        "description": "keyword from transaction description",
        "amount": null,  // null if variable, or specific amount like 25.99
        "day_range": [1, 5],  // start day and end day as integers
        "category": "Category",  // category from the list above
        "confidence": 0.9,  // confidence as decimal between 0 and 1
        "reasoning": "Brief explanation of why this rule aligns with user's categorization preferences"
      },
      // Add more suggestions here
    ]
    
    IMPORTANT: 
    - Use PROPER JSON syntax with double quotes for keys
    - Make sure the confidence value is a number (not a string)
    - Make sure day_range is a proper array with two integers
    - Include complete closing brackets and braces
    - ONLY include rules with confidence >0.85
    - Create at least 3 rule suggestions if possible, focusing on the most consistent correction patterns
    - Choose words that uniquely identify the merchant or transaction type
    """
    
    try:
        # Use our retry-enabled function with special context
        response = call_llm(
            prompt,
            model="gpt-4o",  # Use the more capable model for this analysis
            max_tokens=400,
            temperature=0.2,
            context="user_corrections_analysis"
        )
        content = response.choices[0].message.content.strip()
        
        # The LLM might return a JSON array or individual JSON objects
        if not content.strip().startswith('['):
            # Check if it looks like JSON objects separated by commas
            if '{' in content and '},' in content:
                content = f"[{content}]"
                logger.info("Wrapped content in array brackets to fix JSON format for correction-based rules")
        
        # Apply same robust JSON parsing as in generate_rule_suggestions
        try:
            # Log the content for debugging
            logger.info(f"Original JSON content length: {len(content)}")
            logger.debug(f"Content to parse: {content[:500]}...")
            
            # Clean up the content for better parsing chance
            cleaned_content = content.strip()
            if cleaned_content.startswith('[[') and cleaned_content.endswith(']]'):
                # Fix double-wrapped arrays
                logger.info("Detected double-wrapped array, fixing format")
                cleaned_content = cleaned_content[1:-1]
                
            # Try additional unwrapping if it's still malformed
            if cleaned_content.startswith('[') and not cleaned_content.endswith(']'):
                logger.info("Detected unclosed array, attempting to fix")
                cleaned_content = cleaned_content + ']'
                
            # Fix truncated content
            if "confidence" in cleaned_content and cleaned_content.endswith('0'):
                logger.info("Detected truncated JSON, attempting to fix")
                # Add closing brackets to potentially fix truncated JSON
                if cleaned_content.count('{') > cleaned_content.count('}'):
                    missing_braces = cleaned_content.count('{') - cleaned_content.count('}')
                    cleaned_content = cleaned_content + '}'*missing_braces
                if cleaned_content.count('[') > cleaned_content.count(']'):
                    missing_brackets = cleaned_content.count('[') - cleaned_content.count(']')
                    cleaned_content = cleaned_content + ']'*missing_brackets
                    
            # Use our robust JSON parsing with the cleaned content
            suggestions = safe_json_parse(
                cleaned_content,
                fallback=[],
                context="user correction rules"
            )
            
            # If still empty, try an alternative approach - parse each suggestion separately
            if not suggestions and '{' in content:
                logger.info("First parsing attempt failed, trying to extract individual JSON objects")
                suggestions = []
                # Find all JSON objects in the content
                starts = [m.start() for m in re.finditer(r'{\s*"', content)]
                for i, start in enumerate(starts):
                    # Find the end of this object
                    end = content.find('}", ', start)
                    if end < 0:
                        end = content.find('}', start)
                    if end > 0:
                        # Extract and parse the individual object
                        obj_str = content[start:end+1]
                        try:
                            obj = json.loads(obj_str)
                            suggestions.append(obj)
                            logger.info(f"Successfully extracted object {i+1}")
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse extracted object {i+1}: {obj_str[:50]}...")
            
        except Exception as e:
            logger.error(f"Error preprocessing JSON content: {str(e)}")
            suggestions = []
        
        if not isinstance(suggestions, list):
            suggestions = [suggestions]
            
        # Log what we parsed    
        logger.info(f"Parsed {len(suggestions)} rule suggestions from user corrections")
        
        # Process and add valid rule suggestions
        valid_suggestions = []
        new_rules_added = 0
        
        print("\nGenerated rule suggestions from user corrections:")
        
        for suggestion in suggestions:
            # Skip invalid objects
            if not isinstance(suggestion, dict):
                logger.warning(f"Skipping non-dict suggestion: {suggestion}")
                continue
                
            if suggestion.get("category") and suggestion["category"] != "Uncategorized":
                # Format the suggestion to include required fields
                if "confidence" not in suggestion:
                    suggestion["confidence"] = 0.9  # Default high confidence for user-derived rules
                
                # Skip low confidence suggestions
                if suggestion.get("confidence", 0) < 0.85:
                    continue
                
                # Check if a similar rule already exists
                similar_rule_exists = False
                for rule in CUSTOM_RULES:
                    if (rule.get("description") == suggestion.get("description") and 
                        rule.get("category") == suggestion.get("category")):
                        similar_rule_exists = True
                        break
                
                if not similar_rule_exists:
                    # Create a clean rule without extra fields
                    rule = {
                        "description": suggestion["description"],
                        "amount": suggestion.get("amount"),
                        "day_range": suggestion.get("day_range", [1, 31]),
                        "category": suggestion["category"]
                    }
                    
                    valid_suggestions.append(rule)
                    CUSTOM_RULES.append(rule)
                    new_rules_added += 1
                    
                    # Print details about the new rule
                    print(f"Added rule: '{rule['description']}' → {rule['category']} ({suggestion.get('confidence', 0.9):.1f})")
                    if "reasoning" in suggestion:
                        print(f"  Reasoning: {suggestion['reasoning']}")
        
        if new_rules_added > 0:
            with open("categories.json", "w") as f:
                json.dump({"rules": CUSTOM_RULES}, f, indent=4)
            print(f"\nAdded {new_rules_added} new rules based on your previous corrections")
            
            # Clear the corrections after processing them
            CORRECTIONS = []
            # Save the empty corrections list to disk
            save_cache(CORRECTIONS, CORRECTIONS_CACHE_FILE)
            
            # Ask if user wants to re-categorize now
            recategorize = input("\nRe-categorize transactions with these new rules? ").lower().strip()
            return any(yes_word in recategorize for yes_word in ['y', 'yes', 'yeah', 'sure', 'ok', 'please', 'do it'])
        else:
            print("No additional rules could be derived from user corrections.")
            return False
    except Exception as e:
        print(f"Error generating rules from user corrections: {e}")
        logger.error(f"Error in generate_rules_from_corrections: {e}")
        return False

def generate_low_confidence_rules(df, confidence_threshold=0.7):
    """Generate rules for transactions with low confidence scores."""
    low_confidence = df[(df["Confidence"] < confidence_threshold) & (df["Category"] != "Uncategorized")]
    if len(low_confidence) < 3:  # Require a minimum number of low confidence transactions
        print(f"Not enough low confidence transactions ({len(low_confidence)}) to generate rules.")
        return False
    
    # Log information about low confidence transactions
    logger.info(f"Generating rules for {len(low_confidence)} low confidence transactions")
    
    # Display summary of low confidence transactions to the user
    print(f"\nFound {len(low_confidence)} transactions with confidence below {confidence_threshold}")
    
    # Show distribution of categories in low confidence transactions
    category_counts = low_confidence["Category"].value_counts()
    print("\nCategory distribution for low confidence transactions:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count}")
    
    # Group by category and count to identify common patterns
    if 'NormalizedDesc' in low_confidence.columns:
        # Create composite key from category and description to find common patterns
        low_confidence['CategoryDesc'] = low_confidence['Category'] + ': ' + low_confidence['NormalizedDesc']
        pattern_groups = low_confidence.groupby('CategoryDesc').size().reset_index(name='count')
        pattern_groups = pattern_groups.sort_values('count', ascending=False)
        
        if len(pattern_groups) > 0:
            print("\nMost common low confidence transaction patterns:")
            for idx, row in pattern_groups.head(5).iterrows():
                print(f"  {row['CategoryDesc']}: {row['count']} transactions")
    
    prompt = f"""
    Analyze these low-confidence transactions: {low_confidence.to_json(orient='records')}.
    
    IMPORTANT INSTRUCTIONS:
    1. Find patterns where the model assigned a category with low confidence, but the category is likely correct
    2. Focus on creating rules that can improve confidence for these patterns in the future
    3. Look for common merchants, transaction types, or patterns across multiple transactions
    4. Consider consistent patterns in amounts, descriptions, or dates
    
    Return a JSON ARRAY of rule suggestions in this EXACT format:
    [
      {
        "description": "keyword from transaction description",
        "amount": null,  // null if variable, or specific amount like 25.99
        "day_range": [1, 5],  // start day and end day as integers
        "category": "Category",  // category from the list above
        "confidence": 0.9,  // confidence as decimal between 0 and 1
        "reasoning": "Brief explanation for why this rule would improve categorization"
      },
      // Add more suggestions here
    ]
    
    IMPORTANT: 
    - Use PROPER JSON syntax with double quotes for keys
    - Make sure the confidence value is a number (not a string)
    - Make sure day_range is a proper array with two integers
    - Include complete closing brackets and braces
    - ONLY include rules with confidence >0.8
    - If you see any major miscategorizations, note them in your reasoning
    - Create at least 5 rule suggestions if possible, focusing on the most frequent patterns first
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
        
        # The LLM might return a JSON array or individual JSON objects
        # Try wrapping the content in brackets if it doesn't start with [
        if not content.strip().startswith('['):
            # Check if it looks like JSON objects separated by commas
            if '{' in content and '},' in content:
                content = f"[{content}]"
                logger.info("Wrapped content in array brackets to fix JSON format for low confidence rules")
            
        # Apply same robust JSON parsing as in generate_rule_suggestions
        try:
            # Log the content for debugging
            logger.info(f"Original JSON content length: {len(content)}")
            logger.debug(f"Content to parse: {content[:500]}...")
            
            # Clean up the content for better parsing chance
            cleaned_content = content.strip()
            if cleaned_content.startswith('[[') and cleaned_content.endswith(']]'):
                # Fix double-wrapped arrays
                logger.info("Detected double-wrapped array, fixing format")
                cleaned_content = cleaned_content[1:-1]
                
            # Try additional unwrapping if it's still malformed
            if cleaned_content.startswith('[') and not cleaned_content.endswith(']'):
                logger.info("Detected unclosed array, attempting to fix")
                cleaned_content = cleaned_content + ']'
                
            # Fix truncated content
            if "confidence" in cleaned_content and cleaned_content.endswith('0'):
                logger.info("Detected truncated JSON, attempting to fix")
                # Add closing brackets to potentially fix truncated JSON
                if cleaned_content.count('{') > cleaned_content.count('}'):
                    missing_braces = cleaned_content.count('{') - cleaned_content.count('}')
                    cleaned_content = cleaned_content + '}'*missing_braces
                if cleaned_content.count('[') > cleaned_content.count(']'):
                    missing_brackets = cleaned_content.count('[') - cleaned_content.count(']')
                    cleaned_content = cleaned_content + ']'*missing_brackets
                    
            # Use our robust JSON parsing with the cleaned content
            suggestions = safe_json_parse(
                cleaned_content,
                fallback=[],
                context="low confidence rules"
            )
            
            # If still empty, try an alternative approach - parse each suggestion separately
            if not suggestions and '{' in content:
                logger.info("First parsing attempt failed, trying to extract individual JSON objects")
                suggestions = []
                # Find all JSON objects in the content
                starts = [m.start() for m in re.finditer(r'{\s*"', content)]
                for i, start in enumerate(starts):
                    # Find the end of this object
                    end = content.find('}", ', start)
                    if end < 0:
                        end = content.find('}', start)
                    if end > 0:
                        # Extract and parse the individual object
                        obj_str = content[start:end+1]
                        try:
                            obj = json.loads(obj_str)
                            suggestions.append(obj)
                            logger.info(f"Successfully extracted object {i+1}")
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse extracted object {i+1}: {obj_str[:50]}...")
            
        except Exception as e:
            logger.error(f"Error preprocessing JSON content: {str(e)}")
            suggestions = []
        
        if not isinstance(suggestions, list):
            suggestions = [suggestions]
            
        # Log what we parsed    
        logger.info(f"Parsed {len(suggestions)} low confidence rule suggestions")
        
        # Filter valid suggestions and check for duplicates
        valid_suggestions = []
        new_rules_added = 0
        
        print("\nAnalyzing low confidence patterns...")
        
        for suggestion in suggestions:
            # Skip invalid objects
            if not isinstance(suggestion, dict):
                logger.warning(f"Skipping non-dict suggestion: {suggestion}")
                continue
                
            if suggestion.get("category") and suggestion["category"] != "Uncategorized":
                # Format the suggestion to include required fields
                if "confidence" not in suggestion:
                    suggestion["confidence"] = 0.85  # Default high confidence
                
                # Skip low confidence suggestions
                if suggestion.get("confidence", 0) < 0.8:
                    continue
                    
                # Check if a similar rule already exists
                similar_rule_exists = False
                for rule in CUSTOM_RULES:
                    if (rule.get("description") == suggestion.get("description") and 
                        rule.get("category") == suggestion.get("category")):
                        similar_rule_exists = True
                        break
                
                if not similar_rule_exists:
                    # Create a clean rule without extra fields
                    rule = {
                        "description": suggestion["description"],
                        "amount": suggestion.get("amount"),
                        "day_range": suggestion.get("day_range", [1, 31]),
                        "category": suggestion["category"]
                    }
                    
                    valid_suggestions.append(rule)
                    CUSTOM_RULES.append(rule)
                    new_rules_added += 1
                    
                    # Print details about the new rule
                    print(f"Added rule: '{rule['description']}' → {rule['category']} ({suggestion.get('confidence', 0.8):.1f})")
                    if "reasoning" in suggestion:
                        print(f"  Reasoning: {suggestion['reasoning']}")
        
        if new_rules_added > 0:
            with open("categories.json", "w") as f:
                json.dump({"rules": CUSTOM_RULES}, f, indent=4)
            print(f"\nAdded {new_rules_added} new rules from low confidence transaction analysis")
            
            # Ask if user wants to re-categorize now
            recategorize = input("\nRe-categorize transactions with new rules? ").lower().strip()
            return any(yes_word in recategorize for yes_word in ['y', 'yes', 'yeah', 'sure', 'ok', 'please', 'do it'])
        else:
            print("No additional rules identified from low confidence transactions.")
            return False
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
    # Initialize variables to avoid reference before assignment
    examples_dict = None
    fallback_result = ("Uncategorized", 0.0, "Unable to categorize (fallback)")
    
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
    
    # Also check the simplified cache by description/amount/date
    simple_cache_key = hashlib.md5(f"{df_row['Description']}{df_row['Amount']}{df_row['Date']}".encode()).hexdigest()
    if simple_cache_key in CATEGORIZE_CACHE:
        # Track cache hits for reporting
        CATEGORIZE_CACHE[simple_cache_key]['hits'] += 1
        result = CATEGORIZE_CACHE[simple_cache_key]['result']
        logger.info(f"Simple cache hit for {df_row['Description'][:20]}...")
        return result[0], result[1], result[2]
    
    # Not in cache, so categorize with LLM
    # Include both original and normalized descriptions in the prompt
    normalized_desc = df_row["NormalizedDesc"] if "NormalizedDesc" in df_row else "Not Available"
    
    # Try to categorize using rule-based approach first before calling the API
    desc_lower = df_row['Description'].lower()
    norm_desc_lower = normalized_desc.lower() if normalized_desc != "Not Available" else ""
    
    # Get the fuzzy threshold from global args if available
    fuzzy_threshold = 80  # Default
    if 'args' in globals() and hasattr(globals()['args'], 'fuzzy_threshold'):
        fuzzy_threshold = globals()['args'].fuzzy_threshold
    
    # Check against expanded CATEGORY_RULES
    for category, keywords in CATEGORY_RULES.items():
        # First try exact matching (faster)
        if any(keyword.lower() in desc_lower for keyword in keywords) or \
           any(keyword.lower() in norm_desc_lower for keyword in keywords):
            # Found an exact match in rules, return with high confidence
            match_type = "exact"
            reasoning = f"Matched rule keyword in {'description' if any(keyword.lower() in desc_lower for keyword in keywords) else 'normalized description'}"
            result = (category, 0.95, reasoning)
            
            # Cache the result in both caches
            CATEGORIZE_CACHE[cache_key] = {
                'result': result,
                'timestamp': datetime.now().isoformat(),
                'transaction': {
                    'Description': df_row['Description'],
                    'Amount': float(df_row['Amount']),
                    'Date': df_row['Date'],
                    'Source': df_row['Source']
                },
                'hits': 1,
                'rule_based': True,
                'match_type': match_type
            }
            
            return result
        
        # If no exact match, try fuzzy matching
        for keyword in keywords:
            if (fuzzy_match(keyword.lower(), desc_lower, fuzzy_threshold) or 
                fuzzy_match(keyword.lower(), norm_desc_lower, fuzzy_threshold)):
                
                # Log the match for debugging
                logger.debug(f"Fuzzy match: '{keyword}' -> '{desc_lower}' (category: {category})")
                
                match_type = "fuzzy"
                reasoning = f"Fuzzy matched rule keyword '{keyword}' in {'description' if fuzzy_match(keyword.lower(), desc_lower, fuzzy_threshold) else 'normalized description'}"
                result = (category, 0.9, reasoning)
                
                # Cache the result in both caches
                CATEGORIZE_CACHE[cache_key] = {
                    'result': result,
                    'timestamp': datetime.now().isoformat(),
                    'transaction': {
                        'Description': df_row['Description'],
                        'Amount': float(df_row['Amount']),
                        'Date': df_row['Date'],
                        'Source': df_row['Source']
                    },
                    'hits': 1,
                    'rule_based': True,
                    'match_type': match_type
                }
                
                # Also add to simplified cache
                CATEGORIZE_CACHE[simple_cache_key] = {
                    'result': result,
                    'timestamp': datetime.now().isoformat(),
                    'hits': 1,
                    'rule_based': True,
                    'match_type': match_type
                }
                
                # Periodically save cache to disk
                if len(CATEGORIZE_CACHE) % 10 == 0:
                    save_cache(CATEGORIZE_CACHE, CATEGORIZE_CACHE_FILE)
                    
                logger.info(f"Rule-based categorization: {df_row['Description'][:30]} -> {category}")
                return result
    
    # If no rule match, proceed with LLM categorization with enhanced contextual examples
    # This will be our fallback if something goes wrong in the process
    examples_str = ""
    
    # Check if past_transactions contains our enhanced examples dictionary
    if past_transactions is not None and isinstance(past_transactions, dict) and "examples" in past_transactions:
        examples_dict = past_transactions["examples"]
        
        # Format examples from the dictionary into a string
        if examples_dict:
            examples_str = "Here are some examples of categorized transactions:\n"
            
            # Add examples for each category
            for category, example_list in examples_dict.items():
                if example_list:
                    # Include up to 3 examples per category to keep the prompt concise
                    for example in example_list[:3]:
                        examples_str += f"- {example} → {category}\n"
            
            examples_str += "\n"
            
    # If we don't have dictionary examples, try the traditional dataframe approach
    elif past_transactions is not None and (
        isinstance(past_transactions, pd.DataFrame) or 
        (isinstance(past_transactions, dict) and "dataframe" in past_transactions and isinstance(past_transactions["dataframe"], pd.DataFrame))
    ):
        # Get the dataframe from either format
        df = past_transactions if isinstance(past_transactions, pd.DataFrame) else past_transactions["dataframe"]
        
        if not df.empty:
            # Filter to include only transactions with high confidence and valid categories
            example_txns = df[
                (df['Confidence'] >= 0.9) & 
                (df['Category'].notna()) & 
                (df['Category'] != 'Uncategorized')
            ]
            
            if not example_txns.empty:
                examples_str = "Here are some examples of categorized transactions:\n"
                
                # Group by category to ensure diverse examples
                by_category = {}
                for cat in CATEGORY_RULES.keys():
                    cat_examples = example_txns[example_txns['Category'] == cat]
                    if not cat_examples.empty:
                        # Take up to 2 examples per category
                        by_category[cat] = cat_examples.sample(min(2, len(cat_examples)))
                
                # Add examples to the prompt
                for category, txns in by_category.items():
                    for _, txn in txns.iterrows():
                        desc = txn['Description']
                        amount = txn['Amount']
                        date = txn['Date']
                        examples_str += f"- '{desc}' (${amount:.2f} on {date}) → {category}\n"
                
                examples_str += "\n"
    
    prompt = f"""
    Categorize this transaction into one of: {list(CATEGORY_RULES.keys()) + ['Uncategorized']}.
    
    {examples_str}Transaction: 
    - Date: '{df_row['Date']}'
    - Original Description: '{df_row['Description']}'
    - Normalized Description: '{normalized_desc}'
    - Amount: {df_row['Amount']}
    - Source: '{df_row['Source']}'
    
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
            model="gpt-3.5-turbo",  # Use cheaper model for categorization
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
        
        # If confidence is low, try with the more powerful model
        if confidence < 0.7 and category == "Uncategorized":
            logger.info(f"Low confidence ({confidence}) with gpt-3.5, trying with gpt-4o for {context_id}")
            
            # Use the more powerful model as a fallback, unless model_downgrade is in effect
            fallback_model = "gpt-4o"  # Default to the more powerful model
            # Note: We can't access args directly here, so we use the global model_downgrade flag if it exists
            if 'model_downgrade' in globals() and model_downgrade:
                fallback_model = "gpt-3.5-turbo"
                logger.info(f"Using {fallback_model} for fallback due to model_downgrade flag")
            else:
                logger.info(f"Using {fallback_model} as fallback for low confidence categorization")
                
            response_4o = call_llm(
                prompt, 
                model=fallback_model,
                max_tokens=100, 
                temperature=0.2,
                context=f"{context_id}-4o"
            )
            content_4o = response_4o.choices[0].message.content.strip()
            
            # Parse result from the more powerful model
            result_4o = safe_json_parse(
                content_4o, 
                fallback={
                    "category": category,  # Keep original as fallback
                    "confidence": confidence,
                    "reasoning": reasoning
                },
                context=f"{context_id}-4o"
            )
            
            # Use the 4o result if it has higher confidence
            if result_4o["confidence"] > confidence:
                category = result_4o["category"]
                confidence = result_4o["confidence"]
                reasoning = result_4o["reasoning"]
                logger.info(f"Improved categorization with gpt-4o: {category} ({confidence})")
        
        # Final result tuple
        final_result = (category, confidence, reasoning)
        
        # Cache the result in both caches
        CATEGORIZE_CACHE[cache_key] = {
            'result': final_result,
            'timestamp': datetime.now().isoformat(),
            'transaction': {
                'Description': df_row['Description'],
                'Amount': float(df_row['Amount']),
                'Date': df_row['Date'],
                'Source': df_row['Source']
            },
            'hits': 1,
            'match_type': 'llm'  # Mark this as an LLM match to distinguish from rule matches
        }
        
        # Also add to simplified cache
        CATEGORIZE_CACHE[simple_cache_key] = {
            'result': final_result,
            'timestamp': datetime.now().isoformat(),
            'hits': 1,
            'match_type': 'llm'  # Mark this as an LLM match to distinguish from rule matches
        }
        
        # Periodically save cache to disk
        if len(CATEGORIZE_CACHE) % 5 == 0:
            save_cache(CATEGORIZE_CACHE, CATEGORIZE_CACHE_FILE)
            
        return category, confidence, reasoning
    except Exception as e:
        # Log the error with detailed information
        context_id = f"txn:[{df_row['Description'][:20]}...{df_row['Amount']}]"
        error_msg = f"Failed to categorize {context_id}: {str(e)}"
        logger.error(error_msg)
        logger.exception(f"Exception details for {context_id}:")
        
        # Print a shorter message to console
        print(f"Error categorizing transaction: {e}")
        
        # Use our fallback with error information
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
        
        # Also cache in the simplified cache
        CATEGORIZE_CACHE[simple_cache_key] = {
            'result': fallback_result,
            'timestamp': datetime.now().isoformat(),
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
    
    # First check if we have an exact cache match for this transaction
    # This is the most efficient path and avoids any rule checking or LLM calls
    cache_key = hashlib.md5(f"{df_row['Description']}{df_row['Amount']}{df_row['Date']}".encode()).hexdigest()
    if cache_key in CATEGORIZE_CACHE:
        cached_result = CATEGORIZE_CACHE[cache_key]
        if 'hits' in cached_result:
            cached_result['hits'] += 1
        else:
            cached_result['hits'] = 1
        
        # Extract the result based on cache format
        if 'result' in cached_result:
            if isinstance(cached_result['result'], tuple) and len(cached_result['result']) == 3:
                return cached_result['result']
            elif isinstance(cached_result['result'], dict) and 'category' in cached_result['result']:
                result_dict = cached_result['result']
                return (result_dict['category'], 
                        result_dict.get('confidence', 1.0), 
                        result_dict.get('reasoning', 'From cache'))
        
        # If we have a result field but in unknown format, log and continue to rules
        logger.debug(f"Cache format issue for {df_row['Description'][:20]}..., continuing to rule check")
    
    # Get the fuzzy threshold from global args if available
    fuzzy_threshold = 80  # Default
    if 'args' in globals() and hasattr(globals()['args'], 'fuzzy_threshold'):
        fuzzy_threshold = globals()['args'].fuzzy_threshold
    
    # Check custom and hardcoded rules
    for rule in CUSTOM_RULES:
        # Use fuzzy matching for both normalized and original description
        # Also check if amount matches (if specified) and day is in range
        try:
            transaction_day = pd.to_datetime(df_row["Date"]).day
            day_in_range = transaction_day in range(rule["day_range"][0], rule["day_range"][1] + 1)
        except:
            day_in_range = True  # If date parsing fails, ignore day range check
            
        if ((fuzzy_match(rule["description"], desc, fuzzy_threshold) or 
             fuzzy_match(rule["description"], original_desc, fuzzy_threshold)) and 
            (rule["amount"] is None or df_row["Amount"] == rule["amount"]) and 
            day_in_range):
            
            # Create the cache entries for this hit
            match_type = "fuzzy" if rule["description"] not in desc and rule["description"] not in original_desc else "exact"
            confidence = 0.95 if match_type == "exact" else 0.9  # Slightly lower confidence for fuzzy matches
            result = (rule["category"], confidence, f"Matched custom rule ({match_type} match)")
            
            # Log the match details for debugging
            logger.debug(f"Rule match: '{rule['description']}' -> '{desc}' ({match_type} match, {confidence:.2f} confidence)")
            
            # Add to cache
            CATEGORIZE_CACHE[cache_key] = {
                'result': result,
                'timestamp': datetime.now().isoformat(),
                'hits': 1,
                'rule_based': True,
                'match_type': match_type
            }
            
            return result
    
    # Check expanded hardcoded rules next      
    for category, keywords in CATEGORY_RULES.items():
        # First check for exact matches (faster)
        if any(keyword.lower() in desc for keyword in keywords) or any(keyword.lower() in original_desc for keyword in keywords):
            result = (category, 1.0, "Matched hardcoded rule (exact match)")
            
            # Add to cache
            CATEGORIZE_CACHE[cache_key] = {
                'result': result,
                'timestamp': datetime.now().isoformat(),
                'hits': 1,
                'rule_based': True,
                'match_type': 'exact'
            }
            
            # Save cache periodically
            if len(CATEGORIZE_CACHE) % 20 == 0:
                save_cache(CATEGORIZE_CACHE, CATEGORIZE_CACHE_FILE)
                
            return result
            
        # If no exact match, try fuzzy matching
        # This is separated from exact matching to prioritize exact matches and improve performance
        for keyword in keywords:
            if (fuzzy_match(keyword.lower(), desc, fuzzy_threshold) or 
                fuzzy_match(keyword.lower(), original_desc, fuzzy_threshold)):
                
                # Log the match for debugging
                logger.debug(f"Fuzzy match: '{keyword}' -> '{desc}' (category: {category})")
                
                result = (category, 0.9, "Matched hardcoded rule (fuzzy match)")
                
                # Add to cache
                CATEGORIZE_CACHE[cache_key] = {
                    'result': result,
                    'timestamp': datetime.now().isoformat(),
                    'hits': 1,
                    'rule_based': True,
                    'match_type': 'fuzzy'
                }
                
                # Save cache periodically
                if len(CATEGORIZE_CACHE) % 20 == 0:
                    save_cache(CATEGORIZE_CACHE, CATEGORIZE_CACHE_FILE)
                    
                return result
            
    # Fallback to enhanced LLM categorization if no rule match
    
    # Create example transactions for each category to improve context
    examples = {
        "Entertainment": ["netflix subscription $14.99", "movie theater tickets $24.00", "spotify premium $9.99"],
        "Food": ["starbucks coffee $5.95", "doordash delivery $35.50", "grocery store $125.75"],
        "Payments": ["bank transfer $500.00", "paypal payment $95.00", "bill payment $145.00"],
        "Shopping": ["amazon purchase $45.99", "walmart $87.50", "target $63.25"],
        "Travel": ["uber trip $22.50", "hotel booking $199.00", "airline ticket $350.00"],
        "Fees": ["atm fee $3.00", "overdraft fee $35.00", "credit card annual fee $95.00"],
        "Subscriptions": ["monthly service $8.99", "software subscription $15.00", "streaming service $12.99"],
        "Insurance": ["auto insurance premium $125.00", "health insurance $210.00", "renters insurance $15.00"],
        "Investments": ["stock purchase $500.00", "crypto purchase $100.00", "retirement contribution $250.00"],
        "Rent": ["apartment rent $1500.00", "mortgage payment $2200.00", "housing payment $1900.00"],
        "Income": ["salary deposit $3500.00", "client payment $750.00", "tax refund $1200.00"]
    }
    
    # If we have corrections data, use it to enhance examples
    if CORRECTIONS:
        # Get up to 2 examples per category from user corrections
        for correction in CORRECTIONS:
            category = correction.get("new_category")
            if category in examples and len(examples[category]) < 5:  # Limit to 5 examples per category
                txn = correction.get("transaction", {})
                desc_text = txn.get("Description", "")
                amount = txn.get("Amount", 0)
                if desc_text and amount:
                    examples[category].append(f"{desc_text} ${amount:.2f}")
    
    # Use the existing LLM function but pass our enhanced examples through the past_transactions param
    category, confidence, reasoning = categorize_with_gpt4_advanced(
        df_row, 
        {"hardcoded": CATEGORY_RULES, "custom": CUSTOM_RULES}, 
        {"examples": examples, "dataframe": past_transactions}
    )
    
    # Apply confidence threshold for automatic acceptance
    threshold = 0.8  # Only accept LLM suggestions with confidence >= 0.8
    if confidence < threshold:
        # Keep the original confidence score but mark as needing review
        original_category = category
        original_reasoning = reasoning
        category = "Uncategorized"
        reasoning = f"Low confidence ({confidence:.2f}) for '{original_category}'. Original reasoning: {original_reasoning}"
        logger.info(f"Low confidence categorization ({confidence:.2f}) for '{df_row['Description'][:30]}' - marked for review")
    elif category != "Uncategorized":
        # Only log high confidence suggestions to reduce console output
        logger.info(f"LLM suggested: {category} for '{desc}' (Confidence: {confidence})")
        
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
    # Detailed logging for CSV processing
    logger.info(f"Processing CSV file: {file_path}")
    
    try:
        if "boa" in source_name.lower() or "estmt" in source_name.lower():
            source = "BoA"
            logger.info(f"Identified as Bank of America CSV format")
            df = pd.read_csv(file_path, usecols=["Posted Date", "Payee", "Amount"])
            df.columns = ["Date", "Description", "Amount"]
        elif "sfcu" in source_name.lower() or "accounthistory" in source_name.lower():
            source = "SFCU"
            logger.info(f"Identified as SFCU CSV format")
            df = pd.read_csv(file_path)
            df["Amount"] = df["Debit"].fillna(0) - df["Credit"].fillna(0)
            df = df[["Post Date", "Description", "Amount"]]
            df.columns = ["Date", "Description", "Amount"]
        elif "citi" in source_name.lower():
            source = "Citi"
            logger.info(f"Identified as Citibank CSV format")
            df = pd.read_csv(file_path, usecols=["Posted Date", "Payee", "Amount"])
            df.columns = ["Date", "Description", "Amount"]
        elif "chase" in source_name.lower():
            source = "Chase"
            logger.info(f"Identified as Chase CSV format")
            df = pd.read_csv(file_path, usecols=["Transaction Date", "Description", "Amount"])
            df.columns = ["Date", "Description", "Amount"]
        elif "amex" in source_name.lower() or "american" in source_name.lower():
            source = "AMEX"
            logger.info(f"Identified as American Express CSV format")
            df = pd.read_csv(file_path, usecols=["Date", "Description", "Amount"])
        elif "discover" in source_name.lower():
            source = "Discover"
            logger.info(f"Identified as Discover CSV format")
            df = pd.read_csv(file_path, usecols=["Trans. Date", "Description", "Amount"])
            df.columns = ["Date", "Description", "Amount"]
        elif "capital" in source_name.lower() or "cap1" in source_name.lower():
            source = "Capital One"
            logger.info(f"Identified as Capital One CSV format")
            df = pd.read_csv(file_path, usecols=["Transaction Date", "Description", "Debit"])
            df.columns = ["Date", "Description", "Amount"]
        else:
            # Try to infer the format from column names if source is unknown
            source = "Unknown"
            logger.warning(f"Unknown CSV format for {file_path}, attempting to infer format")
            
            # Read the CSV file and check its columns
            try:
                df_temp = pd.read_csv(file_path)
                logger.info(f"CSV columns: {df_temp.columns.tolist()}")
                
                # Look for date, description and amount columns with various possible names
                date_cols = [col for col in df_temp.columns if 'date' in col.lower() or 'post' in col.lower()]
                desc_cols = [col for col in df_temp.columns if 'desc' in col.lower() or 'payee' in col.lower() or 'narration' in col.lower()]
                amount_cols = [col for col in df_temp.columns if 'amount' in col.lower() or 'debit' in col.lower() or 'credit' in col.lower()]
                
                if date_cols and desc_cols and (amount_cols or ('debit' in df_temp.columns.str.lower() and 'credit' in df_temp.columns.str.lower())):
                    logger.info(f"Inferred columns - Date: {date_cols[0]}, Description: {desc_cols[0]}, Amount: {amount_cols[0] if amount_cols else 'debit/credit'}")
                    
                    # Use the first matching columns
                    if amount_cols:
                        df = df_temp[[date_cols[0], desc_cols[0], amount_cols[0]]]
                        df.columns = ["Date", "Description", "Amount"]
                    else:
                        # Handle separate debit/credit columns
                        debit_col = [col for col in df_temp.columns if 'debit' in col.lower()][0]
                        credit_col = [col for col in df_temp.columns if 'credit' in col.lower()][0]
                        df = df_temp[[date_cols[0], desc_cols[0], debit_col, credit_col]]
                        # Combine debit and credit into a single amount column
                        df["Amount"] = df[credit_col].fillna(0) - df[debit_col].fillna(0)
                        df = df[[date_cols[0], desc_cols[0], "Amount"]]
                        df.columns = ["Date", "Description", "Amount"]
                else:
                    logger.error(f"Could not infer CSV format for {file_path}, skipping")
                    return all_data
            except Exception as e:
                logger.error(f"Error inferring CSV format: {e}")
                return all_data
    except Exception as e:
        logger.error(f"Error processing CSV file {file_path}: {e}")
        return all_data

    # Add source and file information
    df["Source"] = source
    df["File"] = file_path
    
    row_count = len(df)
    logger.info(f"Successfully processed {row_count} transactions from {source} CSV")
    
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
    
    # Core functionality options
    parser.add_argument("--review", action="store_true", help="Enable interactive review of low-confidence transactions")
    parser.add_argument("--skip-normalization", action="store_true", help="Skip LLM-based description normalization (faster)")
    parser.add_argument("--skip-llm-dedup", action="store_true", help="Skip LLM-based deduplication (faster)")
    parser.add_argument("--skip-rule-gen", action="store_true", help="Skip automatic rule generation")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode (skips all LLM features)")
    parser.add_argument("--fuzzy-threshold", type=int, default=80, help="Fuzzy matching threshold (0-100, default: 80)")
    
    # Display and logging options
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress information")
    parser.add_argument("--view-logs", action="store_true", help="View recent log entries")
    parser.add_argument("--log-lines", type=int, default=50, help="Number of log lines to view")
    parser.add_argument("--log-level", choices=["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        default="ALL", help="Filter logs by level")
    parser.add_argument("--log-contains", type=str, help="Filter logs by content")
    
    # Rule management options
    parser.add_argument("--suggest-rules", action="store_true", help="Get suggestions for custom payment rules")
    parser.add_argument("--analyze-uncategorized", action="store_true", 
                        help="Analyze uncategorized transactions and suggest rules")
    
    # API usage optimization options
    parser.add_argument("--model-downgrade", action="store_true", 
                       help="Use gpt-3.5-turbo for all tasks (except rule generation)")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Maximum batch size for batch processing (default: 50)")
    parser.add_argument("--api-stats", action="store_true",
                       help="Show detailed API usage statistics")
    
    args = parser.parse_args()
    
    # Handle the view-logs command
    if args.view_logs:
        view_logs(args.log_lines, args.log_level, args.log_contains)
        return
        
    # Handle the suggest-rules command
    if args.suggest_rules:
        suggest_custom_rules()
        return
        
    # Handle the analyze-uncategorized command
    if args.analyze_uncategorized:
        print("Loading existing transactions from deduplicated file...")
        if os.path.exists(DEDUPE_CSV):
            try:
                df = pd.read_csv(DEDUPE_CSV)
                uncategorized_count = sum(df["Category"] == "Uncategorized")
                
                if uncategorized_count > 0:
                    print(f"Found {uncategorized_count} uncategorized transactions")
                    should_recategorize = generate_rule_suggestions(df)
                    
                    if should_recategorize:
                        print("\nRe-categorizing transactions with new rules...")
                        # Reset categories for uncategorized transactions
                        mask = df["Category"] == "Uncategorized"
                        df.loc[mask, "Category"] = None
                        df.loc[mask, "Confidence"] = None
                        df.loc[mask, "Reasoning"] = None
                        
                        # Count transactions to recategorize
                        to_recategorize = sum(mask)
                        
                        # Re-apply categorization
                        for idx, row in df[mask].iterrows():
                            category, confidence, reasoning = categorize_transaction(row, df)
                            df.at[idx, "Category"] = category
                            df.at[idx, "Confidence"] = confidence
                            df.at[idx, "Reasoning"] = reasoning
                            
                            # Show progress for every 5th transaction
                            if idx % 5 == 0:
                                print(f"Processed {idx}/{to_recategorize} transactions...")
                                
                        # Save updated results
                        df.to_csv(DEDUPE_CSV, index=False)
                        print(f"Updated categories saved to {DEDUPE_CSV}")
                        
                        # Show categorization results
                        uncategorized_count = sum(df["Category"] == "Uncategorized")
                        print(f"\nAfter re-categorization: {uncategorized_count} uncategorized transactions remaining")
                else:
                    print("No uncategorized transactions found!")
            except Exception as e:
                print(f"Error loading or processing deduplicated file: {e}")
        else:
            print(f"No deduplicated file found at {DEDUPE_CSV}. Run the main program first.")
        return
    
    # If fast mode is enabled, set all skip flags
    if args.fast:
        args.skip_normalization = True
        args.skip_llm_dedup = True
        args.skip_rule_gen = True
        print("Running in fast mode - all LLM features disabled")
        
    # Model downgrade handling moved into main function instead of call_llm
    model_downgrade = args.model_downgrade if hasattr(args, 'model_downgrade') else False
    if model_downgrade:
        # Force GPT-3.5 for all tasks except rule generation, which needs the stronger model
        print("Using gpt-3.5-turbo for all tasks except rule generation (model downgrade enabled)")
        # We'll pass this as a parameter to functions that need it
        
    # Set the batch size for batch processing as a global variable
    global max_batch_size, fuzzy_threshold
    max_batch_size = args.batch_size
    fuzzy_threshold = args.fuzzy_threshold
    print(f"Maximum batch size set to {max_batch_size} transactions")
    print(f"Fuzzy matching enabled with threshold {fuzzy_threshold} - will catch similar transaction descriptions")
    
    # Reset API usage tracking for this run
    global API_USAGE
    API_USAGE = {
        "total_calls": 0,
        "gpt4_calls": 0,
        "gpt35_calls": 0,
        "normalization_calls": 0,
        "categorization_calls": 0,
        "rule_generation_calls": 0,
        "deduplication_calls": 0,
        "token_estimate": 0,
        "cost_estimate": 0.0
    }
    
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
        
        # Process in batches according to specified max_batch_size
        batch_size = max_batch_size
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
            
            # Process batch and update dataframe - use global max_batch_size
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
        resume_choice = input("Resume from existing progress? ").lower().strip()
        if any(yes_word in resume_choice for yes_word in ['y', 'yes', 'yeah', 'sure', 'ok', 'please', 'do it']):
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
        resume_dedupe = input("Skip deduplication and use existing file? ").lower().strip()
        if any(yes_word in resume_dedupe for yes_word in ['y', 'yes', 'yeah', 'sure', 'ok', 'skip', 'use existing']):
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
        
        # First analyze uncategorized transactions
        should_recategorize = generate_rule_suggestions(deduped_df)
        
        # Then analyze low confidence transactions
        print("Analyzing low confidence transactions for rule generation...")
        should_recategorize_low_conf = generate_low_confidence_rules(deduped_df)
        
        # Check if we have user corrections to analyze
        should_recategorize_corrections = False
        if CORRECTIONS:
            print("Analyzing user corrections for rule generation...")
            should_recategorize_corrections = generate_rules_from_corrections()
        
        # Combine results - recategorize if any function suggests it
        should_recategorize = should_recategorize or should_recategorize_low_conf or should_recategorize_corrections
        
        # Re-categorize if new rules were added and user wants to apply them
        if should_recategorize:
            print("\nRe-categorizing transactions with new rules...")
            # Make a copy of the dataframe for recategorization
            df_recategorize = deduped_df.copy()
            
            # Reset categories for uncategorized transactions
            mask = df_recategorize["Category"] == "Uncategorized"
            df_recategorize.loc[mask, "Category"] = None
            df_recategorize.loc[mask, "Confidence"] = None
            df_recategorize.loc[mask, "Reasoning"] = None
            
            # Count transactions to recategorize
            to_recategorize = sum(mask)
            print(f"Re-categorizing {to_recategorize} transactions...")
            
            # Re-apply categorization
            results = []
            for idx, row in df_recategorize[mask].iterrows():
                category, confidence, reasoning = categorize_transaction(row, df_recategorize)
                df_recategorize.at[idx, "Category"] = category
                df_recategorize.at[idx, "Confidence"] = confidence
                df_recategorize.at[idx, "Reasoning"] = reasoning
                
                # Show progress
                if len(results) % 10 == 0:
                    print(f"Processed {len(results)}/{to_recategorize} transactions...")
                    
                results.append((category, confidence, reasoning))
            
            # Update the main dataframe with re-categorized results
            deduped_df.update(df_recategorize)
            
            # Save updated results
            deduped_df.to_csv(DEDUPE_CSV, index=False)
            print(f"Updated categories saved to {DEDUPE_CSV}")
            
            # Show categorization results
            uncategorized_count = sum(deduped_df["Category"] == "Uncategorized")
            print(f"\nAfter re-categorization: {uncategorized_count} uncategorized transactions remaining")
            
            # Show category distribution
            categories = deduped_df["Category"].value_counts().to_dict()
            print("\nUpdated category distribution:")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                if pd.notna(cat):
                    print(f"  {cat}: {count}")
    else:
        print("Skipping automatic rule generation...")
    
    # Launch interactive review if requested
    if args.review:
        interactive_review(deduped_df)
        
    # Final analysis of uncategorized transactions
    uncategorized_count = sum(deduped_df["Category"] == "Uncategorized")
    if uncategorized_count > 0 and not args.review:
        print(f"\nThere are still {uncategorized_count} uncategorized transactions.")
        review_prompt = input("Would you like to review these now? ").lower().strip()
        if any(yes_word in review_prompt for yes_word in ['y', 'yes', 'yeah', 'sure', 'ok', 'please', 'do it']):
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
    
    # Log API usage statistics
    logger.info("API usage statistics:")
    logger.info(f"- Total API calls: {API_USAGE['total_calls']}")
    logger.info(f"- GPT-4 calls: {API_USAGE['gpt4_calls']}")
    logger.info(f"- GPT-3.5 calls: {API_USAGE['gpt35_calls']}")
    logger.info(f"- Normalization calls: {API_USAGE['normalization_calls']}")
    logger.info(f"- Categorization calls: {API_USAGE['categorization_calls']}")
    logger.info(f"- Rule generation calls: {API_USAGE['rule_generation_calls']}")
    logger.info(f"- Deduplication calls: {API_USAGE['deduplication_calls']}")
    logger.info(f"- Estimated tokens: {int(API_USAGE['token_estimate'])}")
    logger.info(f"- Estimated cost: ${API_USAGE['cost_estimate']:.4f}")
    
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
    
    # If we tracked match types, show the matching statistics
    exact_matches = sum(1 for key, cache_item in CATEGORIZE_CACHE.items() 
                      if cache_item.get('match_type') == 'exact')
    fuzzy_matches = sum(1 for key, cache_item in CATEGORIZE_CACHE.items() 
                      if cache_item.get('match_type') == 'fuzzy')
    llm_matches = sum(1 for key, cache_item in CATEGORIZE_CACHE.items() 
                     if cache_item.get('match_type') == 'llm')
    
    total_categorized = exact_matches + fuzzy_matches + llm_matches
    
    if total_categorized > 0:
        print("\nCategorization Method Statistics:")
        print(f"- Rule-based exact matches: {exact_matches} ({(exact_matches/total_categorized)*100:.1f}%)")
        print(f"- Rule-based fuzzy matches: {fuzzy_matches} ({(fuzzy_matches/total_categorized)*100:.1f}%) (threshold: {fuzzy_threshold})")
        print(f"- LLM-based categorization: {llm_matches} ({(llm_matches/total_categorized)*100:.1f}%)")
        
        if fuzzy_matches > 0 and (exact_matches + fuzzy_matches) > 0:
            rule_matches = exact_matches + fuzzy_matches
            fuzzy_percentage = (fuzzy_matches / rule_matches) * 100
            print(f"- {fuzzy_percentage:.1f}% of rule-based matches were fuzzy matches")
            
        # Show fuzzy matching cost savings
        if fuzzy_matches > 0:
            # Assuming average cost of $0.0001 per LLM categorization (very rough estimate)
            estimated_savings = fuzzy_matches * 0.0001
            print(f"- Estimated cost savings from fuzzy matching: ${estimated_savings:.4f}")
    
    # API usage statistics
    print("\nAPI Usage Statistics:")
    print(f"- Total API calls: {API_USAGE['total_calls']}")
    print(f"- GPT-4 calls: {API_USAGE['gpt4_calls']}")
    print(f"- GPT-3.5 calls: {API_USAGE['gpt35_calls']}")
    print(f"- By purpose:")
    print(f"  - Normalization: {API_USAGE['normalization_calls']}")
    print(f"  - Categorization: {API_USAGE['categorization_calls']}")
    print(f"  - Rule generation: {API_USAGE['rule_generation_calls']}")
    print(f"  - Deduplication: {API_USAGE['deduplication_calls']}")
    print(f"- Estimated tokens: {int(API_USAGE['token_estimate'])}")
    print(f"- Estimated cost: ${API_USAGE['cost_estimate']:.4f}")
    
    print("\n" + "="*50)
    print(f"Processing complete! Results saved to {DEDUPE_CSV}")
    print("="*50)

if __name__ == "__main__":
    main()