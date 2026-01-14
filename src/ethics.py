"""
AI Analyst Agent - Ethical AI Module
Bias detection, guardrails, and content safety checks.
"""

import re
from typing import Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import brown

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown', quiet=True)


# ==============================================================================
# Gender Bias Detection
# ==============================================================================
class GenderBiasDetector:
    """Detect gender bias in text using word frequency analysis."""
    
    # Expanded gender-associated word lists
    MALE_WORDS = {
        "he", "him", "his", "himself", "man", "men", "male", "boy", "boys",
        "father", "son", "brother", "husband", "uncle", "nephew", "grandfather",
        "gentleman", "gentlemen", "sir", "mr", "king", "prince", "businessman"
    }
    
    # Re-defined to ensure valid encoding
    FEMALE_WORDS = {
        "she", "her", "hers", "herself", "woman", "women", "female", "girl", "girls",
        "mother", "daughter", "sister", "wife", "aunt", "niece", "grandmother",
        "lady", "ladies", "madam", "ms", "mrs", "queen", "princess", "businesswoman"
    }
    
    # Stereotype-associated words (simplified)
    MALE_STEREOTYPES = {
        "aggressive", "dominant", "competitive", "strong", "rational", "logical",
        "ambitious", "assertive", "independent", "leader"
    }
    
    FEMALE_STEREOTYPES = {
        "emotional", "nurturing", "gentle", "caring", "sensitive", "passive",
        "supportive", "dependent", "collaborative", "follower"
    }
    
    @classmethod
    def analyze(cls, text: str) -> dict:
        """
        Analyze text for gender bias.
        
        Returns:
            dict with bias_score (0-1), details, and recommendations
        """
        if not text or not text.strip():
            return {
                "bias_score": 0.0,
                "category": "neutral",
                "details": "No text provided",
                "male_terms": 0,
                "female_terms": 0,
                "recommendations": []
            }
        
        # Tokenize and lowercase (Use Regex for maximum robustness)
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # Count gendered words
        male_count = len(words & cls.MALE_WORDS)
        female_count = len(words & cls.FEMALE_WORDS)
        
        # Count stereotypes
        male_stereo = len(words & cls.MALE_STEREOTYPES)
        female_stereo = len(words & cls.FEMALE_STEREOTYPES)
        
        total_gendered = male_count + female_count
        total_stereo = male_stereo + female_stereo
        
        # Calculate bias scores
        # Logic: absolute difference / max(1, total_gendered)
        # This prevents 1.0 score for single-word occurrences unless extreme
        if total_gendered == 0:
            gender_bias = 0.0
        else:
            # New Formula: bias is ratio of imbalance
            # e.g., M=1, F=0 -> diff=1. bias = 1.0 (Still high, but correct for imbalance)
            # e.g., M=1, F=1 -> diff=0. bias = 0.0
            gender_bias = abs(male_count - female_count) / total_gendered
        
        if total_stereo == 0:
            stereo_bias = 0.0
        else:
            stereo_bias = total_stereo / len(words) if len(words) > 0 else 0.0
        
        # Combined bias score (weighted)
        # Cap gender bias contribution if total word count is low (handles boilerplate + short input)
        if len(words) < 20 and total_gendered == 1:
             gender_bias *= 0.5  # Reduce penalty for short texts with 1 gender term
             
        bias_score = 0.7 * gender_bias + 0.3 * min(stereo_bias * 10, 1.0)
        bias_score = min(round(bias_score, 4), 1.0)
        
        # Categorize
        if bias_score < 0.1:
            category = "minimal"
        elif bias_score < 0.3:
            category = "low"
        elif bias_score < 0.5:
            category = "moderate"
        else:
            category = "high"
        
        # Generate recommendations
        recommendations = []
        if male_count > female_count + 2:
            recommendations.append("Consider using more gender-neutral language")
        if female_count > male_count + 2:
            recommendations.append("Consider balancing gendered references")
        if total_stereo > 2:
            recommendations.append("Review stereotypical language usage")
        
        return {
            "bias_score": bias_score,
            "category": category,
            "details": f"Male terms: {male_count}, Female terms: {female_count}, Stereotypes: {total_stereo}",
            "male_terms": male_count,
            "female_terms": female_count,
            "stereotypes_found": total_stereo,
            "recommendations": recommendations
        }


# ==============================================================================
# Content Safety Guardrails
# ==============================================================================
class ContentGuardrails:
    """Safety checks and content filtering for queries and responses."""
    
    # Harmful query patterns
    HARMFUL_PATTERNS = [
        r"\b(hack|exploit|attack)\b.*\b(system|database|server)\b",
        r"\b(steal|leak)\b.*\b(data|information|credentials)\b",
        r"\b(bypass|circumvent)\b.*\b(security|authentication)\b",
        r"\b(inject|execute)\b.*\b(sql|code|script)\b",
        r"\bdrop\s+table\b",
        r"\bdelete\s+from\b.*\bwhere\b",
        r"\b(password|secret|key)\s*=",
    ]
    
    # PII patterns to redact
    PII_PATTERNS = [
        (r"\b[\w.-]+@[\w.-]+\.\w+\b", "[EMAIL]"),  # Email
        (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]"),  # Phone
        (r"\b\d{3}[-]?\d{2}[-]?\d{4}\b", "[SSN]"),  # SSN
        (r"\b\d{16}\b", "[CARD]"),  # Credit card
    ]
    
    # Blocked topics
    BLOCKED_TOPICS = [
        "personal information", "private data", "credit card",
        "social security", "password", "authentication bypass"
    ]
    
    @classmethod
    def check_query(cls, query: str) -> dict:
        """
        Check if a query is safe to process.
        
        Returns:
            dict with is_safe, reason, and modified_query
        """
        if not query:
            return {"is_safe": True, "reason": None, "modified_query": query}
        
        query_lower = query.lower()
        
        # Check harmful patterns
        for pattern in cls.HARMFUL_PATTERNS:
            if re.search(pattern, query_lower):
                return {
                    "is_safe": False,
                    "reason": "Query contains potentially harmful content",
                    "modified_query": None
                }
        
        # Check blocked topics
        for topic in cls.BLOCKED_TOPICS:
            if topic in query_lower:
                return {
                    "is_safe": False,
                    "reason": f"Query contains blocked topic: {topic}",
                    "modified_query": None
                }
        
        return {"is_safe": True, "reason": None, "modified_query": query}
    
    @classmethod
    def redact_pii(cls, text: str) -> str:
        """Redact personally identifiable information from text."""
        if not text:
            return text
        
        result = text
        for pattern, replacement in cls.PII_PATTERNS:
            result = re.sub(pattern, replacement, result)
        
        return result
    
    @classmethod
    def sanitize_sql(cls, query: str) -> Optional[str]:
        """
        Sanitize SQL query to prevent injection.
        
        Returns:
            Sanitized query or None if query is unsafe
        """
        if not query:
            return None
        
        # Forbidden operations
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", 
                    "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE"]
        
        query_upper = query.upper()
        for op in forbidden:
            if op in query_upper:
                return None
        
        # Check for comment injection
        if "--" in query or "/*" in query:
            return None
        
        # Check for stacked queries
        if ";" in query:
            # Only allow one statement
            parts = [p.strip() for p in query.split(";") if p.strip()]
            if len(parts) > 1:
                return None
        
        return query


# ==============================================================================
# Ethical AI Report Generator
# ==============================================================================
class EthicalAIReport:
    """Generate ethical AI assessment reports."""
    
    @staticmethod
    def generate_report(text: str, query: str = None) -> dict:
        """
        Generate a comprehensive ethical AI report.
        
        Returns:
            dict with bias analysis, safety checks, and overall score
        """
        # Bias analysis
        bias_result = GenderBiasDetector.analyze(text)
        
        # Query safety
        query_safety = ContentGuardrails.check_query(query) if query else {"is_safe": True}
        
        # Calculate overall ethical score (higher is better)
        bias_penalty = bias_result["bias_score"] * 0.5
        safety_penalty = 0 if query_safety["is_safe"] else 0.5
        
        ethical_score = max(0, 1.0 - bias_penalty - safety_penalty)
        
        return {
            "ethical_score": round(ethical_score, 4),
            "bias_analysis": bias_result,
            "query_safety": query_safety,
            "passed": ethical_score >= 0.5,
            "summary": f"Ethical score: {ethical_score:.2%} | Bias: {bias_result['category']} | Safe: {query_safety['is_safe']}"
        }


# ==============================================================================
# Convenience Functions
# ==============================================================================
def check_bias(text: str) -> dict:
    """Quick bias check for text."""
    return GenderBiasDetector.analyze(text)


def is_safe_query(query: str) -> bool:
    """Quick safety check for query."""
    return ContentGuardrails.check_query(query)["is_safe"]


def redact_sensitive(text: str) -> str:
    """Redact sensitive information from text."""
    return ContentGuardrails.redact_pii(text)


def get_ethical_report(text: str, query: str = None) -> dict:
    """Get full ethical AI report."""
    return EthicalAIReport.generate_report(text, query)


# ==============================================================================
# Main (for testing)
# ==============================================================================
if __name__ == "__main__":
    print("Testing Ethical AI Module...")
    
    # Test bias detection
    test_texts = [
        "The chairman made a decision.",
        "She is a great nurse and he is a strong leader.",
        "The data shows revenue trends by region.",
    ]
    
    print("\n=== Bias Detection Tests ===")
    for text in test_texts:
        result = check_bias(text)
        print(f"\nText: '{text}'")
        print(f"  Bias Score: {result['bias_score']}")
        print(f"  Category: {result['category']}")
        print(f"  Details: {result['details']}")
    
    # Test guardrails
    test_queries = [
        "Top 5 locations by fare",
        "DROP TABLE trips",
        "Show me customer passwords",
        "Churn rate by region",
    ]
    
    print("\n=== Query Safety Tests ===")
    for query in test_queries:
        result = ContentGuardrails.check_query(query)
        print(f"\nQuery: '{query}'")
        print(f"  Safe: {result['is_safe']}")
        print(f"  Reason: {result['reason']}")
    
    # Test PII redaction
    print("\n=== PII Redaction Test ===")
    text_with_pii = "Contact john@email.com or call 555-123-4567"
    print(f"Original: {text_with_pii}")
    print(f"Redacted: {redact_sensitive(text_with_pii)}")
    
    print("\nAll tests completed!")
