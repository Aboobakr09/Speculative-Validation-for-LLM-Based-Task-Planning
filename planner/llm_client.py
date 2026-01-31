"""
Groq LLM Client for planning.
Provides a simple wrapper with rate limiting for the Groq API.
"""

import os
import time
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class GroqClient:
    """
    Simple Groq API wrapper with rate limiting.
    
    Usage:
        client = GroqClient()
        response = client.generate("Say hello")
    """
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize Groq client.
        
        Args:
            model: Model to use. Current options include:
                   - "llama-3.3-70b-versatile" (default, best quality)
                   - "llama-3.1-8b-instant" (faster, lower quality)
                   - "mixtral-8x7b-32768" (good balance)
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        self.last_call_time = 0
        self.min_delay = 3.0  # Seconds between calls (20/min limit on free tier)
        self.total_calls = 0
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.2) -> str:
        """
        Generate text with rate limiting.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0, lower = more deterministic)
            
        Returns:
            Generated text string
        """
        # Rate limiting
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=False,
            stop=None
        )
        
        self.last_call_time = time.time()
        self.total_calls += 1
        
        return response.choices[0].message.content
    
    def get_stats(self) -> dict:
        """Return usage statistics."""
        return {
            "total_calls": self.total_calls,
            "model": self.model
        }


def get_llm_client(provider: str = "groq", **kwargs) -> GroqClient:
    """
    Factory function for creating LLM clients.
    
    Args:
        provider: LLM provider ("groq" is currently the only supported option)
        **kwargs: Additional arguments passed to client constructor
        
    Returns:
        LLM client instance
    """
    if provider == "groq":
        return GroqClient(**kwargs)
    raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Groq LLM Client...")
    print("=" * 40)
    
    try:
        llm = get_llm_client()
        response = llm.generate("Say hello in one word", max_tokens=10)
        print(f"Response: {response}")
        print(f"Stats: {llm.get_stats()}")
        print("✓ LLM Client working!")
    except Exception as e:
        print(f"✗ Error: {e}")
