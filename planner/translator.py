"""
Translator: Natural Language → Structured Actions
Huang et al. style semantic matching with LLM fallback
"""

import difflib
from typing import Optional, Tuple, List
from simulator.action_space import VALID_ACTIONS, ROOMS, OBJECTS


class ActionTranslator:
    """
    Two-stage translator: cheap difflib first, expensive LLM only when needed.
    
    Usage:
        translator = ActionTranslator()
        action, confidence, method = translator.translate("grab the cup")
        # Returns: ("pickup cup", 1.0, "exact")
    """
    
    def __init__(self, llm_client=None, confidence_threshold: float = 0.6):
        """
        Initialize translator.
        
        Args:
            llm_client: Optional LLM client for fallback disambiguation
            confidence_threshold: Min confidence for difflib match (0.0-1.0)
        """
        self.llm = llm_client
        self.threshold = confidence_threshold
        
        # Expanded synonym mapping for verbs
        self.verb_synonyms = {
            "pickup": ["pick", "grab", "take", "get", "hold", "lift", "carry", "fetch"],
            "goto": ["go", "move", "walk", "head", "navigate", "enter", "visit"],
            "drop": ["put", "place", "set", "leave", "release", "deposit", "down"],
            "toggle": ["turn", "switch", "flip", "activate", "deactivate", "on", "off"],
            "use": ["apply", "utilize", "operate", "wash", "clean", "make", "brush", "scrub"]
        }
        
        # Object synonyms (common alternate names)
        self.object_synonyms = {
            "cup": ["mug", "glass", "drink"],
            "coffee_maker": ["coffeemaker", "espresso", "brewer"],
            "light": ["lights", "bulb"],
            "lamp": ["bedlamp", "nightlight"],
            "faucet": ["tap", "sink", "water"],
            "remote": ["controller", "control"],
            "toothbrush": ["teeth", "tooth"],  # "brush teeth" → use toothbrush
        }
        
        # Flatten verb synonyms for reverse lookup
        self.nl_to_action = {}
        for action, synonyms in self.verb_synonyms.items():
            for syn in synonyms:
                self.nl_to_action[syn] = action
        
        # Flatten object synonyms for reverse lookup
        self.nl_to_object = {}
        for obj, synonyms in self.object_synonyms.items():
            for syn in synonyms:
                self.nl_to_object[syn] = obj
    
    def translate(self, nl_step: str) -> Tuple[Optional[str], float, str]:
        """
        Translate natural language to action string.
        
        Args:
            nl_step: Natural language instruction like "grab the cup"
        
        Returns:
            Tuple of (action_string, confidence, method)
            - action_string: e.g., "pickup cup" or None if failed
            - confidence: 0.0-1.0 score
            - method: "exact", "difflib", "llm_fallback", or "failed"
        """
        nl_step = nl_step.strip().lower()
        
        # Remove common filler words
        nl_step = self._clean_input(nl_step)
        
        words = nl_step.split()
        if not words:
            return None, 0.0, "failed"
        
        first_word = words[0]
        
        # Stage 0: Exact match in our synonym dict
        if first_word in self.nl_to_action:
            verb = self.nl_to_action[first_word]
            arg = self._extract_argument(words[1:], verb)
            if arg:
                return f"{verb} {arg}", 1.0, "exact"
        
        # Also check if first word IS a valid action
        if first_word in VALID_ACTIONS:
            arg = self._extract_argument(words[1:], first_word)
            if arg:
                return f"{first_word} {arg}", 1.0, "exact"
        
        # Stage 1: Difflib matching for verb
        verb, verb_confidence = self._match_verb_difflib(first_word)
        if verb and verb_confidence >= self.threshold:
            arg = self._extract_argument(words[1:], verb)
            if arg:
                return f"{verb} {arg}", verb_confidence, "difflib"
        
        # Stage 1b: Try matching verb from any position (for "turn on the faucet")
        for i, word in enumerate(words):
            verb, conf = self._match_verb_difflib(word)
            if verb and conf >= self.threshold:
                # Collect remaining words as potential argument
                other_words = words[:i] + words[i+1:]
                arg = self._extract_argument(other_words, verb)
                if arg:
                    return f"{verb} {arg}", conf, "difflib"
        
        # Stage 2: LLM fallback (if available)
        if self.llm:
            return self._llm_disambiguate(nl_step)
        
        return None, 0.0, "failed"
    
    def _clean_input(self, text: str) -> str:
        """Remove common filler words."""
        fillers = ["the", "a", "an", "to", "with", "on", "in", "at", "from", "please", "now"]
        words = text.split()
        cleaned = [w for w in words if w not in fillers]
        return " ".join(cleaned)
    
    def _match_verb_difflib(self, word: str) -> Tuple[Optional[str], float]:
        """Match verb using difflib sequence matching."""
        # Check against all valid actions + synonyms
        all_verbs = list(VALID_ACTIONS) + list(self.nl_to_action.keys())
        matches = difflib.get_close_matches(word, all_verbs, n=1, cutoff=0.0)
        
        if not matches:
            return None, 0.0
        
        matched = matches[0]
        similarity = difflib.SequenceMatcher(None, word, matched).ratio()
        
        # Map back to canonical action
        if matched in VALID_ACTIONS:
            return matched, similarity
        elif matched in self.nl_to_action:
            return self.nl_to_action[matched], similarity
        
        return None, 0.0
    
    def _extract_argument(self, words: List[str], verb: str) -> Optional[str]:
        """Extract room/object argument from remaining words."""
        if not words:
            return None
        
        # Determine valid targets based on verb
        if verb == "goto":
            valid_targets = ROOMS
        else:
            valid_targets = OBJECTS
        
        # Try joining all remaining words (handle "living room" → "living_room")
        arg_str = "_".join(words).strip("_")
        
        # Exact match
        if arg_str in valid_targets:
            return arg_str
        
        # Check object synonyms
        if arg_str in self.nl_to_object:
            return self.nl_to_object[arg_str]
        
        # Try each word individually
        for word in words:
            if word in valid_targets:
                return word
            if word in self.nl_to_object:
                return self.nl_to_object[word]
        
        # Difflib match for argument (fuzzy matching)
        all_targets = list(valid_targets)
        if verb != "goto":
            all_targets += list(self.nl_to_object.keys())
        
        matches = difflib.get_close_matches(arg_str, all_targets, n=1, cutoff=0.6)
        if matches:
            matched = matches[0]
            if matched in self.nl_to_object:
                return self.nl_to_object[matched]
            return matched
        
        # Try individual words with difflib
        for word in words:
            matches = difflib.get_close_matches(word, all_targets, n=1, cutoff=0.6)
            if matches:
                matched = matches[0]
                if matched in self.nl_to_object:
                    return self.nl_to_object[matched]
                return matched
        
        return None
    
    def _llm_disambiguate(self, nl_step: str) -> Tuple[Optional[str], float, str]:
        """Use LLM to translate when difflib is uncertain."""
        prompt = f"""Map this instruction to exactly one action from the list.

Valid actions: {', '.join(VALID_ACTIONS)}
Valid rooms: {', '.join(ROOMS)}
Valid objects: {', '.join(OBJECTS[:10])}...

Instruction: "{nl_step}"

Output format: action target (e.g., "pickup cup" or "goto kitchen")
Output only the action, nothing else."""
        
        try:
            response = self.llm.generate(prompt, max_tokens=20)
            parts = response.strip().lower().split()
            if len(parts) >= 2:
                action = parts[0]
                target = "_".join(parts[1:])
                if action in VALID_ACTIONS:
                    return f"{action} {target}", 0.9, "llm_fallback"
        except Exception as e:
            pass
        
        return None, 0.0, "failed"
    
    def batch_translate(self, steps: List[str]) -> List[Tuple[Optional[str], float, str]]:
        """Translate multiple steps at once."""
        return [self.translate(step) for step in steps]


# =============================================================================
# Testing Suite
# =============================================================================

def test_translator():
    """Test with 30+ diverse phrasings."""
    translator = ActionTranslator()
    
    test_cases = [
        # Exact matches
        ("pickup cup", "pickup cup"),
        ("goto kitchen", "goto kitchen"),
        ("drop plate", "drop plate"),
        ("toggle faucet", "toggle faucet"),
        ("use soap", "use soap"),
        
        # Synonym verbs
        ("grab the cup", "pickup cup"),
        ("take the cup", "pickup cup"),
        ("get cup", "pickup cup"),
        ("lift cup", "pickup cup"),
        ("fetch cup", "pickup cup"),
        
        # Navigation synonyms
        ("go to the kitchen", "goto kitchen"),
        ("walk to kitchen", "goto kitchen"),
        ("enter bedroom", "goto bedroom"),
        ("visit bathroom", "goto bathroom"),
        ("move to living room", "goto living_room"),
        
        # Drop synonyms
        ("put down the cup", "drop cup"),
        ("place cup", "drop cup"),
        ("set plate down", "drop plate"),
        ("leave towel", "drop towel"),
        
        # Toggle synonyms
        ("turn on faucet", "toggle faucet"),
        ("switch on the light", "toggle light"),
        ("flip lamp", "toggle lamp"),
        ("turn off light", "toggle light"),
        
        # Use synonyms
        ("wash with soap", "use soap"),
        ("clean with towel", "use towel"),
        ("operate coffee maker", "use coffee_maker"),
        ("brush teeth", "use toothbrush"),
        
        # Object synonyms
        ("grab the mug", "pickup cup"),
        ("turn on tap", "toggle faucet"),
        
        # Edge cases
        ("go to the living room", "goto living_room"),
        ("pick up cup", "pickup cup"),
    ]
    
    print("=" * 60)
    print(f"Testing ActionTranslator with {len(test_cases)} cases...")
    print("=" * 60)
    
    passed = 0
    failed = []
    
    for nl, expected in test_cases:
        result, conf, method = translator.translate(nl)
        success = result == expected
        status = "✓" if success else "✗"
        print(f"  {status} '{nl}' → {result} (conf: {conf:.2f}, {method})")
        
        if success:
            passed += 1
        else:
            failed.append((nl, expected, result))
    
    print("=" * 60)
    print(f"Passed: {passed}/{len(test_cases)} ({100*passed/len(test_cases):.1f}%)")
    
    if failed:
        print("\nFailed cases:")
        for nl, exp, got in failed:
            print(f"  Expected: '{exp}', Got: '{got}'")
    
    return passed, len(test_cases)


if __name__ == "__main__":
    test_translator()
