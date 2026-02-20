"""
Reward computation for RL training on math word problems.

This module implements the RewardWorker class that computes scalar rewards
based on answer correctness and format compliance for GSM8K-style math problems.
"""

import re
from typing import Optional


class RewardWorker:
    """
    Computes scalar rewards based on answer correctness and format compliance.
    Runs on CPU nodes, horizontally scalable.
    
    Reward = correctness_score + format_score
    - correctness_score: 1.0 if final answer equals ground truth, 0.0 otherwise
    - format_score: 0.0 to 0.2 based on format compliance
    """
    
    # Patterns for extracting numeric answers
    ANSWER_PATTERNS = [
        r"(?:the answer is|answer:|final answer:?)\s*\$?(-?[\d,]+\.?\d*)\$?",
        r"####\s*(-?[\d,]+\.?\d*)",
        r"=\s*\$?(-?[\d,]+\.?\d*)\$?\s*$",
        r"(?:^|\s)(-?[\d,]+\.?\d*)\s*$",
    ]
    
    # Format compliance criteria
    FORMAT_CRITERIA = {
        "has_reasoning": 0.1,      # Contains step-by-step reasoning
        "has_final_answer": 0.1,   # Has clearly marked final answer
    }
    
    def extract_answer(self, completion: str) -> Optional[str]:
        """
        Extract final numeric answer from completion.
        
        Tries multiple patterns to find the answer:
        1. "the answer is X" or "answer: X"
        2. "#### X" (GSM8K format)
        3. "= X" at end of line
        4. Last number in the completion
        
        Args:
            completion: Model-generated completion text
            
        Returns:
            Extracted numeric answer as string, or None if not found
        """
        if not completion or not completion.strip():
            return None
        
        completion = completion.strip()
        
        for pattern in self.ANSWER_PATTERNS:
            match = re.search(pattern, completion, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1)
                # Remove commas from numbers like "1,000"
                answer = answer.replace(",", "")
                return answer
        
        return None
    
    def _normalize_answer(self, answer: str) -> Optional[float]:
        """
        Normalize answer string to float for comparison.
        
        Args:
            answer: Answer string to normalize
            
        Returns:
            Float value or None if parsing fails
        """
        if answer is None:
            return None
        
        try:
            # Remove commas and whitespace
            cleaned = answer.replace(",", "").strip()
            return float(cleaned)
        except (ValueError, TypeError):
            return None
    
    def check_format(self, completion: str) -> float:
        """
        Check if completion follows expected format.
        
        Evaluates:
        - Has step-by-step reasoning (0.1 points)
        - Has clearly marked final answer (0.1 points)
        
        Args:
            completion: Model-generated completion text
            
        Returns:
            Format score between 0.0 and 0.2
        """
        if not completion or not completion.strip():
            return 0.0
        
        score = 0.0
        
        # Check for reasoning indicators
        reasoning_indicators = [
            r"step\s*\d",
            r"first[,\s]",
            r"then[,\s]",
            r"next[,\s]",
            r"finally[,\s]",
            r"therefore",
            r"so[,\s]",
            r"because",
            r"let's",
            r"we need to",
            r"we have",
        ]
        
        has_reasoning = any(
            re.search(pattern, completion, re.IGNORECASE)
            for pattern in reasoning_indicators
        )
        
        if has_reasoning:
            score += self.FORMAT_CRITERIA["has_reasoning"]
        
        # Check for final answer markers
        answer_markers = [
            r"(?:the )?answer is",
            r"final answer",
            r"####",
            r"answer:",
        ]
        
        has_final_answer = any(
            re.search(pattern, completion, re.IGNORECASE)
            for pattern in answer_markers
        )
        
        if has_final_answer:
            score += self.FORMAT_CRITERIA["has_final_answer"]
        
        return score
    
    def compute_reward(
        self,
        prompt: str,
        completion: str,
        ground_truth: str,
    ) -> float:
        """
        Compute reward for a single trajectory.
        
        Binary reward: 1.0 if final answer matches ground truth, 0.0 otherwise.
        Clean binary signal gives GRPO advantages a clear correct/incorrect split.
        
        Args:
            prompt: The math problem prompt (unused but kept for interface consistency)
            completion: Model-generated completion text
            ground_truth: Expected numeric answer
            
        Returns:
            1.0 if correct, 0.0 otherwise
        """
        extracted = self.extract_answer(completion)
        extracted_value = self._normalize_answer(extracted)
        ground_truth_value = self._normalize_answer(ground_truth)
        
        if extracted_value is not None and ground_truth_value is not None:
            return 1.0 if abs(extracted_value - ground_truth_value) < 1e-6 else 0.0
        
        return 0.0
