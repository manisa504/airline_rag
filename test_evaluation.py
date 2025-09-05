#!/usr/bin/env python3
"""
Quick evaluation test script
Run this to test the LLM-as-a-Judge evaluation system
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from llm_judge_evaluator import RAGEvaluator

def main():
    """Run a quick evaluation test."""
    print("🚁 Testing Aviation RAG Evaluation System")
    print("=" * 50)
    
    # Check if API key is available
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("Please set your API key in the .env file")
        return
    
    try:
        print("🔄 Initializing evaluator...")
        evaluator = RAGEvaluator()
        
        print("📊 Running evaluation on 3 test cases...")
        results = evaluator.run_evaluation(num_cases=3)
        
        print("\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved to evaluation_results_*.json")
        print(f"🎯 Average Overall Score: {results['aggregate_metrics']['avg_overall_score']:.3f}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your GOOGLE_API_KEY in .env file")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Ensure simple_rag.py is working correctly")

if __name__ == "__main__":
    main()
