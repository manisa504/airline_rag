"""
LLM-as-a-Judge Evaluation System for Aviation RAG
Provides comprehensive metrics including accuracy, ROUGE, BLEU, and custom aviation-specific evaluations.
"""

import json
import os
import time
from typing import Dict, List, Tuple, Any
import google.generativeai as genai
from dataclasses import dataclass
from pathlib import Path
import logging
from simple_rag import SimpleAviationRAG
import numpy as np
from datetime import datetime

# For ROUGE and BLEU metrics
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import word_tokenize
except ImportError:
    print("Installing evaluation dependencies...")
    import subprocess
    subprocess.run(["pip", "install", "rouge-score", "nltk"], check=True)
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import word_tokenize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    question: str
    predicted_answer: str
    reference_answer: str
    sources_used: List[Dict]
    accuracy_score: float
    relevance_score: float
    completeness_score: float
    rouge_1_f1: float
    rouge_2_f1: float
    rouge_l_f1: float
    bleu_score: float
    source_quality_score: float
    response_time: float
    overall_score: float

class LLMJudge:
    """LLM-as-a-Judge evaluation system for aviation RAG."""
    
    def __init__(self):
        """Initialize the LLM Judge with Gemini."""
        self.genai = genai
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
        
    def evaluate_accuracy(self, question: str, predicted: str, reference: str) -> float:
        """Evaluate factual accuracy using LLM judge."""
        prompt = f"""
        You are an expert aviation evaluator. Compare the predicted answer with the reference answer for factual accuracy.
        
        Question: {question}
        
        Reference Answer: {reference}
        
        Predicted Answer: {predicted}
        
        Rate the factual accuracy of the predicted answer on a scale of 0.0 to 1.0, where:
        - 1.0 = Completely accurate, all facts correct
        - 0.8 = Mostly accurate, minor factual errors
        - 0.6 = Partially accurate, some significant errors
        - 0.4 = Limited accuracy, many errors
        - 0.2 = Poor accuracy, mostly incorrect
        - 0.0 = Completely inaccurate
        
        Consider:
        - Factual correctness of aviation concepts
        - Technical accuracy of procedures and regulations
        - Consistency with aviation standards
        
        Respond with only a single number between 0.0 and 1.0.
        """
        
        try:
            response = self.model.generate_content(prompt)
            score = float(response.text.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Error evaluating accuracy: {e}")
            return 0.5
    
    def evaluate_relevance(self, question: str, answer: str) -> float:
        """Evaluate how well the answer addresses the question."""
        prompt = f"""
        You are an expert evaluator. Rate how well this answer addresses the specific question asked.
        
        Question: {question}
        
        Answer: {answer}
        
        Rate the relevance on a scale of 0.0 to 1.0, where:
        - 1.0 = Perfectly addresses all aspects of the question
        - 0.8 = Addresses most aspects well
        - 0.6 = Addresses some aspects, misses others
        - 0.4 = Partially relevant, goes off-topic
        - 0.2 = Minimally relevant
        - 0.0 = Completely irrelevant
        
        Respond with only a single number between 0.0 and 1.0.
        """
        
        try:
            response = self.model.generate_content(prompt)
            score = float(response.text.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Error evaluating relevance: {e}")
            return 0.5
    
    def evaluate_completeness(self, question: str, answer: str) -> float:
        """Evaluate how complete and comprehensive the answer is."""
        prompt = f"""
        You are an expert evaluator. Rate how complete and comprehensive this answer is for the aviation question.
        
        Question: {question}
        
        Answer: {answer}
        
        Rate the completeness on a scale of 0.0 to 1.0, where:
        - 1.0 = Comprehensive, covers all important aspects
        - 0.8 = Well-rounded, covers most important points
        - 0.6 = Adequate coverage, some gaps
        - 0.4 = Basic coverage, significant gaps
        - 0.2 = Minimal coverage, many missing elements
        - 0.0 = Incomplete or superficial
        
        Consider whether the answer provides sufficient detail for understanding the aviation concept.
        
        Respond with only a single number between 0.0 and 1.0.
        """
        
        try:
            response = self.model.generate_content(prompt)
            score = float(response.text.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Error evaluating completeness: {e}")
            return 0.5
    
    def calculate_rouge_scores(self, predicted: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        try:
            scores = self.rouge_scorer.score(reference, predicted)
            return {
                'rouge_1_f1': scores['rouge1'].fmeasure,
                'rouge_2_f1': scores['rouge2'].fmeasure,
                'rouge_l_f1': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"Error calculating ROUGE scores: {e}")
            return {'rouge_1_f1': 0.0, 'rouge_2_f1': 0.0, 'rouge_l_f1': 0.0}
    
    def calculate_bleu_score(self, predicted: str, reference: str) -> float:
        """Calculate BLEU score."""
        try:
            reference_tokens = word_tokenize(reference.lower())
            predicted_tokens = word_tokenize(predicted.lower())
            
            # BLEU expects list of reference sentences
            bleu_score = sentence_bleu(
                [reference_tokens], 
                predicted_tokens,
                smoothing_function=self.smoothing_function
            )
            return bleu_score
        except Exception as e:
            logger.warning(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def evaluate_source_quality(self, sources: List[Dict]) -> float:
        """Evaluate the quality and relevance of sources used."""
        if not sources:
            return 0.0
        
        # Calculate average similarity score
        scores = [s.get('similarity_score', 0.0) for s in sources]
        avg_similarity = np.mean(scores)
        
        # Check source diversity
        unique_sources = len(set(s.get('source', '') for s in sources))
        diversity_bonus = min(unique_sources / len(sources), 1.0) * 0.1
        
        # Penalize very low similarity scores
        quality_penalty = sum(1 for score in scores if score < 0.3) * 0.1
        
        return max(0.0, min(1.0, avg_similarity + diversity_bonus - quality_penalty))
    
    def evaluate_single_response(
        self, 
        question: str, 
        predicted_answer: str, 
        reference_answer: str,
        sources_used: List[Dict],
        response_time: float
    ) -> EvaluationResult:
        """Evaluate a single RAG response comprehensively."""
        
        logger.info(f"Evaluating: {question[:50]}...")
        
        # LLM-based evaluations
        accuracy_score = self.evaluate_accuracy(question, predicted_answer, reference_answer)
        relevance_score = self.evaluate_relevance(question, predicted_answer)
        completeness_score = self.evaluate_completeness(question, predicted_answer)
        
        # Lexical metrics
        rouge_scores = self.calculate_rouge_scores(predicted_answer, reference_answer)
        bleu_score = self.calculate_bleu_score(predicted_answer, reference_answer)
        
        # Source quality
        source_quality_score = self.evaluate_source_quality(sources_used)
        
        # Overall score (weighted combination)
        overall_score = (
            accuracy_score * 0.3 +
            relevance_score * 0.25 +
            completeness_score * 0.25 +
            rouge_scores['rouge_l_f1'] * 0.1 +
            bleu_score * 0.05 +
            source_quality_score * 0.05
        )
        
        return EvaluationResult(
            question=question,
            predicted_answer=predicted_answer,
            reference_answer=reference_answer,
            sources_used=sources_used,
            accuracy_score=accuracy_score,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            rouge_1_f1=rouge_scores['rouge_1_f1'],
            rouge_2_f1=rouge_scores['rouge_2_f1'],
            rouge_l_f1=rouge_scores['rouge_l_f1'],
            bleu_score=bleu_score,
            source_quality_score=source_quality_score,
            response_time=response_time,
            overall_score=overall_score
        )

class RAGEvaluator:
    """Complete evaluation system for the Aviation RAG."""
    
    def __init__(self):
        """Initialize the RAG evaluator."""
        self.rag_system = SimpleAviationRAG()
        self.judge = LLMJudge()
        self.test_cases = self.load_test_cases()
    
    def load_test_cases(self) -> List[Dict]:
        """Load or create test cases for evaluation."""
        test_cases_file = "evaluation_test_cases.json"
        
        if Path(test_cases_file).exists():
            with open(test_cases_file, 'r') as f:
                return json.load(f)
        
        # Default test cases if file doesn't exist
        default_cases = [
            {
                "question": "What is ETOPS and why is it important for twin-engine aircraft?",
                "reference_answer": "ETOPS (Extended Twin-engine Operational Performance Standards) is a certification that allows twin-engine aircraft to fly routes that are more than 60 minutes flying time from the nearest airport. It's important because it enables airlines to use more fuel-efficient twin-engine aircraft on long-haul routes while maintaining safety standards through rigorous aircraft and crew requirements."
            },
            {
                "question": "What is the role of a flight dispatcher?",
                "reference_answer": "A flight dispatcher is responsible for flight planning, monitoring weather conditions, calculating fuel requirements, determining optimal routes, and maintaining communication with pilots during flight. They share legal responsibility with the captain for the safe operation of the flight and can delay, cancel, or redirect flights if necessary."
            },
            {
                "question": "What is MEL and when is it used?",
                "reference_answer": "MEL (Minimum Equipment List) is a document that specifies which aircraft systems can be inoperative while still allowing safe flight operations. It's used when certain non-critical equipment fails, allowing the aircraft to continue operations under specific conditions and time limitations until the equipment can be repaired."
            },
            {
                "question": "How does Airport Collaborative Decision Making (A-CDM) work?",
                "reference_answer": "A-CDM is a process that improves airport efficiency by enabling real-time information sharing between all airport stakeholders including airlines, ground handlers, air traffic control, and airport operators. It helps optimize resource allocation, reduce delays, and improve predictability of airport operations through better coordination and data sharing."
            },
            {
                "question": "What factors affect aircraft fuel economy?",
                "reference_answer": "Aircraft fuel economy is affected by aerodynamics, aircraft weight, engine efficiency, altitude, airspeed, weather conditions, flight planning, and operational factors like seating configuration and passenger load factor. Optimal cruise altitude and speed, reduced weight, and efficient route planning are key to improving fuel economy."
            }
        ]
        
        # Save default cases
        with open(test_cases_file, 'w') as f:
            json.dump(default_cases, f, indent=2)
        
        return default_cases
    
    def run_evaluation(self, num_cases: int = None) -> Dict[str, Any]:
        """Run comprehensive evaluation on test cases."""
        test_cases = self.test_cases[:num_cases] if num_cases else self.test_cases
        results = []
        
        logger.info(f"Starting evaluation on {len(test_cases)} test cases...")
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Evaluating case {i}/{len(test_cases)}")
            
            question = test_case["question"]
            reference_answer = test_case["reference_answer"]
            
            # Get RAG response with timing
            start_time = time.time()
            rag_result = self.rag_system.query(question)
            response_time = time.time() - start_time
            
            # Evaluate the response
            evaluation = self.judge.evaluate_single_response(
                question=question,
                predicted_answer=rag_result['answer'],
                reference_answer=reference_answer,
                sources_used=rag_result['sources'],
                response_time=response_time
            )
            
            results.append(evaluation)
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(results)
        
        # Save detailed results
        self.save_results(results, aggregate_metrics)
        
        return {
            'individual_results': results,
            'aggregate_metrics': aggregate_metrics,
            'total_cases': len(results)
        }
    
    def calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate aggregate metrics across all results."""
        if not results:
            return {}
        
        metrics = {
            'avg_accuracy': np.mean([r.accuracy_score for r in results]),
            'avg_relevance': np.mean([r.relevance_score for r in results]),
            'avg_completeness': np.mean([r.completeness_score for r in results]),
            'avg_rouge_1_f1': np.mean([r.rouge_1_f1 for r in results]),
            'avg_rouge_2_f1': np.mean([r.rouge_2_f1 for r in results]),
            'avg_rouge_l_f1': np.mean([r.rouge_l_f1 for r in results]),
            'avg_bleu_score': np.mean([r.bleu_score for r in results]),
            'avg_source_quality': np.mean([r.source_quality_score for r in results]),
            'avg_response_time': np.mean([r.response_time for r in results]),
            'avg_overall_score': np.mean([r.overall_score for r in results]),
            'std_overall_score': np.std([r.overall_score for r in results])
        }
        
        return metrics
    
    def save_results(self, results: List[EvaluationResult], aggregate_metrics: Dict[str, float]):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_results = []
        for result in results:
            detailed_results.append({
                'question': result.question,
                'predicted_answer': result.predicted_answer,
                'reference_answer': result.reference_answer,
                'accuracy_score': result.accuracy_score,
                'relevance_score': result.relevance_score,
                'completeness_score': result.completeness_score,
                'rouge_1_f1': result.rouge_1_f1,
                'rouge_2_f1': result.rouge_2_f1,
                'rouge_l_f1': result.rouge_l_f1,
                'bleu_score': result.bleu_score,
                'source_quality_score': result.source_quality_score,
                'response_time': result.response_time,
                'overall_score': result.overall_score,
                'num_sources': len(result.sources_used)
            })
        
        # Save to JSON
        results_file = f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'aggregate_metrics': aggregate_metrics,
                'detailed_results': detailed_results
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Print summary
        self.print_summary(aggregate_metrics)
    
    def print_summary(self, metrics: Dict[str, float]):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("🔍 AVIATION RAG EVALUATION SUMMARY")
        print("="*60)
        print(f"📊 Overall Score:      {metrics['avg_overall_score']:.3f}")
        print(f"✅ Accuracy:          {metrics['avg_accuracy']:.3f}")
        print(f"🎯 Relevance:         {metrics['avg_relevance']:.3f}")
        print(f"📝 Completeness:      {metrics['avg_completeness']:.3f}")
        print(f"📄 ROUGE-L F1:        {metrics['avg_rouge_l_f1']:.3f}")
        print(f"🔤 BLEU Score:        {metrics['avg_bleu_score']:.3f}")
        print(f"📚 Source Quality:    {metrics['avg_source_quality']:.3f}")
        print(f"⏱️  Avg Response Time: {metrics['avg_response_time']:.2f}s")
        print("="*60)

def main():
    """Run the evaluation system."""
    print("🚁 Starting Aviation RAG Evaluation...")
    
    evaluator = RAGEvaluator()
    results = evaluator.run_evaluation()
    
    print("\n✅ Evaluation completed!")
    print(f"📁 Check the generated JSON file for detailed results.")

if __name__ == "__main__":
    main()
