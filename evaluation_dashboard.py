"""
Evaluation Dashboard for Streamlit Integration
Provides a UI for running and viewing LLM-as-a-Judge evaluations
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from llm_judge_evaluator import RAGEvaluator
import time
from datetime import datetime

def load_latest_evaluation_results():
    """Load the most recent evaluation results."""
    results_files = list(Path(".").glob("evaluation_results_*.json"))
    if not results_files:
        return None
    
    latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def display_evaluation_dashboard():
    """Display the evaluation dashboard in Streamlit."""
    st.markdown("## 🔍 LLM-as-a-Judge Evaluation Dashboard")
    
    # Load existing results
    existing_results = load_latest_evaluation_results()
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 🚀 Run New Evaluation")
        
        num_cases = st.slider(
            "Number of test cases",
            min_value=1,
            max_value=10,
            value=5,
            help="Select how many test cases to evaluate"
        )
        
        if st.button("🔄 Run Evaluation", type="primary"):
            with st.spinner("Running comprehensive evaluation..."):
                try:
                    evaluator = RAGEvaluator()
                    results = evaluator.run_evaluation(num_cases=num_cases)
                    st.success("✅ Evaluation completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")
        
        st.markdown("---")
        
        if existing_results:
            st.markdown("### 📊 Quick Stats")
            metrics = existing_results['aggregate_metrics']
            
            st.metric(
                "Overall Score",
                f"{metrics['avg_overall_score']:.3f}",
                help="Weighted combination of all metrics"
            )
            
            st.metric(
                "Accuracy",
                f"{metrics['avg_accuracy']:.3f}",
                help="Factual correctness evaluation"
            )
            
            st.metric(
                "ROUGE-L F1",
                f"{metrics['avg_rouge_l_f1']:.3f}",
                help="Lexical similarity with reference"
            )
    
    with col1:
        if existing_results:
            st.markdown("### 📈 Evaluation Results")
            
            # Display timestamp
            timestamp = existing_results['timestamp']
            st.info(f"📅 Last evaluation: {timestamp}")
            
            # Aggregate metrics
            metrics = existing_results['aggregate_metrics']
            
            # Create metrics visualization
            metric_names = ['Accuracy', 'Relevance', 'Completeness', 'ROUGE-L', 'BLEU', 'Source Quality']
            metric_values = [
                metrics['avg_accuracy'],
                metrics['avg_relevance'],
                metrics['avg_completeness'],
                metrics['avg_rouge_l_f1'],
                metrics['avg_bleu_score'],
                metrics['avg_source_quality']
            ]
            
            # Radar chart
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=metric_values,
                theta=metric_names,
                fill='toself',
                name='Scores',
                line_color='#1f77b4'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                title="📊 Performance Across Metrics",
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Detailed results table
            st.markdown("### 📝 Detailed Results")
            
            detailed_results = existing_results['detailed_results']
            df = pd.DataFrame(detailed_results)
            
            # Select columns to display
            display_columns = [
                'question', 'overall_score', 'accuracy_score', 
                'relevance_score', 'completeness_score', 'rouge_l_f1', 
                'bleu_score', 'response_time'
            ]
            
            df_display = df[display_columns].copy()
            df_display.columns = [
                'Question', 'Overall', 'Accuracy', 'Relevance', 
                'Completeness', 'ROUGE-L', 'BLEU', 'Time (s)'
            ]
            
            # Format numeric columns
            numeric_cols = ['Overall', 'Accuracy', 'Relevance', 'Completeness', 'ROUGE-L', 'BLEU']
            for col in numeric_cols:
                df_display[col] = df_display[col].round(3)
            df_display['Time (s)'] = df_display['Time (s)'].round(2)
            
            st.dataframe(df_display, use_container_width=True)
            
            # Individual question analysis
            st.markdown("### 🔍 Question Analysis")
            
            selected_question = st.selectbox(
                "Select a question to analyze:",
                options=range(len(detailed_results)),
                format_func=lambda x: f"Q{x+1}: {detailed_results[x]['question'][:50]}..."
            )
            
            if selected_question is not None:
                result = detailed_results[selected_question]
                
                col1_q, col2_q = st.columns(2)
                
                with col1_q:
                    st.markdown("**Question:**")
                    st.write(result['question'])
                    
                    st.markdown("**Generated Answer:**")
                    st.write(result['predicted_answer'])
                
                with col2_q:
                    st.markdown("**Reference Answer:**")
                    st.write(result['reference_answer'])
                    
                    st.markdown("**Scores:**")
                    score_data = {
                        'Metric': ['Accuracy', 'Relevance', 'Completeness', 'ROUGE-L', 'BLEU'],
                        'Score': [
                            result['accuracy_score'],
                            result['relevance_score'],
                            result['completeness_score'],
                            result['rouge_l_f1'],
                            result['bleu_score']
                        ]
                    }
                    
                    fig_bar = px.bar(
                        score_data, 
                        x='Metric', 
                        y='Score',
                        title=f"Scores for Question {selected_question + 1}",
                        color='Score',
                        color_continuous_scale='viridis'
                    )
                    fig_bar.update_layout(height=300)
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # Performance trends (if multiple evaluation files exist)
            st.markdown("### 📈 Performance Trends")
            results_files = list(Path(".").glob("evaluation_results_*.json"))
            
            if len(results_files) > 1:
                trend_data = []
                for file in sorted(results_files, key=lambda x: x.stat().st_mtime):
                    with open(file, 'r') as f:
                        data = json.load(f)
                        trend_data.append({
                            'timestamp': data['timestamp'],
                            'overall_score': data['aggregate_metrics']['avg_overall_score'],
                            'accuracy': data['aggregate_metrics']['avg_accuracy'],
                            'rouge_l': data['aggregate_metrics']['avg_rouge_l_f1']
                        })
                
                if len(trend_data) > 1:
                    trend_df = pd.DataFrame(trend_data)
                    
                    fig_trend = px.line(
                        trend_df, 
                        x='timestamp', 
                        y=['overall_score', 'accuracy', 'rouge_l'],
                        title="Performance Over Time",
                        labels={'value': 'Score', 'variable': 'Metric'}
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("Run multiple evaluations to see performance trends.")
            else:
                st.info("Run multiple evaluations to see performance trends.")
        
        else:
            st.info("👆 Run your first evaluation to see results here!")
            
            st.markdown("""
            ### 🎯 What This Evaluates
            
            The LLM-as-a-Judge system measures:
            
            - **🎯 Accuracy**: Factual correctness of aviation information
            - **📍 Relevance**: How well answers address the questions
            - **📝 Completeness**: Comprehensiveness of responses
            - **📄 ROUGE Scores**: Lexical similarity with reference answers
            - **🔤 BLEU Score**: N-gram overlap with references
            - **📚 Source Quality**: Relevance of retrieved documents
            - **⏱️ Response Time**: Speed of answer generation
            
            ### 📊 Scoring
            - **Overall Score**: Weighted combination (0.0 - 1.0)
            - **Individual Metrics**: Each scored 0.0 - 1.0
            - **Higher is Better**: All metrics
            """)

def main():
    """Main function for standalone evaluation dashboard."""
    st.set_page_config(
        page_title="Aviation RAG Evaluation",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Aviation RAG Evaluation Dashboard")
    display_evaluation_dashboard()

if __name__ == "__main__":
    main()
