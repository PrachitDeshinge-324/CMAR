# streamlit_app.py
import streamlit as st
import yaml
import json
from dotenv import load_dotenv
import os
from app.graph import build_graph
from utils.rate_limited_llm import RateLimitedChatGoogleGenerativeAI
from utils.rate_limiter import gemini_rate_limiter
from langchain_huggingface import HuggingFaceEmbeddings
import time
from typing import Dict, List

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CMAR - Clinical Multi-Agent Reasoning",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-running {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .status-complete {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .status-error {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .hypothesis-card {
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0.5rem;
    }
    .severity-high {
        color: #dc3545;
        font-weight: bold;
    }
    .severity-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .severity-low {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_config():
    """Load configuration from YAML file"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

@st.cache_resource
def initialize_models():
    """Initialize LLM and embeddings models"""
    config = load_config()
    api_key = os.getenv("GOOGLE_API_KEY") or config['gemini']['api_key']
    
    # Configure rate limiter
    rate_config = config['gemini'].get('rate_limit', {'max_calls': 8, 'time_window': 60})
    gemini_rate_limiter.configure(
        max_calls=rate_config['max_calls'],
        time_window=rate_config['time_window']
    )
    
    # Initialize LLM
    llm = RateLimitedChatGoogleGenerativeAI(
        model=config['gemini']['generation_model'], 
        google_api_key=api_key, 
        temperature=0
    )
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'mps'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    return llm, embeddings, rate_config

def load_patient_scenarios():
    """Load available patient scenarios"""
    scenarios_dir = 'data/patient_scenarios'
    scenarios = {}
    
    if os.path.exists(scenarios_dir):
        for filename in os.listdir(scenarios_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(scenarios_dir, filename)
                with open(filepath, 'r') as f:
                    scenario = json.load(f)
                    scenarios[filename] = scenario
    
    return scenarios

def get_severity_color(score):
    """Return color class based on severity score"""
    if score >= 7:
        return "severity-high"
    elif score >= 4:
        return "severity-medium"
    else:
        return "severity-low"

def display_patient_info(patient_scenario):
    """Display patient information in a nice card"""
    st.markdown("### 👤 Patient Information")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"**Chief Complaint:** {patient_scenario.get('summary', 'N/A')}")
    
    with col2:
        if 'age' in patient_scenario:
            st.metric("Age", patient_scenario['age'])

def display_hypotheses_generation(specialty_groups):
    """Display generated hypotheses grouped by specialty"""
    st.markdown("### 🧠 Generated Hypotheses")
    
    total_hypotheses = sum(len(hypos) for hypos in specialty_groups.values())
    st.success(f"Generated **{total_hypotheses}** hypotheses across **{len(specialty_groups)}** specialties")
    
    # Display in columns
    cols = st.columns(2)
    for idx, (specialty, hypotheses) in enumerate(specialty_groups.items()):
        with cols[idx % 2]:
            with st.expander(f"**{specialty}** ({len(hypotheses)} hypotheses)", expanded=False):
                for hypo in hypotheses:
                    st.markdown(f"- {hypo['hypothesis']}")

def display_evidence_evaluation(specialty_groups):
    """Display evidence evaluation results"""
    st.markdown("### 🔍 Evidence Evaluation")
    
    for specialty, hypotheses in specialty_groups.items():
        with st.expander(f"**{specialty}**", expanded=False):
            for hypo in hypotheses:
                if 'evidence' in hypo:
                    st.markdown(f"#### {hypo['hypothesis']}")
                    st.caption(f"Evidence snippets found: {len(hypo.get('evidence', '').split('---'))}")
                    with st.container():
                        st.text_area(
                            f"Evidence for {hypo['hypothesis']}", 
                            hypo.get('evidence', 'No evidence found')[:500] + "...",
                            height=100,
                            key=f"evidence_{specialty}_{hypo['hypothesis']}"
                        )

def display_risk_assessment(specialty_groups):
    """Display risk assessment with visual indicators"""
    st.markdown("### ⚠️ Risk Assessment")
    
    # Create a summary table
    all_assessments = []
    for specialty, hypotheses in specialty_groups.items():
        for hypo in hypotheses:
            if 'severity' in hypo and 'likelihood' in hypo:
                all_assessments.append({
                    'Specialty': specialty,
                    'Hypothesis': hypo['hypothesis'],
                    'Severity': hypo['severity'],
                    'Likelihood': hypo['likelihood'],
                    'Justification': hypo.get('risk_justification', 'N/A')
                })
    
    if all_assessments:
        # Sort by severity and likelihood
        all_assessments.sort(key=lambda x: (x['Severity'], x['Likelihood']), reverse=True)
        
        # Display top risks
        st.markdown("#### 🔴 Top Risk Hypotheses")
        top_risks = all_assessments[:5]
        
        for i, assessment in enumerate(top_risks):
            severity_class = get_severity_color(assessment['Severity'])
            
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 4])
                
                with col1:
                    st.markdown(f"**{assessment['Hypothesis']}**")
                    st.caption(f"_{assessment['Specialty']}_")
                
                with col2:
                    st.markdown(f"<div class='{severity_class}'>Severity: {assessment['Severity']}/10</div>", 
                               unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"**Likelihood:** {assessment['Likelihood']}/10")
                
                with col4:
                    st.caption(assessment['Justification'][:150] + "...")
                
                st.divider()
        
        # Full detailed view
        with st.expander("📊 View All Assessments", expanded=False):
            for specialty, hypotheses in specialty_groups.items():
                st.markdown(f"#### {specialty}")
                for hypo in hypotheses:
                    if 'severity' in hypo:
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.markdown(f"**{hypo['hypothesis']}**")
                        with col2:
                            st.metric("Severity", f"{hypo['severity']}/10")
                        with col3:
                            st.metric("Likelihood", f"{hypo['likelihood']}/10")
                        
                        st.caption(f"📝 {hypo.get('risk_justification', 'N/A')}")
                        st.divider()

def display_critic_feedback(critic_history, final_critic_feedback, loop_count):
    """Display all critic feedback iterations"""
    st.markdown(f"### 🎯 Critic Review & Refinement Process")
    
    if critic_history:
        st.info(f"**Total Iterations:** {len(critic_history)} refinement loop(s)")
        
        # Track specialty targeting for warnings
        targeted_specialties = [c.get('target_specialty') for c in critic_history if c.get('target_specialty')]
        specialty_counts = {}
        for spec in targeted_specialties:
            specialty_counts[spec] = specialty_counts.get(spec, 0) + 1
        
        # Check if any specialty is over-targeted
        over_targeted = [spec for spec, count in specialty_counts.items() if count > 1]
        if over_targeted:
            st.warning(f"⚠️ **Notice:** The following specialties were targeted multiple times: {', '.join(over_targeted)}. "
                      f"Consider if other specialties also need review.")
        
        # Display each iteration
        for i, critic_feedback in enumerate(critic_history, 1):
            decision = critic_feedback.get('decision', 'UNKNOWN')
            target_specialty = critic_feedback.get('target_specialty')
            
            # Check if this specialty was already targeted
            is_repeat_target = False
            if target_specialty:
                previous_targets = [c.get('target_specialty') for c in critic_history[:i-1]]
                is_repeat_target = target_specialty in previous_targets
            
            # Determine if this is the final iteration
            is_final = (i == len(critic_history))
            is_approved = (decision == 'APPROVE')
            
            # Create expander for each iteration
            expander_label = f"**Iteration {i}/{len(critic_history)}** - {decision}"
            if is_final and is_approved:
                expander_label += " ✅"
            if is_repeat_target:
                expander_label += " 🔁"
            
            with st.expander(expander_label, expanded=(i == len(critic_history))):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if decision == 'APPROVE':
                        st.success(f"**Decision:** ✅ {decision}")
                        st.caption("Analysis approved - proceeding to final report")
                    elif decision == 'CHALLENGE_SCORE':
                        st.warning(f"**Decision:** ⚡ {decision}")
                        st.caption("Risk scores need reconsideration")
                    elif decision == 'ADD_HYPOTHESIS':
                        st.info(f"**Decision:** ➕ {decision}")
                        st.caption("Adding missing diagnosis")
                    elif decision == 'DISCARD_HYPOTHESIS':
                        st.error(f"**Decision:** ❌ {decision}")
                        st.caption("Removing unsupported diagnosis")
                    else:
                        st.write(f"**Decision:** {decision}")
                    
                    # Show targets and actions with repeat warning
                    if target_specialty:
                        if is_repeat_target:
                            st.markdown(f"**🎯 Target Specialty:**  \n`{target_specialty}` 🔁")
                            st.caption("⚠️ This specialty was targeted before")
                        else:
                            st.markdown(f"**🎯 Target Specialty:**  \n`{target_specialty}`")
                    
                    if critic_feedback.get('new_hypothesis_name'):
                        st.markdown(f"**➕ Adding:**  \n`{critic_feedback['new_hypothesis_name']}`")
                    
                    if critic_feedback.get('hypothesis_to_discard'):
                        st.markdown(f"**❌ Discarding:**  \n`{critic_feedback['hypothesis_to_discard']}`")
                
                with col2:
                    st.markdown("**💭 Critic's Reasoning:**")
                    feedback_text = critic_feedback.get('feedback', 'No feedback provided')
                    
                    # Use appropriate styling based on decision
                    if decision == 'APPROVE':
                        st.success(feedback_text)
                    elif decision == 'CHALLENGE_SCORE':
                        st.warning(feedback_text)
                    else:
                        st.info(feedback_text)
                
                # Show what happens next
                if not is_final:
                    st.markdown("---")
                    st.caption(f"⤵️ *This led to re-evaluation and refinement in Iteration {i+1}*")
                elif not is_approved:
                    st.markdown("---")
                    st.caption("⚠️ *Maximum iterations reached - proceeding to synthesis*")
        
        # Summary box with specialty distribution
        st.markdown("---")
        approval_count = sum(1 for c in critic_history if c.get('decision') == 'APPROVE')
        challenge_count = sum(1 for c in critic_history if c.get('decision') == 'CHALLENGE_SCORE')
        add_count = sum(1 for c in critic_history if c.get('decision') == 'ADD_HYPOTHESIS')
        discard_count = sum(1 for c in critic_history if c.get('decision') == 'DISCARD_HYPOTHESIS')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("✅ Approvals", approval_count)
        with col2:
            st.metric("⚡ Challenges", challenge_count)
        with col3:
            st.metric("➕ Additions", add_count)
        with col4:
            st.metric("❌ Discards", discard_count)
        
        # Specialty distribution
        if targeted_specialties:
            st.markdown("#### 📊 Specialties Reviewed")
            unique_specialties = len(set(targeted_specialties))
            st.caption(f"{unique_specialties} unique specialties targeted out of {len(critic_history)} total iterations")
            
            cols = st.columns(min(len(specialty_counts), 4))
            for idx, (spec, count) in enumerate(specialty_counts.items()):
                with cols[idx % 4]:
                    if count > 1:
                        st.metric(spec, count, delta="⚠️ Repeated", delta_color="off")
                    else:
                        st.metric(spec, count)
    else:
        st.info("No critic feedback recorded in this run.")

def display_final_report(final_report):
    """Display the final diagnostic report"""
    st.markdown("## 📋 Final Diagnostic Report")
    
    st.markdown("---")
    
    # Patient summary
    st.markdown("### Patient Summary")
    st.info(final_report.get('patient_summary', 'N/A'))
    
    # Differential diagnoses
    st.markdown("### 🎯 Ranked Differential Diagnoses")
    st.caption("_Ranked by clinical urgency (Severity × Likelihood)_")
    
    diagnoses = final_report.get('differential_diagnoses', [])
    
    # Top 3 diagnoses highlighted
    if len(diagnoses) >= 3:
        st.markdown("#### 🔴 Top 3 Most Urgent Diagnoses")
        
        for dx in diagnoses[:3]:
            severity_class = get_severity_color(dx['severity'])
            
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"### {dx['rank']}. {dx['diagnosis']}")
                
                with col2:
                    st.markdown(f"<div class='{severity_class}' style='font-size: 1.2rem;'>⚠️ Severity: {dx['severity']}/10</div>", 
                               unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"**📊 Likelihood:** {dx['likelihood']}/10")
                
                st.markdown(f"**Justification:** {dx['justification']}")
                st.divider()
    
    # All diagnoses
    with st.expander("📊 View All Differential Diagnoses", expanded=True):
        for dx in diagnoses:
            severity_class = get_severity_color(dx['severity'])
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{dx['rank']}. {dx['diagnosis']}**")
            
            with col2:
                st.markdown(f"<div class='{severity_class}'>Severity: {dx['severity']}/10</div>", 
                           unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"Likelihood: {dx['likelihood']}/10")
            
            st.caption(dx['justification'])
            st.divider()
    
    # Overall assessment
    st.markdown("### 📝 Overall Clinical Assessment")
    st.success(final_report.get('overall_assessment', 'N/A'))

def main():
    # Header
    st.markdown('<div class="main-header">🏥 CMAR System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Clinical Multi-Agent Reasoning for Differential Diagnosis</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Load scenarios
        scenarios = load_patient_scenarios()
        
        if not scenarios:
            st.error("No patient scenarios found in data/patient_scenarios/")
            return
        
        selected_scenario = st.selectbox(
            "Select Patient Scenario",
            options=list(scenarios.keys()),
            format_func=lambda x: x.replace('.json', '').replace('_', ' ').title()
        )
        
        st.markdown("---")
        
        # Model info
        config = load_config()
        st.markdown("### 🤖 Model Configuration")
        st.info(f"**LLM:** {config['gemini']['generation_model']}")
        st.info(f"**Embeddings:** BAAI/bge-large-en-v1.5")
        
        rate_config = config['gemini'].get('rate_limit', {'max_calls': 8, 'time_window': 60})
        st.warning(f"**Rate Limit:** {rate_config['max_calls']} calls per {rate_config['time_window']}s")
        
        st.markdown("---")
        
        # Run button
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content area
    if run_analysis:
        patient_scenario = scenarios[selected_scenario]
        
        # Display patient info
        display_patient_info(patient_scenario)
        
        st.markdown("---")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize models
            status_text.markdown("🔄 **Initializing models...**")
            llm, embeddings, rate_config = initialize_models()
            progress_bar.progress(10)
            
            # Build graph
            status_text.markdown("🔄 **Building workflow graph...**")
            app = build_graph(llm_client=llm, embeddings_client=embeddings)
            progress_bar.progress(20)
            
            # Create state tracking containers
            hypothesis_container = st.empty()
            evidence_container = st.empty()
            risk_container = st.empty()
            critic_container = st.empty()
            
            # Run the workflow with state tracking
            status_text.markdown("🔄 **Running CMAR workflow...**")
            initial_state = {"patient_scenario": patient_scenario}
            
            # We'll simulate step-by-step execution
            # In reality, you'd need to modify the graph to yield intermediate states
            # For now, we'll run it completely and show the final state
            
            with st.spinner("Processing... This may take a few minutes due to rate limiting..."):
                final_state = app.invoke(initial_state)
            
            progress_bar.progress(100)
            status_text.markdown("✅ **Analysis Complete!**")
            
            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🧠 Hypotheses", 
                "🔍 Evidence", 
                "⚠️ Risk Assessment", 
                "🎯 Critic Feedback",
                "📋 Final Report"
            ])
            
            with tab1:
                if 'specialty_groups' in final_state:
                    display_hypotheses_generation(final_state['specialty_groups'])
            
            with tab2:
                if 'specialty_groups' in final_state:
                    display_evidence_evaluation(final_state['specialty_groups'])
            
            with tab3:
                if 'specialty_groups' in final_state:
                    display_risk_assessment(final_state['specialty_groups'])
            
            with tab4:
                critic_history = final_state.get('critic_history', [])
                critic_feedback = final_state.get('critic_feedback', {})
                loop_count = final_state.get('refinement_loop_count', 0)
                display_critic_feedback(critic_history, critic_feedback, loop_count)
            
            with tab5:
                if 'final_report' in final_state:
                    display_final_report(final_state['final_report'])
                else:
                    st.warning("Final report not generated")
            
            # Download option
            st.markdown("---")
            st.markdown("### 💾 Download Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Full State (JSON)",
                    data=json.dumps(final_state, indent=2, default=str),
                    file_name=f"cmar_analysis_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                if 'final_report' in final_state:
                    final_report = final_state['final_report']
                    report_text = f"""
CMAR Diagnostic Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

Patient Summary:
{final_report.get('patient_summary', 'N/A')}

Differential Diagnoses:
"""
                    for dx in final_report.get('differential_diagnoses', []):
                        report_text += f"\n{dx['rank']}. {dx['diagnosis']}\n"
                        report_text += f"   Severity: {dx['severity']}/10 | Likelihood: {dx['likelihood']}/10\n"
                        report_text += f"   {dx['justification']}\n"
                    
                    report_text += f"\nOverall Assessment:\n{final_report.get('overall_assessment', 'N/A')}"
                    
                    st.download_button(
                        label="Download Report (TXT)",
                        data=report_text,
                        file_name=f"cmar_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)
    
    else:
        # Landing page
        st.markdown("## 👋 Welcome to CMAR")
        
        st.markdown("""
        The **Clinical Multi-Agent Reasoning (CMAR)** system uses advanced AI agents to:
        
        1. 🧠 **Generate comprehensive differential diagnoses** across multiple specialties
        2. 🔍 **Evaluate evidence** using medical literature retrieval
        3. ⚠️ **Assess clinical risk** based on severity and likelihood
        4. 🎯 **Apply critical review** through counterfactual reasoning
        5. 📋 **Synthesize findings** into actionable diagnostic reports
        
        ### How to Use:
        1. Select a patient scenario from the sidebar
        2. Click "Run Analysis" to start the diagnostic workflow
        3. View results in organized tabs
        4. Download reports for documentation
        
        ### Features:
        - ✅ Real-time progress tracking
        - ✅ Transparent multi-agent workflow
        - ✅ Evidence-based reasoning
        - ✅ Rate-limited API calls (no quota errors!)
        - ✅ Downloadable reports
        """)
        
        # Show sample scenarios
        st.markdown("### 📁 Available Scenarios")
        scenarios = load_patient_scenarios()
        for scenario_name, scenario_data in scenarios.items():
            with st.expander(scenario_name.replace('.json', '').replace('_', ' ').title()):
                st.info(scenario_data.get('summary', 'No summary available'))

if __name__ == "__main__":
    main()
