# streamlit_app.py
import streamlit as st
import yaml
import json
import time
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from app.graph import build_graph
from utils.rate_limited_llm import RateLimitedChatGoogleGenerativeAI
from utils.rate_limiter import gemini_rate_limiter

load_dotenv()

st.set_page_config(page_title="CMAR - Diagnostic Assistant", page_icon="🏥", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; }
    .status-box { padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; background-color: #f0f2f6; }
    .severity-high { color: #dc3545; font-weight: bold; }
    .severity-medium { color: #ffc107; font-weight: bold; }
    .intake-header { font-size: 1.1rem; font-weight: bold; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    with open('config/config.yaml', 'r') as f: config = yaml.safe_load(f)
    api_key = os.getenv("GOOGLE_API_KEY") or config['gemini']['api_key']
    
    gemini_rate_limiter.configure(
        max_calls=config['gemini']['rate_limit']['max_calls'],
        time_window=config['gemini']['rate_limit']['time_window']
    )
    
    llm = RateLimitedChatGoogleGenerativeAI(
        model=config['gemini']['generation_model'], 
        google_api_key=api_key, 
        temperature=0
    )
    
    embeddings = HuggingFaceEmbeddings(
        model_name="NeuML/pubmedbert-base-embeddings",
        model_kwargs={'device': 'mps'}, # Change to 'cpu' or 'cuda' if needed
        encode_kwargs={'normalize_embeddings': True}
    )
    return llm, embeddings, config

def main():
    st.markdown('<div class="main-header">🏥 CMAR System</div>', unsafe_allow_html=True)
    
    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.header("1. Select Scenario")
        scenarios_dir = 'data/patient_scenarios'
        if os.path.exists(scenarios_dir):
            files = [f for f in os.listdir(scenarios_dir) if f.endswith('.json')]
            selected_file = st.selectbox("Choose Patient File", files)
        else:
            st.error("Scenario directory not found.")
            selected_file = None

        st.divider()
        
        # --- NEW: PATIENT INTAKE FORM ---
        st.header("2. Clinical Intake (Optional)")
        st.caption("Add structured data here to prevent the AI from asking basic questions.")
        
        col_a, col_b = st.columns(2)
        with col_a:
            age_input = st.text_input("Age", placeholder="e.g. 45")
        with col_b:
            gender_input = st.selectbox("Gender", ["Unspecified", "Male", "Female", "Other"])
            
        with st.expander("🩺 Vital Signs & History", expanded=False):
            bp_input = st.text_input("BP (mmHg)", placeholder="120/80")
            hr_input = st.text_input("Heart Rate (bpm)", placeholder="80")
            temp_input = st.text_input("Temp (°C/°F)", placeholder="98.6 F")
            rr_input = st.text_input("Resp. Rate", placeholder="16")
            spo2_input = st.text_input("SpO2 (%)", placeholder="98")
            pmh_input = st.text_area("Past Medical History", placeholder="e.g. Diabetes, Hypertension...")

        st.divider()
        
        # Run Button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            st.session_state['running'] = True
            st.session_state['current_scenario_file'] = selected_file
            # Reset workflow state
            if 'graph_state' in st.session_state: del st.session_state['graph_state']
            if 'human_input_needed' in st.session_state: del st.session_state['human_input_needed']
            if 'extra_info' in st.session_state: del st.session_state['extra_info']
            
            # Store intake data in session to persist during run
            st.session_state['intake_data'] = {
                'Age': age_input,
                'Gender': gender_input,
                'BP': bp_input,
                'HR': hr_input,
                'Temp': temp_input,
                'RR': rr_input,
                'SpO2': spo2_input,
                'History': pmh_input
            }

    # --- MAIN CONTENT ---
    
    # Load Scenario Logic
    scenario_data = {}
    if 'current_scenario_file' in st.session_state and st.session_state['current_scenario_file']:
        try:
            with open(os.path.join('data/patient_scenarios', st.session_state['current_scenario_file'])) as f:
                base_data = json.load(f)
            
            # --- MERGE INTAKE DATA WITH SUMMARY ---
            # This ensures the Agents see the vitals immediately
            intake = st.session_state.get('intake_data', {})
            
            # Build a formatted string of vitals
            vitals_str = []
            if intake.get('Age'): vitals_str.append(f"Age: {intake['Age']}")
            if intake.get('Gender') and intake['Gender'] != "Unspecified": vitals_str.append(f"Gender: {intake['Gender']}")
            if intake.get('BP'): vitals_str.append(f"BP: {intake['BP']}")
            if intake.get('HR'): vitals_str.append(f"HR: {intake['HR']}")
            if intake.get('Temp'): vitals_str.append(f"Temp: {intake['Temp']}")
            if intake.get('SpO2'): vitals_str.append(f"SpO2: {intake['SpO2']}")
            
            structured_context = ""
            if vitals_str:
                structured_context += f"**Patient Vitals/Demographics:** {', '.join(vitals_str)}\n"
            if intake.get('History'):
                structured_context += f"**Past Medical History:** {intake['History']}\n"
                
            # Combine with original summary
            # If we have extra info from a previous "Ask Human" loop, append that too
            extra_info = st.session_state.get('extra_info', '')
            
            full_summary = f"{structured_context}\n**Clinical Presentation:**\n{base_data.get('summary', '')}\n{extra_info}"
            
            scenario_data = base_data.copy()
            scenario_data['summary'] = full_summary.strip()
            
            # Display the current context being analyzed
            st.info(scenario_data['summary'])
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return

    # Initialize Graph
    try:
        llm, embeddings, config = load_resources()
        app = build_graph(llm, embeddings)
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return

    # --- RUNNING THE WORKFLOW ---
    if st.session_state.get('running', False):
        
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        initial_state = {"patient_scenario": scenario_data}
        
        try:
            for output in app.stream(initial_state):
                node_name = list(output.keys())[0]
                current_state = output[node_name]
                st.session_state['graph_state'] = current_state
                
                if node_name == "hypothesis_generator":
                    progress_bar.progress(20)
                    status_container.markdown("✅ **Hypotheses Generated**")
                elif node_name == "evidence_evaluator":
                    progress_bar.progress(40)
                    status_container.markdown("✅ **Evidence Gathered**")
                elif node_name == "risk_assessor":
                    progress_bar.progress(60)
                    status_container.markdown("✅ **Risks Assessed**")
                elif node_name == "critic":
                    progress_bar.progress(80)
                    decision = current_state.get('critic_feedback', {}).get('decision')
                    status_container.markdown(f"🤔 **Critic Review:** {decision}")
                    
                    # HITL Check
                    if decision == "ASK_HUMAN":
                        st.session_state['human_input_needed'] = True
                        st.session_state['questions'] = current_state['critic_feedback'].get('questions_for_human', [])
                        st.session_state['running'] = False
                        break 
                        
                elif node_name == "synthesizer":
                    progress_bar.progress(100)
                    status_container.success("🎉 **Analysis Complete**")
                    st.session_state['running'] = False
                    
        except Exception as e:
            st.error(f"Runtime Error: {e}")
            st.session_state['running'] = False

    # --- HUMAN INPUT FORM (Paused State) ---
    if st.session_state.get('human_input_needed'):
        st.warning("🛑 **The AI needs additional information.**")
        st.caption("Please answer the questions below to refine the diagnosis.")
        
        questions = st.session_state.get('questions', [])
        answers = {}
        
        with st.form("human_input_form"):
            # Iterate through questions list (Batching fix from previous step)
            for i, q in enumerate(questions):
                st.markdown(f"**{i+1}. {q}**")
                answers[q] = st.text_area(f"Answer:", key=f"q_{i}", height=68)
            
            submitted = st.form_submit_button("Submit & Continue")
            
            if submitted:
                # Format responses
                combined_response = "\n".join([f"Q: {q}\nA: {a}" for q, a in answers.items() if a.strip()])
                
                if combined_response:
                    if 'extra_info' not in st.session_state:
                        st.session_state['extra_info'] = ""
                    st.session_state['extra_info'] += f"\n\n[Clinician Responses]:\n{combined_response}"
                
                st.session_state['human_input_needed'] = False
                st.session_state['running'] = True
                st.rerun()

    # --- FINAL RESULTS ---
    state = st.session_state.get('graph_state', {})
    if 'final_report' in state and not st.session_state.get('running'):
        report = state['final_report']
        
        st.divider()
        st.markdown("## 📋 Final Diagnostic Report")
        
        # Display Top 3
        for dx in report.get('differential_diagnoses', [])[:3]:
            with st.expander(f"**{dx['rank']}. {dx['diagnosis']}** (Likelihood: {dx['likelihood']}/10)", expanded=True):
                st.write(f"**Severity:** {dx['severity']}/10")
                st.write(f"**Reasoning:** {dx['justification']}")
        
        # Download
        st.download_button(
            "Download Full Analysis (JSON)",
            data=json.dumps(state, default=str, indent=2),
            file_name=f"cmar_report_{int(time.time())}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()