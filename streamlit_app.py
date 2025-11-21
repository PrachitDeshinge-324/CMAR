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
    .severity-low { color: #28a745; font-weight: bold; }
    .hypothesis-card { padding: 1rem; border: 1px solid #ddd; border-radius: 0.5rem; margin: 0.5rem 0; background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS FOR DISPLAY ---
def get_severity_color(score):
    if score >= 7: return "severity-high"
    elif score >= 4: return "severity-medium"
    return "severity-low"

def display_hypotheses_generation(specialty_groups):
    st.markdown("### 🧠 Generated Hypotheses")
    cols = st.columns(2)
    if not specialty_groups:
        st.info("No hypotheses found in state.")
        return

    for idx, (specialty, hypotheses) in enumerate(specialty_groups.items()):
        with cols[idx % 2]:
            with st.expander(f"**{specialty}** ({len(hypotheses)})", expanded=False):
                for hypo in hypotheses:
                    st.markdown(f"- {hypo['hypothesis']}")

def display_evidence_evaluation(specialty_groups):
    st.markdown("### 🔍 Evidence Evaluation")
    if not specialty_groups:
        st.info("No evidence data found.")
        return

    for specialty, hypotheses in specialty_groups.items():
        with st.expander(f"**{specialty}**", expanded=False):
            for hypo in hypotheses:
                if 'evidence' in hypo:
                    st.markdown(f"#### {hypo['hypothesis']}")
                    st.caption(f"Evidence Content (First 500 chars):")
                    st.text_area(
                        f"Evidence:", 
                        hypo.get('evidence', 'No evidence found')[:500] + "...",
                        height=100,
                        key=f"ev_{specialty}_{hypo['hypothesis']}"
                    )

def display_risk_assessment(specialty_groups):
    st.markdown("### ⚠️ Risk Assessment")
    if not specialty_groups:
        st.info("No risk assessment data found.")
        return

    for specialty, hypotheses in specialty_groups.items():
        st.markdown(f"#### {specialty}")
        for hypo in hypotheses:
            if 'severity' in hypo:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1: st.markdown(f"**{hypo['hypothesis']}**")
                with col2: 
                    color = get_severity_color(hypo['severity'])
                    st.markdown(f":{color}[Severity: {hypo['severity']}/10]")
                with col3: st.markdown(f"Likelihood: {hypo['likelihood']}/10")
                st.caption(f"📝 {hypo.get('risk_justification', 'N/A')}")
                st.divider()

def display_critic_history(critic_history):
    st.markdown("### 🎯 Critic Review Process")
    if not critic_history:
        st.info("No critic feedback generated.")
        return
        
    for i, feedback in enumerate(critic_history, 1):
        decision = feedback.get('decision')
        with st.expander(f"Iteration {i}: {decision}", expanded=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**Action:** {decision}")
                if feedback.get('target_specialty'):
                    st.write(f"**Target:** {feedback.get('target_specialty')}")
            with col2:
                st.write(f"**Reasoning:** {feedback.get('feedback')}")
                
            if decision == "ASK_HUMAN":
                st.warning(f"❓ Asked Questions: {feedback.get('questions_for_human')}")

# --- RESOURCE LOADING ---
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
        model_kwargs={'device': 'cuda'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    return llm, embeddings, config

# --- MAIN APP ---
def main():
    st.markdown('<div class="main-header">🏥 CMAR System</div>', unsafe_allow_html=True)
    
    # Sidebar Intake
    with st.sidebar:
        st.header("1. Select Scenario")
        scenarios_dir = 'data/patient_scenarios'
        if os.path.exists(scenarios_dir):
            files = [f for f in os.listdir(scenarios_dir) if f.endswith('.json')]
            selected_file = st.selectbox("Choose Patient File", files)
        else:
            selected_file = None

        st.divider()
        st.header("2. Clinical Intake (Optional)")
        col_a, col_b = st.columns(2)
        with col_a: age_input = st.text_input("Age", placeholder="e.g. 45")
        with col_b: gender_input = st.selectbox("Gender", ["Unspecified", "Male", "Female"])
            
        with st.expander("🩺 Vital Signs & History"):
            bp_input = st.text_input("BP", placeholder="120/80")
            hr_input = st.text_input("HR", placeholder="80")
            temp_input = st.text_input("Temp", placeholder="98.6")
            pmh_input = st.text_area("Past History", placeholder="Diabetes...")

        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            st.session_state['running'] = True
            st.session_state['current_scenario_file'] = selected_file
            # Clear previous state completely
            if 'graph_state' in st.session_state: del st.session_state['graph_state']
            if 'human_input_needed' in st.session_state: del st.session_state['human_input_needed']
            if 'extra_info' in st.session_state: del st.session_state['extra_info']
            
            st.session_state['intake_data'] = {
                'Age': age_input, 'Gender': gender_input, 'BP': bp_input, 
                'HR': hr_input, 'Temp': temp_input, 'History': pmh_input
            }

    # Load Scenario & Merge Intake
    scenario_data = {}
    if 'current_scenario_file' in st.session_state and st.session_state['current_scenario_file']:
        try:
            with open(os.path.join('data/patient_scenarios', st.session_state['current_scenario_file'])) as f:
                base_data = json.load(f)
            
            intake = st.session_state.get('intake_data', {})
            vitals_str = [f"{k}: {v}" for k, v in intake.items() if v and k != 'History' and v != "Unspecified"]
            
            context = ""
            if vitals_str: context += f"**Vitals/Demographics:** {', '.join(vitals_str)}\n"
            if intake.get('History'): context += f"**History:** {intake['History']}\n"
            
            extra_info = st.session_state.get('extra_info', '')
            full_summary = f"{context}\n**Presentation:**\n{base_data.get('summary', '')}\n{extra_info}"
            
            scenario_data = base_data.copy()
            scenario_data['summary'] = full_summary.strip()
            st.info(scenario_data['summary'])
            
        except Exception as e:
            st.error(f"Error: {e}")
            return

    # Initialize
    try:
        llm, embeddings, config = load_resources()
        app = build_graph(llm, embeddings)
    except Exception as e:
        st.error(f"Init Error: {e}")
        return

    # Execution Loop
    if st.session_state.get('running', False):
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        # Ensure graph_state exists
        if 'graph_state' not in st.session_state:
            st.session_state['graph_state'] = {}

        try:
            # Initial state
            current_accumulated_state = {"patient_scenario": scenario_data}
            # Merge any existing state if resuming (though we clear it on new run)
            current_accumulated_state.update(st.session_state['graph_state'])
            
            for output in app.stream(current_accumulated_state):
                node_name = list(output.keys())[0]
                node_update = output[node_name]
                
                # --- FIX: MERGE UPDATE INTO STATE INSTEAD OF OVERWRITING ---
                st.session_state['graph_state'].update(node_update)
                
                # Use the updated state for logic
                current_state = st.session_state['graph_state']
                
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
                    # Critic feedback is nested
                    feedback = current_state.get('critic_feedback', {})
                    decision = feedback.get('decision')
                    status_container.markdown(f"🤔 **Critic Review:** {decision}")
                    
                    if decision == "ASK_HUMAN":
                        st.session_state['human_input_needed'] = True
                        st.session_state['questions'] = feedback.get('questions_for_human', [])
                        st.session_state['running'] = False
                        break 
                elif node_name == "synthesizer":
                    progress_bar.progress(100)
                    status_container.success("🎉 **Analysis Complete**")
                    st.session_state['running'] = False
                    
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state['running'] = False

    # HITL Form
    if st.session_state.get('human_input_needed'):
        st.warning("🛑 **Additional Information Needed**")
        questions = st.session_state.get('questions', [])
        answers = {}
        with st.form("human_input"):
            for i, q in enumerate(questions):
                st.markdown(f"**{i+1}. {q}**")
                answers[q] = st.text_area("Answer", key=f"q_{i}")
            if st.form_submit_button("Submit & Continue"):
                resp = "\n".join([f"Q: {q}\nA: {a}" for q, a in answers.items() if a])
                if 'extra_info' not in st.session_state: st.session_state['extra_info'] = ""
                st.session_state['extra_info'] += f"\n\n[Clinician Responses]:\n{resp}"
                st.session_state['human_input_needed'] = False
                st.session_state['running'] = True
                st.rerun()

    # --- RESTORED: FULL TRANSPARENCY DISPLAY ---
    state = st.session_state.get('graph_state', {})
    
    # Only show results if synthesizer has finished OR we have meaningful partial data (optional)
    # For now, let's show what we have if not running
    if not st.session_state.get('running') and ('final_report' in state or 'specialty_groups' in state):
        
        st.divider()
        st.markdown("## 📋 Results & Transparency")
        
        # Create Tabs
        tabs = ["📄 Final Report", "🧠 Hypotheses", "🔍 Evidence", "⚠️ Risk Assessment", "🎯 Critic History"]
        tab_report, tab_hypo, tab_evid, tab_risk, tab_critic = st.tabs(tabs)
        
        with tab_report:
            if 'final_report' in state:
                report = state['final_report']
                st.subheader("Differential Diagnoses")
                for dx in report.get('differential_diagnoses', [])[:5]:
                    with st.expander(f"{dx['rank']}. {dx['diagnosis']} (Likelihood: {dx['likelihood']}/10)", expanded=True):
                        st.write(f"**Severity:** {dx['severity']}")
                        st.write(f"**Reasoning:** {dx['justification']}")
                st.markdown("### Overall Assessment")
                st.info(report.get('overall_assessment'))
            else:
                st.info("Final report not yet generated.")

        with tab_hypo:
            if 'specialty_groups' in state:
                display_hypotheses_generation(state['specialty_groups'])
            else:
                st.warning("No hypotheses data found.")
        
        with tab_evid:
            if 'specialty_groups' in state:
                display_evidence_evaluation(state['specialty_groups'])
            else:
                st.warning("No evidence data found.")
                
        with tab_risk:
            if 'specialty_groups' in state:
                display_risk_assessment(state['specialty_groups'])
            else:
                st.warning("No risk data found.")
                
        with tab_critic:
            if 'critic_history' in state:
                display_critic_history(state['critic_history'])
            else:
                st.info("No critic history available.")

        # JSON Download
        st.download_button(
            "💾 Download Full Case Data",
            data=json.dumps(state, default=str, indent=2),
            file_name=f"cmar_case_{int(time.time())}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()