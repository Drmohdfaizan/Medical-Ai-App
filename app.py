import streamlit as st
import google.generativeai as genai
from datetime import datetime
import json
import base64
from PIL import Image
import io
import PyPDF2
from database import Database
import hashlib

# Configure Gemini API with your key
GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# Hide Streamlit elements (GitHub icon, Toolbar, Footer)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            .viewerBadge_container__1QS13 {display: none !important;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Professional System Prompt for Medical Analysis
MEDICAL_SYSTEM_PROMPT = """You are a Medical Diagnostic Expert AI with advanced training in clinical diagnosis, radiology, pathology, and pharmacology.

Your Core Responsibilities:
1. MEDICAL ANALYSIS: Provide accurate, evidence-based medical analysis of symptoms, lab reports, and medical images
2. STEP-BY-STEP REASONING: Always explain your diagnostic reasoning process clearly
3. REFERENCE RANGES: When analyzing lab reports, carefully compare each value against normal reference ranges and explain any deviations
4. IMAGING ANALYSIS: For X-rays, MRIs, CT scans, and other medical images, systematically identify:
   - Normal anatomical structures
   - Any structural abnormalities, lesions, or pathological findings
   - Density changes, alignment issues, or asymmetries
   - Recommendations for further imaging if needed
5. MEDICATION ANALYSIS: Cross-reference current medications with reported symptoms to identify potential side effects or drug interactions
6. DIFFERENTIAL DIAGNOSIS: Always provide 2-3 possible diagnoses ranked by likelihood with confidence levels
7. SCIENTIFIC BASIS: Reference relevant medical literature, studies, or clinical guidelines when available
8. SAFETY FIRST: Always indicate when immediate medical attention is required

Your Analysis Must Include:
- Clear, structured sections for easy reading
- Evidence-based reasoning for each conclusion
- Specific attention to abnormal findings with clinical significance
- Appropriate medical terminology adjusted to the user's mode (patient/doctor)
- Red flags that require urgent medical care
- Recommended next steps for diagnosis or treatment

Important Guidelines:
- Be factual and precise - do not speculate beyond available evidence
- Use proper medical terminology while maintaining clarity
- Always compare lab values against standard reference ranges
- Identify and explain any critical or concerning findings
- Maintain professional medical standards in all communications
- End every response with: "⚠️ DISCLAIMER: This analysis is for educational purposes only and should not replace professional medical consultation. Please consult a qualified healthcare provider for proper diagnosis and treatment."

Remember: Your goal is to provide the most accurate, helpful, and professionally sound medical analysis possible while emphasizing the importance of professional medical care."""

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DocPro-Ai",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR LOGO ---
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.divider()


# Initialize database
db = Database()

# Translations
TRANSLATIONS = {
    'en': {
        'title': 'DocPro-Ai',
        'login': 'Login',
        'signup': 'Sign Up',
        'username': 'Username',
        'password': 'Password',
        'email': 'Email',
        'logout': 'Logout',
        'symptoms': 'Describe your symptoms',
        'upload_image': 'Upload Medical Image',
        'upload_pdf': 'Upload Lab Report (PDF)',
        'analyze': 'Analyze',
        'save_vault': 'Save to Vault',
        'view_vault': 'View Health Vault',
        'current_meds': 'Current Medications',
        'patient_mode': 'Patient Mode',
        'doctor_mode': 'Doctor Mode',
        'disclaimer': '⚠️ For Educational Purposes Only. Not a substitute for professional medical advice.',
    },
    'hi': {
        'title': 'व्यावसायिक CDSS और स्वास्थ्य तिजोरी',
        'login': 'लॉगिन',
        'signup': 'साइन अप',
        'username': 'उपयोगकर्ता नाम',
        'password': 'पासवर्ड',
        'email': 'ईमेल',
        'logout': 'लॉगआउट',
        'symptoms': 'अपने लक्षण बताएं',
        'upload_image': 'मेडिकल इमेज अपलोड करें',
        'upload_pdf': 'लैब रिपोर्ट अपलोड करें (PDF)',
        'analyze': 'विश्लेषण करें',
        'save_vault': 'तिजोरी में सहेजें',
        'view_vault': 'स्वास्थ्य तिजोरी देखें',
        'current_meds': 'वर्तमान दवाएं',
        'patient_mode': 'रोगी मोड',
        'doctor_mode': 'डॉक्टर मोड',
        'disclaimer': '⚠️ केवल शैक्षिक उद्देश्यों के लिए। पेशेवर चिकित्सा सलाह का विकल्प नहीं।',
    },
    'hinglish': {
        'title': 'Professional CDSS aur Health Vault',
        'login': 'Login',
        'signup': 'Sign Up',
        'username': 'Username',
        'password': 'Password',
        'email': 'Email',
        'logout': 'Logout',
        'symptoms': 'Apne symptoms batayein',
        'upload_image': 'Medical image upload karein',
        'upload_pdf': 'Lab report upload karein (PDF)',
        'analyze': 'Analyze karein',
        'save_vault': 'Vault mein save karein',
        'view_vault': 'Health Vault dekhein',
        'current_meds': 'Current medicines',
        'patient_mode': 'Patient Mode',
        'doctor_mode': 'Doctor Mode',
        'disclaimer': '⚠️ Sirf educational purposes ke liye. Professional medical advice ka substitute nahi.',
    }
}

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'mode' not in st.session_state:
    st.session_state.mode = 'patient'
if 'conversation_state' not in st.session_state:
    st.session_state.conversation_state = 'initial'
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {}
if 'follow_up_count' not in st.session_state:
    st.session_state.follow_up_count = 0

def get_text(key):
    """Get translated text"""
    return TRANSLATIONS[st.session_state.language].get(key, key)

def init_gemini():
    """Initialize Gemini API with optimized configuration for medical analysis"""
    try:
        # Configure generation parameters for maximum accuracy
        generation_config = genai.GenerationConfig(
            temperature=0.1,  # Low temperature for factual, consistent responses
            top_p=0.95,       # High top_p for comprehensive analysis
            top_k=40,
            max_output_tokens=8192,
        )
        
        # Initialize model with Gemini 2.0 Flash and professional configuration
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            generation_config=generation_config,
            system_instruction=MEDICAL_SYSTEM_PROMPT
        )
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini: {str(e)}")
        return None

def get_language_name():
    """Get full language name"""
    lang_map = {
        'en': 'English',
        'hi': 'Hindi',
        'hinglish': 'Hinglish (mix of Hindi and English)'
    }
    return lang_map.get(st.session_state.language, 'English')

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def analyze_with_gemini(model, prompt, image=None):
    """Analyze with Gemini API using professional system prompt"""
    try:
        if image:
            response = model.generate_content([prompt, image])
        else:
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error during analysis: {str(e)}"

def create_diagnosis_prompt(symptoms, follow_up_answers, medications, mode, language):
    """Create diagnosis prompt for Gemini with professional context"""
    lang_instruction = {
        'en': 'Respond in English',
        'hi': 'Respond in Hindi',
        'hinglish': 'Respond in Hinglish (mix of Hindi and English)'
    }
    
    mode_instruction = {
        'patient': 'Use simple, easy-to-understand language suitable for patients. Avoid excessive medical jargon.',
        'doctor': 'Use technical medical terminology, include ICD-10 codes where applicable, and provide detailed clinical reasoning.'
    }
    
    prompt = f"""MEDICAL ANALYSIS REQUEST

LANGUAGE: {lang_instruction[language]}
COMMUNICATION MODE: {mode_instruction[mode]}

PRIMARY SYMPTOMS AND CLINICAL PRESENTATION:
{symptoms}

FOLLOW-UP INFORMATION GATHERED:
{follow_up_answers}

CURRENT MEDICATIONS:
{medications if medications else 'None reported'}

ANALYSIS REQUIREMENTS:
Please provide a comprehensive medical analysis following this structure:

1. DIFFERENTIAL DIAGNOSIS
   - List 2-3 possible diagnoses ranked by likelihood
   - Provide confidence level for each (High/Medium/Low)
   - Include relevant ICD-10 codes (if in Doctor Mode)

2. DETAILED CLINICAL REASONING
   - Explain the diagnostic reasoning step-by-step
   - Highlight key symptoms supporting each diagnosis
   - Note any contradicting or atypical presentations

3. MEDICATION ANALYSIS (if applicable)
   - Assess if any current medications could cause reported symptoms
   - Identify potential drug interactions or side effects
   - Note contraindications if any

4. LABORATORY/IMAGING FINDINGS ANALYSIS
   - If lab values provided, compare each against normal reference ranges
   - Explain clinical significance of any abnormal values
   - If medical images provided, systematically analyze for structural abnormalities

5. SCIENTIFIC BASIS AND EVIDENCE
   - Reference relevant medical literature or clinical guidelines
   - Cite studies or evidence supporting the diagnosis
   - Include PubMed references when available

6. RECOMMENDED NEXT STEPS
   - Suggest further diagnostic tests if needed
   - Provide treatment considerations
   - Lifestyle or management recommendations

7. RED FLAGS AND URGENT CARE INDICATORS
   - Identify symptoms requiring immediate medical attention
   - Note any critical or life-threatening possibilities
   - Specify when to seek emergency care

Remember to maintain professionalism and end with the standard disclaimer."""
    
    return prompt

def create_follow_up_questions(symptoms, language):
    """Generate follow-up questions based on initial symptoms"""
    lang_map = {
        'en': 'English',
        'hi': 'Hindi',
        'hinglish': 'Hinglish'
    }
    
    prompt = f"""Based on these symptoms: {symptoms}

As a Medical Diagnostic Expert, generate 4 clinically relevant follow-up questions in {lang_map[language]} that would help narrow down the differential diagnosis. These questions should gather information about:
- Onset, duration, and progression
- Severity and characteristics
- Aggravating or relieving factors
- Associated symptoms

For each question, provide 3-4 realistic answer options that patients would commonly report.

Format as JSON:
{{
    "questions": [
        {{
            "question": "question text",
            "options": ["option1", "option2", "option3", "option4"]
        }}
    ]
}}

Return ONLY the JSON, no other text."""
    
    return prompt

def login_page():
    """Login page"""
    st.title(get_text('title'))
    
    tab1, tab2 = st.tabs([get_text('login'), get_text('signup')])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input(get_text('username'))
            password = st.text_input(get_text('password'), type="password")
            submit = st.form_submit_button(get_text('login'))
            
            if submit:
                user = db.authenticate_user(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_id = user[0]
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input(get_text('username'))
            new_email = st.text_input(get_text('email'))
            new_password = st.text_input(get_text('password'), type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button(get_text('signup'))
            
            if submit:
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    if db.create_user(new_username, new_email, new_password):
                        st.success("Account created! Please login.")
                    else:
                        st.error("Username already exists")

def main_app():
    """Main application"""
    st.title(get_text('title'))
    
    # Sidebar
    with st.sidebar:
        st.write(f"👤 Welcome, {st.session_state.username}!")
        
        # Language selector
        lang_option = st.selectbox(
            "Language / भाषा",
            ['English', 'हिंदी', 'Hinglish'],
            index=['English', 'हिंदी', 'Hinglish'].index(
                {'en': 'English', 'hi': 'हिंदी', 'hinglish': 'Hinglish'}[st.session_state.language]
            )
        )
        st.session_state.language = {'English': 'en', 'हिंदी': 'hi', 'Hinglish': 'hinglish'}[lang_option]
        
        # Mode selector
        mode = st.radio(
            "Mode",
            [get_text('patient_mode'), get_text('doctor_mode')]
        )
        st.session_state.mode = 'patient' if mode == get_text('patient_mode') else 'doctor'
        
        st.divider()
        
        # Navigation
        page = st.radio("Navigation", ["Analyze Symptoms", "Health Vault"])
        
        if st.button(get_text('logout')):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.user_id = None
            st.rerun()
    
    # Main content
    if page == "Analyze Symptoms":
        analyze_symptoms_page()
    else:
        health_vault_page()
    
    # Footer
    st.divider()
    st.markdown(f"<div style='text-align: center; color: #ff6b6b; font-weight: bold;'>{get_text('disclaimer')}</div>", unsafe_allow_html=True)

def analyze_symptoms_page():
    """Symptoms analysis page"""
    model = init_gemini()
    
    if not model:
        st.error("⚠️ Failed to initialize Gemini API. Please check your API key and internet connection.")
        return
    
    st.header("Medical Analysis")
    
    # Show model configuration info
    #with st.expander("ℹ️ AI Configuration"):
        #st.info("""
        #**Model:** Gemini 2.0 Flash
        #**Configuration:** Optimized for medical accuracy
        #- Temperature: 0.1 (factual, consistent responses)
        #- Top P: 0.95 (comprehensive analysis)
        #- System Prompt: Medical Diagnostic Expert
        #""")
    
    # Input methods
    col1, col2 = st.columns(2)
    
    with col1:
        symptoms_text = st.text_area(
            get_text('symptoms'), 
            height=150, 
            key="symptoms_input",
            placeholder="Describe your symptoms in detail..."
        )
        
        medications = st.text_area(
            get_text('current_meds'), 
            height=100, 
            placeholder="List any medications you're currently taking"
        )
    
    with col2:
        uploaded_image = st.file_uploader(get_text('upload_image'), type=['png', 'jpg', 'jpeg'])
        uploaded_pdf = st.file_uploader(get_text('upload_pdf'), type=['pdf'])
    
    # Display uploaded files
    image_data = None
    pdf_text = None
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", width=300)
        image_data = image
    
    if uploaded_pdf:
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        with st.expander("PDF Content Preview"):
            st.text(pdf_text[:500] + "..." if len(pdf_text) > 500 else pdf_text)
    
    # Analyze button
    if st.button(get_text('analyze'), type="primary"):
        if not symptoms_text and not uploaded_image and not uploaded_pdf:
            st.error("Please provide symptoms, image, or PDF report")
            return
        
        # Combine all inputs
        full_input = symptoms_text
        if pdf_text:
            full_input += f"\n\nLab Report Content:\n{pdf_text}"
        
        # Store initial data
        st.session_state.analysis_data = {
            'symptoms': full_input,
            'medications': medications,
            'image': uploaded_image,
            'image_data': image_data,
            'follow_up_answers': {}
        }
        st.session_state.conversation_state = 'follow_up'
        st.session_state.follow_up_count = 0
        st.rerun()
    
    # Follow-up questions
    if st.session_state.conversation_state == 'follow_up':
        st.divider()
        st.subheader("Follow-up Questions")
        
        # Generate follow-up questions
        if st.session_state.follow_up_count < 4:
            questions_prompt = create_follow_up_questions(
                st.session_state.analysis_data['symptoms'],
                st.session_state.language
            )
            
            with st.spinner("Generating follow-up questions..."):
                response = analyze_with_gemini(model, questions_prompt)
                
                try:
                    # Extract JSON from response
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        questions_data = json.loads(json_str)
                        
                        if 'questions' in questions_data and st.session_state.follow_up_count < len(questions_data['questions']):
                            q = questions_data['questions'][st.session_state.follow_up_count]
                            
                            st.write(f"**Question {st.session_state.follow_up_count + 1}:** {q['question']}")
                            
                            # Create buttons for options
                            cols = st.columns(len(q['options']))
                            for i, option in enumerate(q['options']):
                                if cols[i].button(option, key=f"option_{st.session_state.follow_up_count}_{i}"):
                                    st.session_state.analysis_data['follow_up_answers'][q['question']] = option
                                    st.session_state.follow_up_count += 1
                                    st.rerun()
                        else:
                            st.session_state.conversation_state = 'diagnosis'
                            st.rerun()
                    else:
                        st.warning("Could not generate questions. Proceeding to analysis...")
                        st.session_state.conversation_state = 'diagnosis'
                        st.rerun()
                except Exception as e:
                    st.warning(f"Error generating questions: {str(e)}. Proceeding to analysis...")
                    st.session_state.conversation_state = 'diagnosis'
                    st.rerun()
        
        if st.session_state.follow_up_count >= 4 or st.button("Skip to Diagnosis"):
            st.session_state.conversation_state = 'diagnosis'
            st.rerun()
    
    # Final diagnosis
    if st.session_state.conversation_state == 'diagnosis':
        st.divider()
        st.subheader("📋 Professional Medical Analysis")
        
        # Prepare follow-up answers text
        follow_up_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.analysis_data['follow_up_answers'].items()])
        
        # Create diagnosis prompt
        diagnosis_prompt = create_diagnosis_prompt(
            st.session_state.analysis_data['symptoms'],
            follow_up_text,
            st.session_state.analysis_data['medications'],
            st.session_state.mode,
            st.session_state.language
        )
        
        with st.spinner("🔬 Analyzing with Professional Medical AI..."):
            if st.session_state.analysis_data.get('image_data'):
                result = analyze_with_gemini(model, diagnosis_prompt, st.session_state.analysis_data['image_data'])
            else:
                result = analyze_with_gemini(model, diagnosis_prompt)
        
        # Display results
        st.markdown(result)
        
        # Store result for saving
        st.session_state.analysis_data['result'] = result
        st.session_state.analysis_data['timestamp'] = datetime.now().isoformat()
        
        # Save to vault button
        st.divider()
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button(get_text('save_vault'), type="primary"):
                category = 'General'
                if st.session_state.analysis_data.get('image'):
                    category = 'Radiology'
                elif 'Lab Report' in st.session_state.analysis_data['symptoms']:
                    category = 'Pathology'
                
                db.save_report(
                    st.session_state.user_id,
                    category,
                    st.session_state.analysis_data['symptoms'][:200],
                    result
                )
                st.success("✅ Saved to Health Vault!")
        
        with col2:
            if st.button("🔄 New Analysis"):
                st.session_state.conversation_state = 'initial'
                st.session_state.analysis_data = {}
                st.session_state.follow_up_count = 0
                st.rerun()

def health_vault_page():
    """Health vault page"""
    st.header(get_text('view_vault'))
    
    # Get user's reports
    reports = db.get_user_reports(st.session_state.user_id)
    
    if not reports:
        st.info("📭 No saved reports yet. Start by analyzing symptoms!")
        return
    
    # Category filter
    categories = list(set([r[2] for r in reports]))
    selected_category = st.selectbox("Filter by Category", ["All"] + categories)
    
    # Display reports
    filtered_reports = reports if selected_category == "All" else [r for r in reports if r[2] == selected_category]
    
    st.write(f"**Total Reports:** {len(filtered_reports)}")
    st.divider()
    
    for report in filtered_reports:
        report_id, user_id, category, symptoms, diagnosis, created_at = report
        
        with st.expander(f"📋 {category} - {created_at} - {symptoms[:50]}..."):
            st.markdown(f"**Category:** {category}")
            st.markdown(f"**Date:** {created_at}")
            st.markdown(f"**Symptoms:** {symptoms}")
            st.divider()
            st.markdown("**Analysis:**")
            st.markdown(diagnosis)
            
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("🗑️ Delete", key=f"delete_{report_id}"):
                    db.delete_report(report_id, st.session_state.user_id)
                    st.success("Report deleted!")
                    st.rerun()

# Main execution
if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()
