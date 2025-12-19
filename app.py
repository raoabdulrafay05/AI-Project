from typing import TypedDict, List, Optional, Literal
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import MetaData
from datetime import datetime
from dotenv import load_dotenv
import cv2
from langchain_core.embeddings import Embeddings
import supervision as sv
from ultralytics import YOLO
from flask import Response  # Ensure these are in your flask imports
import os
import base64
import numpy as np
import cv2
from deepface import DeepFace


# --- LIBRARIES FOR VECTOR SEARCH ---
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
# --- AI & LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
import warnings

# Load environment variables (API Keys)
load_dotenv()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=All, 1=Filter INFO, 2=Filter INFO & WARNING, 3=Filter All

# --- 2. SUPPRESS PYTHON WARNINGS ---
# This hides the LangChain and Pydantic warnings
warnings.filterwarnings('ignore')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__, template_folder="template")

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://postgres:1234@localhost:5432/Hospital_Management_System'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
metadata = MetaData()

# --- MASK DETECTION CONFIGURATION ---
try:
    # Load the YOLO model (Ensure 'mask.pt' is in your project folder)
    mask_model = YOLO('mask.pt') 
    print("Mask Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    mask_model = None

# Annotators for drawing boxes and labels
box_annotator = sv.BoxAnnotator(thickness=3)
label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=2)

# Global status dictionary to share data between video feed and UI
current_status = {
    "mask_detected": False,
    "confidence": 0.0,
    "status": "waiting"
}

CLASS_ID_MASK = 1    
CLASS_ID_NO_MASK = 0

# --- CRITICAL FIX: MANUALLY PUSH CONTEXT ---
# This line makes the app "active" so the models below can access the DB engine immediately.
app.app_context().push()

# --- DATABASE MODELS ---
# Now these will work globally because we pushed the context above
class Patient(db.Model):
    __table__ = db.Table('patients', metadata, autoload_with=db.engine)

class Department(db.Model):
    __table__ = db.Table('departments', metadata, autoload_with=db.engine)

class Doctor(db.Model):
    __table__ = db.Table('doctors', metadata, autoload_with=db.engine)

class Appointment(db.Model):
    __table__ = db.Table('appointments', metadata, autoload_with=db.engine)

class MedicalRecord(db.Model):
    __table__ = db.Table('medicalrecords', metadata, autoload_with=db.engine)

class Bill(db.Model):
    __table__ = db.Table('bills', metadata, autoload_with=db.engine)

class Test(db.Model):
    __table__ = db.Table('tests', metadata, autoload_with=db.engine)
    
class Pharmacy(db.Model):
    __table__ = db.Table('pharmacy', metadata, autoload_with=db.engine)


# ==========================================
# ============ AI CHATBOT LOGIC ============
# ==========================================

class AppointmentDetails(BaseModel):
    """Details required to schedule a medical appointment."""
    patient_id: int = Field(description="The unique integer ID of the patient")
    doctor_id: int = Field(description="The unique integer ID of the doctor")
    appointment_date: str = Field(description="Date of appointment in YYYY-MM-DD format")
    time_slot: str = Field(description="Time of appointment, e.g., '10:00 AM'")

# --- 1. UNIFIED STATE ---
class ChatState(TypedDict):
    messages: List[BaseMessage]
    intent: Optional[Literal["schedule", "doctor", "other","out_of_scope"]]
    
    # Scheduler Keys
    patient_id: Optional[int]
    doctor_id: Optional[int]
    appointment_date: Optional[str]
    time_slot: Optional[str]
    success: Optional[bool]
    
    # Doctor Search Keys
    recommended_doctors: Optional[str] 

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- NODE 1: MASTER ROUTER ---
# --- NODE 1: MASTER ROUTER (STRICTER) ---
# --- NODE 1: MASTER ROUTER (UPDATED) ---
def detect_master_intent(state: ChatState) -> ChatState:
    user_text = state["messages"][-1].content
    
    system_prompt = (
        "You are a medical assistant router. Classify the user request strictly:\n\n"
        "1. 'schedule': ONLY if the user provides concrete details like a Patient ID, Doctor ID, or Date.\n"
        "2. 'doctor': The user is describing symptoms to find a doctor.\n"
        "3. 'other': General hospital questions (e.g., 'Can you schedule?', 'Hi', 'Where are you located?').\n"
        "4. 'out_of_scope': Questions completely UNRELATED to medical/hospital topics "
        "(e.g., 'What is the weather?', 'Who is Messi?', 'Solve 2+2', 'Write Python code').\n\n"
        "Return ONLY one word: 'schedule', 'doctor', 'general', or 'out_of_scope'."
    )

    result = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ])
    
    state["intent"] = result.content.strip().lower()
    return state

# --- SECTION 6: OUT OF CONTEXT HANDLER ---
def handle_out_of_scope(state: ChatState) -> ChatState:
    # A static, polite refusal to keep the bot focused
    response = (
        "I am Diagnosify, a specialized hospital assistant. "
        "I cannot answer questions about general topics, weather, or other subjects. "
        "Please ask me about doctors, appointments, or medical services!"
    )
    state["messages"].append(AIMessage(content=response))
    return state

# --- SECTION 2: APPOINTMENT SCHEDULER (Structured) ---
def extract_appointment_details(state: ChatState) -> ChatState:
    user_text = state["messages"][-1].content
    structured_llm = llm.with_structured_output(AppointmentDetails)

    try:
        details = structured_llm.invoke(user_text)
        state["patient_id"] = details.patient_id
        state["doctor_id"] = details.doctor_id
        state["appointment_date"] = details.appointment_date
        state["time_slot"] = details.time_slot
    except Exception as e:
        print(f"Extraction Error: {e}")
        state["success"] = False

    return state

def schedule_appointment_node(state: ChatState) -> ChatState:
    # Check if we have data
    if not all([state.get("patient_id"), state.get("doctor_id"), state.get("appointment_date"), state.get("time_slot")]):
        state["messages"].append(AIMessage(content="I am missing some details. Please provide Patient ID, Doctor ID, Date (YYYY-MM-DD), and Time."))
        state["success"] = False
        return state

    # --- ROBUST TIME PARSING START ---
    time_obj = None
    raw_time = state["time_slot"].strip().upper()  # specific cleanup
    
    # List of formats to try: 24hr, 12hr with space, 12hr without space, etc.
    time_formats = ["%H:%M", "%I:%M %p", "%I:%M%p", "%I %p", "%H:%M:%S"]
    
    for fmt in time_formats:
        try:
            time_obj = datetime.strptime(raw_time, fmt).time()
            break  # If successful, stop trying other formats
        except ValueError:
            continue # Try next format
            
    if not time_obj:
        state["messages"].append(AIMessage(content=f"I couldn't understand the time '{state['time_slot']}'. Please try standard format like '20:00' or '8:00 PM'."))
        state["success"] = False
        return state
    # --- ROBUST TIME PARSING END ---

    # Access DB within App Context
    with app.app_context():
        try:
            if not Patient.query.get(state["patient_id"]):
                state["messages"].append(AIMessage(content=f"Patient ID {state['patient_id']} not found."))
                return state
            
            if not Doctor.query.get(state["doctor_id"]):
                state["messages"].append(AIMessage(content=f"Doctor ID {state['doctor_id']} not found."))
                return state

            new_appt = Appointment(
                patientid=state["patient_id"],
                doctorid=state["doctor_id"],
                appointmentdate=datetime.strptime(state["appointment_date"], '%Y-%m-%d').date(),
                timeslot=time_obj,  # Use the safely parsed time object
                status="Scheduled"
            )
            db.session.add(new_appt)
            db.session.commit()
            state["messages"].append(AIMessage(content="Appointment successfully scheduled!"))
            state["success"] = True
            
        except Exception as e:
            db.session.rollback()
            state["messages"].append(AIMessage(content=f"Database Error: {e}"))
            state["success"] = False
            
    return state

# --- SECTION 3: INTELLIGENT DOCTOR MATCHING ---
def recommend_doctor(state: ChatState) -> ChatState:
    user_text = state["messages"][-1].content

    with app.app_context():
        all_doctors = Doctor.query.all()
        doctors_list_str = "\n".join([
            f"- ID: {d.doctorid}, Name: {d.fullname}, Specialization: {d.specialization}, Fee: {d.doctorfee}"
            for d in all_doctors
        ])

    system_prompt = (
        "You are a medical receptionist. Match the user's symptoms to the BEST doctor(s) from the list below.\n"
        f"--- DOCTORS ---\n{doctors_list_str}\n"
        "----------------\n"
        "Reply directly to the user with the doctor's Name, ID, and why they are a good match."
    )

    result = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ])

    state["messages"].append(AIMessage(content=result.content))
    return state

# --- SECTION 4: GENERAL NODE ---
# --- SECTION 4: GENERAL NODE (Must be smart to answer your question) ---
def non_doctor_response(state: ChatState) -> ChatState:
    user_text = state["messages"][-1].content
    
    system_prompt = (
        "You are a helpful medical assistant. The user is asking a general question.\n"
        "If they ask 'Can you schedule?' or 'What details?', reply:\n"
        "'Yes, I can schedule appointments. Please provide: Patient ID, Doctor ID, Date (YYYY-MM-DD), and Time.'\n"
        "Otherwise, answer their question helpfully."
        "If the question is inappropriate or out of context of medical assistance, just reply i cant help you with that"
        "If questions is not in medical perspective, reply directly i cant help with that"
    )

    result = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ])

    state["messages"].append(result)
    return state
# --- SECTION 5: ROUTING & GRAPH ---
def route_by_intent(state: ChatState) -> str:
    intent = state.get("intent")
    if intent == "schedule": return "schedule_flow"
    elif intent == "doctor": return "doctor_flow"
    elif intent == "out_of_scope": return "out_of_scope_flow" # <--- Added this
    else: return "general_flow" # Changed "other" to "general" to match router

# Build Graph
# Build Graph
workflow = StateGraph(ChatState)

# Add Nodes
workflow.add_node("detect_master_intent", detect_master_intent)
workflow.add_node("extract_details", extract_appointment_details)
workflow.add_node("schedule_appointment", schedule_appointment_node)
workflow.add_node("recommend_doctor", recommend_doctor)
workflow.add_node("non_doctor", non_doctor_response)
workflow.add_node("out_of_scope_node", handle_out_of_scope) # <--- Added Node

# Set Entry
workflow.set_entry_point("detect_master_intent")

# Add Conditional Routing
workflow.add_conditional_edges(
    "detect_master_intent",
    route_by_intent,
    {
        "schedule_flow": "extract_details",
        "doctor_flow": "recommend_doctor",
        "general_flow": "non_doctor",
        "out_of_scope_flow": "out_of_scope_node", # <--- Added Route
    },
)

# Connect Edges
workflow.add_edge("extract_details", "schedule_appointment")
workflow.add_edge("schedule_appointment", END)
workflow.add_edge("recommend_doctor", END)
workflow.add_edge("non_doctor", END)
workflow.add_edge("out_of_scope_node", END) # <--- Added Edge

medical_chatbot = workflow.compile()


# ==========================================
# ============ FLASK ROUTES ================
# ==========================================

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.get_json()
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Run the AI Workflow
        initial_state = {"messages": [HumanMessage(content=user_message)]}
        result = medical_chatbot.invoke(initial_state)
        
        # Get Bot Response
        bot_response = result["messages"][-1].content
        
        return jsonify({"response": bot_response})

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"response": "Sorry, I encountered an internal error."}), 500


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/patient_management')
def patient_management():
    return render_template('patient_management.html')

# Doctor Management
@app.route('/doctor_management')
def doctor_management():
    return render_template('doctor_management.html')

# Appointment Scheduling
@app.route('/appointment_scheduling')
def appointment_scheduling():
    return render_template('appointment_scheduling.html')

# Medical Records Management
@app.route('/medical_records_management')
def medical_records_management():
    return render_template('medical_records_management.html')

# Pharmacy Management
@app.route('/pharmacy_management')
def pharmacy_management():
    return render_template('pharmacy.html')

# Billing
@app.route('/billing')
def billing():
    return render_template('billing.html')

# Lab Tests
@app.route('/lab_tests')
def lab_tests():
    return render_template('lab_tests.html')

# Route to add a patient
@app.route('/add_patient', methods=['POST'])
def add_patient():
    full_name = request.form['FullName']
    dob = request.form['DOB']
    gender = request.form['Gender']
    contactno = request.form['ContactNo']
    address = request.form['Address']
    bloodgroup = request.form['BloodGroup']
    medicalhistory = request.form['MedicalHistory']
    
    # Create new patient record
    new_patient = Patient(
        fullname=full_name,
        dob=dob,
        gender=gender,
        contactno=contactno,
        address=address,
        bloodgroup=bloodgroup,
        medicalhistory=medicalhistory
    )
    
    db.session.add(new_patient)
    db.session.commit()
    return redirect(url_for('patient_management'))

# Route to view all patients
@app.route('/view_all_patients')
def view_all_patients():
    patients = Patient.query.all()  # Fetching all patients from the database
    print(patients)  # Logs to the console to check the data being fetched
    return render_template('view_all_patients.html', patients=patients)

def get_patient_by_id(patient_id):
    return Patient.query.get(patient_id)

# Route to view specific patient details
@app.route('/view_patient_details', methods=['GET'])
def view_patient_details():
    patient_id = request.args.get('patient_id')  # Get Patient ID from the query string
    if patient_id:
        patient = get_patient_by_id(patient_id)  # Replace with your actual database query
        if patient:
            return render_template('view_patient_details.html', patient=patient)
        else:
            return "Patient not found", 404
    return render_template('patient_id_form.html')

@app.route('/edit_patient', methods=['GET', 'POST'])
def edit_patient():
    if request.method == 'POST':
        patient_id = request.form.get('patient_id')
        print("Received patient_id:", patient_id)

        if not patient_id or not patient_id.isdigit():
            return "Invalid Patient ID", 400

        patient = Patient.query.get(patient_id)

        if not patient:
            return "Patient not found", 404

        # Update fields
        patient.fullname = request.form['FullName']
        patient.dob = request.form['DOB']
        patient.gender = request.form['Gender']
        patient.contactno = request.form['ContactNo']
        patient.address = request.form['Address']
        patient.bloodgroup = request.form['BloodGroup']
        patient.medicalhistory = request.form['MedicalHistory']

        db.session.commit()

        return redirect(url_for('patient_management'))

    # On GET, just show the form
    return render_template('edit_patient.html')


# Route to delete patient
@app.route('/delete_patient', methods=['GET', 'POST'])
def delete_patient():
    if request.method == 'POST':
        # Get patient_id from the form
        patient_id = request.form['patient_id']
        
        # Get the patient object from the database, or return a 404 if not found
        patient = Patient.query.get_or_404(patient_id)
        
        # Delete the patient from the database
        db.session.delete(patient)
        db.session.commit()

        # Redirect to the 'view_all_patients' route (ensure this route exists)
        return redirect(url_for('view_all_patients'))  # Redirect to the patient list

    # If GET request, display the form
    return render_template('patient_delete_form.html')


@app.route('/add_doctor', methods=['POST'])
def add_doctor():
    fullname = request.form['fullname']
    specialization = request.form['specialization']
    phone_no = request.form['phoneno']
    email = request.form['Email']
    department_id = request.form['departmentid']
    doctor_fee = request.form['DoctorFee']
    salary = request.form['Salary']
    
    # Create new doctor record
    new_doctor = Doctor(
        fullname=fullname,
        specialization=specialization,
        phoneno=phone_no,
        email=email,
        departmentid=department_id,
        doctorfee=doctor_fee,
        doctorsalary=salary
    )
    
    db.session.add(new_doctor)
    db.session.commit()
    return redirect(url_for('doctor_management'))

@app.route('/view_all_doctors')
def view_all_doctors():
    # Fetching all doctors from the database
    doctors = Doctor.query.all()  # Assuming Doctor is your model for doctors
    
    # Render the HTML template with the list of doctors
    return render_template('view_all_doctor.html', doctors=doctors)


def get_doctor_by_id(doctor_id):
    return Doctor.query.get(doctor_id)
     

@app.route('/view_doctor_details', methods=['GET'])
def view_doctor_details():
    doctor_id = request.args.get('doctor_id')  # Get Doctor ID from the query string
    
    if doctor_id:
        # Replace with actual logic to query the doctor from your database
        doctor = get_doctor_by_id(doctor_id)  # Example function to fetch doctor details by ID
        
        if doctor:
            return render_template('view_doctor_details.html', doctor=doctor)
        else:
            return "Doctor not found", 404  # Handle the case where the doctor is not found
    
    # If no doctor_id is passed, render a form to enter the doctor_id
    return render_template('doctor_view_form.html')

@app.route('/edit_doctor', methods=['GET', 'POST'])
def edit_doctor():
    if request.method == 'POST':
        doctor_id = request.form.get('doctor_id')  # Get the doctor ID from the form
        print("Received doctor_id:", doctor_id)

        if not doctor_id or not doctor_id.isdigit():  # Check for valid doctor ID
            return "Invalid Doctor ID", 400

        doctor = Doctor.query.get(doctor_id)  # Fetch the doctor from the database

        if not doctor:  # Check if doctor exists
            return "Doctor not found", 404

        # Update doctor fields with new data
        doctor.fullname = request.form['fullname']
        doctor.specialization = request.form['specialization']
        doctor.phoneno = request.form['phoneno']
        doctor.email = request.form['email']
        doctor.doctorfee = request.form['doctorfee']
        doctor.doctorsalary = request.form['doctorsalary']
        doctor.departmentid = request.form['departmentid']

        db.session.commit()  # Commit the changes to the database

        return redirect(url_for('view_all_doctors'))
    return render_template('edit_doctor.html')

@app.route('/delete_doctor', methods=['GET', 'POST'])
def delete_doctor():
    if request.method == 'POST':
        doctor_id = request.form['doctor_id']
        
        # Get the doctor object from the database
        doctor = Doctor.query.get(doctor_id)

        if not doctor:
            # If doctor not found, return a 404 error or message
            return "Doctor not found", 404

        try:
            # Delete the doctor from the database
            db.session.delete(doctor)
            db.session.commit()

            # Redirect to the doctor list
            return redirect(url_for('view_all_doctors'))  # Make sure this route exists

        except Exception as e:
            db.session.rollback()  # Rollback in case of an error
            return f"Error occurred while deleting doctor: {str(e)}", 500  # Display error message

    return render_template('doctor_delete_form.html')

@app.route('/add_appointment', methods=['POST'])
def add_appointment():
    patient_id = request.form.get('patientid')
    doctor_id = request.form.get('doctorid')
    date = request.form.get('appointmentdate')
    time = request.form.get('timeslot')
    
    if not all([patient_id, doctor_id, date, time]):
        return jsonify({"error": "Missing form data"}), 400
    
    # Check if the patient exists
    patient = Patient.query.get(patient_id)
    if not patient:
        return render_template('app_patient_not_found.html')

    # Check if the doctor exists
    doctor = Doctor.query.get(doctor_id)
    if not doctor:
        return render_template('app_doctor_not_found.html')  # You can create a page for doctor not found

    # Create the new appointment
    new_appointment = Appointment(
        patientid=patient_id,
        doctorid=doctor_id,
        appointmentdate=datetime.strptime(date, '%Y-%m-%d').date(),
        timeslot=datetime.strptime(time, '%H:%M').time(),
        status='scheduled'
    )

    db.session.add(new_appointment)
    db.session.commit()

    return redirect(url_for('appointment_scheduling'))


@app.route('/view_appointments', methods = ['GET'])
def view_appointments():
    Appointments = Appointment.query.all()
    
    return render_template('view_appointments.html', appointments = Appointments)

@app.route('/edit_appointment', methods=['GET', 'POST'])
def edit_appointment():
    if request.method == 'POST':
        # Get the appointment ID from the form
        appointment_id = request.form.get('appointment_id')
        print("Received appointment_id:", appointment_id)

        if not appointment_id or not appointment_id.isdigit():
            return "Invalid Appointment ID", 400

        # Fetch the appointment from the database using the ID
        appointment = Appointment.query.get(appointment_id)

        if not appointment:
            return "Appointment not found", 404

        # Update fields with the new data from the form
        appointment.patientid = request.form['patient_id']
        appointment.doctorid = request.form['doctor_id']
        appointment.appointmentdate = datetime.strptime(request.form['appointment_date'], '%Y-%m-%d')
        appointment.timeslot = datetime.strptime(request.form['appointment_time'], '%H:%M').time()
        appointment.status = request.form['status']

        # Commit the changes to the database
        db.session.commit()

        # Redirect to the appointments view page
        return redirect(url_for('view_appointments'))

    return render_template('edit_appointment.html')

def get_appointment_by_id(appointment_id):
    return Appointment.query.get(appointment_id)
     

@app.route('/view_appointment_details', methods=['GET'])
def view_appointment_details():
    appointment_id = request.args.get('appointment_id')  # Get Doctor ID from the query string
    
    if appointment_id:
        # Replace with actual logic to query the doctor from your database
        appointment = get_appointment_by_id(appointment_id)  # Example function to fetch doctor details by ID
        
        if appointment:
            return render_template('view_an_appointment.html', appointment=appointment)
        else:
            return "Appointment not found", 404  # Handle the case where the doctor is not found
    
    # If no doctor_id is passed, render a form to enter the doctor_id
    return render_template('appointment_id_form.html')

@app.route('/delete_appointment', methods=['GET', 'POST'])
def delete_appointment():
    if request.method == 'POST':
        appointment_id = request.form['appointment_id']
        
        # Get the doctor object from the database
        appointment = get_appointment_by_id(appointment_id)

        if not appointment:
            # If doctor not found, return a 404 error or message
            return "Appointment not found", 404

        try:
            # Delete the doctor from the database
            db.session.delete(appointment)
            db.session.commit()
            print("DELETED")
            # Redirect to the doctor list
            return   redirect(url_for('view_appointments'))
        
        except Exception as e:
            db.session.rollback()  # Rollback in case of an error
            return f"Error occurred while deleting appointment: {str(e)}", 500  # Display error message

    return render_template('appointment_delete_id_form.html')

@app.route('/add_medical_record', methods=['POST'])
def add_add_medical_record():
    patient_id = request.form['patient_id']
    doctor_id = request.form['doctor_id']
    diagnosis = request.form['diagnosis']
    prescription = request.form['prescription']
    date = request.form['date']
    
    
    # Create new patient record
    new_MedicalRecord = MedicalRecord(
        patientid=patient_id,
        doctorid=doctor_id,
        diagnosis=diagnosis,
        prescription=prescription,
        date=date,
    )
    
    db.session.add(new_MedicalRecord)
    db.session.commit()
    return redirect(url_for('medical_records_management'))

@app.route('/view_all_medical_records', methods = ['GET'])
def view_all_medical_records():
    MedicalRecords = MedicalRecord.query.all()
    
    return render_template('view_medical_records.html', MedicalRecords = MedicalRecords)

@app.route('/enter_medical_record_id', methods=['GET', 'POST'])
def enter_medical_record_id():
    if request.method == 'POST':
        # Get the medical record ID from the form
        medical_record_id = request.form.get('medical_record_id')
        print("Received medical_record_id:", medical_record_id)

        # Validate the medical record ID
        if not medical_record_id or not medical_record_id.isdigit():
            return "Invalid Medical Record ID", 400

        # Fetch the medical record from the database
        medical_record = MedicalRecord.query.get(medical_record_id)

        if not medical_record:
            return "Medical Record not found", 404

        # Redirect to the second step for updating the record
        return redirect(url_for('edit_medical_record', medical_record_id=medical_record_id))

    # Render the form to enter the medical record ID
    return render_template('edit_medical_record.html')

@app.route('/edit_medical_record/<int:medical_record_id>', methods=['GET','POST'])
def edit_medical_record(medical_record_id):
    medical_record = MedicalRecord.query.get(medical_record_id)
    print(medical_record_id)

    if not medical_record:
        return "Medical Record not found", 404

    if request.method == 'POST':
        print("nfjnfvjnvjn ifvunfuinefviunvuifnvfuinvfuinnevfiu")
        # Update the fields of the medical record with new data
        medical_record.patient_id = request.form['PatientID']
        medical_record.doctor_id = request.form['DoctorID']
        medical_record.diagnosis = request.form['Diagnosis']
        medical_record.prescription = request.form['Prescription']
        medical_record.date = request.form['Date']

        # Commit the changes to the database
        db.session.commit()

        return redirect(url_for('medical_records_management'))

    # Render the form with the existing medical record's details
    return render_template(
    'edit_medical_record_form.html',
    medical_record=medical_record,
    medical_record_id=medical_record_id
    )
    
@app.route('/view_medical_record_details', methods=['GET'])
def view_medical_record_details():
    MedicalRecordID = request.args.get('medical_record_id')  # Get Doctor ID from the query string
    
    if MedicalRecordID:
        # Replace with actual logic to query the doctor from your database
        medical_record = MedicalRecord.query.get(MedicalRecordID)
        
        if medical_record:
            return render_template('view_an_medical_record.html', medical_record=medical_record)
        else:
            return "Medical Record not found", 404  # Handle the case where the doctor is not found
    
    # If no doctor_id is passed, render a form to enter the doctor_id
    return render_template('view_an_medical_record_form.html')

@app.route('/delete_medical_record', methods=['GET', 'POST'])
def delete_medical_record():
    if request.method == 'POST':
        medical_record_id = request.form['medical_record_id']
        
        # Get the doctor object from the database
        record = MedicalRecord.query.get(medical_record_id)

        if not record:
            # If doctor not found, return a 404 error or message
            return "Medical Record not found", 404

        try:
            # Delete the doctor from the database
            db.session.delete(record)
            db.session.commit()
            # Redirect to the doctor list
            return   redirect(url_for('view_all_medical_records'))
        
        except Exception as e:
            db.session.rollback()  # Rollback in case of an error
            return f"Error occurred while deleting Medical Records: {str(e)}", 500  # Display error message

    return render_template('medical_record_delete_form.html')

###
@app.route('/add_medicine', methods=['POST'])
def add_medicine():
    medicine_name = request.form['medicine_name']
    quantity = request.form['quantity']
    price_per_unit = request.form['price_per_unit']
    patientid = request.form['patient_id']

    
    # Create new patient record
    new_medicine = Pharmacy(
        medicinename=medicine_name,
        quantity=quantity,
        priceperunit=price_per_unit,
        patientid=patientid,
    )
    
    db.session.add(new_medicine)
    db.session.commit()
    return redirect(url_for('pharmacy_management'))

###

@app.route('/add_lab_test', methods=['POST'])
def add_lab_test():
    try:
        patient_id = request.form.get('patient_id')
        testtype = request.form.get('test_type')
        testresult = request.form.get('test_result')
        testdate = request.form.get('test_date')
        testprice = request.form.get('price')
        
        if not all([patient_id, testtype, testprice]):
            return jsonify({"error": "Missing form data"}), 400
        
        new_test = Test(
            patientid=patient_id,
            testdate=datetime.strptime(testdate, '%Y-%m-%d').date(),
            testtype=testtype,
            testresult = testresult,
            testprice = testprice
        )
        db.session.add(new_test)
        db.session.commit()
        return redirect(url_for('lab_tests'))
    except Exception as e:
        db.session.rollback()
        # return render_template('app_patient_not_found.html')
        return redirect(url_for('lab_tests'))
    
    
@app.route('/view_all_lab_tests', methods = ['GET'])
def view_all_lab_tests():
    tests = Test.query.all()
    
    return render_template('view_all_test.html', tests = tests)

@app.route('/enter_test_id', methods=['GET', 'POST'])
def enter_test_id():
    if request.method == 'POST':
        # Get the test ID from the form
        test_id = request.form.get('test_id')
        print("Received test_id:", test_id)

        # Validate the test ID
        if not test_id or not test_id.isdigit():
            return "Invalid Test Record ID", 400

        # Fetch the test record from the database
        test = Test.query.get(test_id)

        if not test:
            return "Test Record not found", 404

        # Redirect to the edit page for updating the record
        return redirect(url_for('edit_test_record', test_record_id=test_id))

    # Render the form to enter the test ID
    return render_template('edit_test_form.html')

@app.route('/edit_test_record/<int:test_record_id>', methods=['GET', 'POST'])
def edit_test_record(test_record_id):
    # Fetch the test record by test_record_id
    test_record = Test.query.get(test_record_id)

    if not test_record:
        return "Test Record not found", 404

    if request.method == 'POST':
        # Update the fields of the test record with new data from the form
        test_record.patientid = request.form['patientid']
        test_record.testtype = request.form['testtype']
        test_record.testresult = request.form['testresult']
        test_record.testdate = request.form['testdate']
        test_record.testprice = request.form['testprice']

        # Commit the changes to the database
        db.session.commit()

        # Redirect to view all lab tests after updating
        return redirect(url_for('view_all_lab_tests'))

    # Render the form to edit the test record
    return render_template('edit_test_record.html', test_record_id=test_record_id)

@app.route('/search_lab_test', methods=['GET'])
def search_lab_test():
    testid = request.args.get('test_id')  # Get Doctor ID from the query string
    
    if testid:
        # Replace with actual logic to query the doctor from your database
        test_record = Test.query.get(testid)
        
        if test_record:
            return render_template('view_an_test_record.html', test_record=test_record)
        else:
            return "Test Record not found", 404  # Handle the case where the doctor is not found
    
    # If no doctor_id is passed, render a form to enter the doctor_id
    return render_template('view_an_test_record_form.html')

@app.route('/delete_lab_test', methods=['GET', 'POST'])
def delete_lab_test():
    if request.method == 'POST':
        testid = request.form['test_id']
        
        # Get the doctor object from the database
        test = Test.query.get(testid)

        if not test:
            # If doctor not found, return a 404 error or message
            return "Test Record not found", 404

        try:
            # Delete the doctor from the database
            db.session.delete(test)
            db.session.commit()
            # Redirect to the doctor list
            return   redirect(url_for('view_all_lab_tests'))
        
        except Exception as e:
            db.session.rollback()  # Rollback in case of an error
            return f"Error occurred while deleting Test Records: {str(e)}", 500  # Display error message

    return render_template('lab_record_delete_form.html')


def calculate_patient_bill(appointment_id, pharmacy_id, test_id):
    # Fetch the doctor's fee via appointment → doctor
    doctor_fee = db.session.query(Doctor.doctorfee).join(Appointment, Doctor.doctorid == Appointment.doctorid)\
        .filter(Appointment.appointmentid == appointment_id).scalar() or 0

    # Fetch the medicine fee using the pharmacy ID
    medicine_fee = db.session.query(
        db.func.sum(Pharmacy.quantity * Pharmacy.priceperunit)
    ).filter(Pharmacy.pharmacy_id == pharmacy_id).scalar() or 0

    # Fetch the lab test fee using the test ID
    lab_fee = db.session.query(Test.testprice)\
        .filter(Test.testid == test_id).scalar() or 0

    print(doctor_fee)
    print(lab_fee)
    print(medicine_fee)

    # Calculate the total bill
    total_amount = doctor_fee + medicine_fee + lab_fee
    print(total_amount)
    return total_amount



@app.route('/add_bill', methods=['POST'])
def add_bill():
    # Get data from a POSTed HTML form (not URL and not JSON)
    appointment_id = request.form.get('appointment_id')
    pharmacy_id = request.form.get('pharmacy_id')
    test_id = request.form.get('lab_test_id')
    patient_id = request.form.get('patient_id')
    bill_date = request.form.get('bill_date')

    # Validate required fields
    if not all([appointment_id, pharmacy_id, test_id, patient_id, bill_date]):
        return "Missing required fields", 400

    # Calculate the total bill
    total = calculate_patient_bill(appointment_id, pharmacy_id, test_id)
    print(total)
    # Create and save the bill
    new_bill = Bill(
        patientid=patient_id,
        billdate=bill_date,
        paymentstatus="unpaid",
        expenses=total
    )

    db.session.add(new_bill)
    db.session.commit()

    # Silent success (or change this line as needed)
    return redirect(url_for('billing'))

@app.route('/view_pharmcay', methods = ['GET'])
def view_pharmcay():
    medicines_history = Pharmacy.query.all()
    
    return render_template('view_all_medicine_details.html', medicines_history = medicines_history)

@app.route('/view_all_bills', methods = ['GET'])
def view_all_bills():
    bill = Bill.query.all()
    
    return render_template('view_all_bills.html', bills = bill)


@app.route('/enter_bill_id', methods=['GET', 'POST'])
def enter_bill_id():
    if request.method == 'POST':
        bill_id = request.form.get('bill_id')
        if not bill_id or not bill_id.isdigit():
            return "Invalid Bill ID", 400

        bill = Bill.query.get(int(bill_id))
        if not bill:
            return "Bill not found", 404

        return redirect(url_for('edit_bill', bill_id=bill_id))

    return render_template('enter_bill_id.html')

@app.route('/edit_bill/<int:bill_id>', methods=['GET', 'POST'])
def edit_bill(bill_id):
    bill = Bill.query.get(bill_id)
    if not bill:
        return "Bill not found", 404

    if request.method == 'POST':
        bill.patientid = request.form['patientid']
        appointment_id = request.form['appointmentid']
        pharmacy_id = request.form['pharmacyid']
        test_id = request.form['lab_test_id']

        # Recalculate the total bill
        bill.expenses = calculate_patient_bill(appointment_id, pharmacy_id, test_id)
        bill.appointmentid = appointment_id
        bill.pharmacyid = pharmacy_id
        bill.lab_test_id = test_id
        bill.paymentstatus = request.form['paymentstatus']
        bill.billdate = request.form['billdate']

        db.session.commit()
        return redirect(url_for('view_all_bills'))

    return render_template('edit_bill.html', bill=bill)

@app.route('/find_bill', methods=['GET'])
def find_bill():
    bill_id = request.args.get('bill_id')  # Get Bill ID from the query string
    
    if bill_id:
        # Replace with actual logic to query the bill from your database
        bill_record = Bill.query.get(bill_id)
        
        if bill_record:
            return render_template('view_a_bill_record.html', bill_record=bill_record)
        else:
            return "Bill Record not found", 404  # Handle the case where the bill is not found
    
    # If no bill_id is passed, render a form to enter the bill_id
    return render_template('view_a_bill_record_form.html')

@app.route('/delete_bill', methods=['GET', 'POST'])
def delete_bill():
    if request.method == 'POST':
        bill_id = request.form['bill_id']
        
        # Fetch bill from database
        bill = Bill.query.get(bill_id)

        if not bill:
            return "Bill Record not found", 404

        try:
            db.session.delete(bill)
            db.session.commit()
            return redirect(url_for('view_all_bills'))  # Update with your actual view
        except Exception as e:
            db.session.rollback()
            return f"Error occurred while deleting Bill Record: {str(e)}", 500

    return render_template('bill_record_delete_form.html')


# --- MASK DETECTION LOGIC ---

@app.route('/detect_mask_feed', methods=['POST'])
def detect_mask_feed():
    try:
        # 1. Get the image from the POST request (sent as Base64 string)
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image data"}), 400

        image_data = data['image']
        
        # 2. Decode Base64 string to OpenCV Image
        # Remove the header "data:image/jpeg;base64," if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        decoded_data = base64.b64decode(image_data)
        np_data = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        # 3. Run YOLO Inference (Your existing logic)
        if mask_model:
            results = mask_model(frame, verbose=False)[0]
            # Use basic YOLO results to avoid dependency complexity for now
            # or continue using sv.Detections if you prefer.
            
            detections = []
            
            # Extract boxes and classes manually to send back as JSON
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label_name = mask_model.names[class_id]
                
                # Check your specific class IDs from your training
                is_mask = (class_id == CLASS_ID_MASK) 

                detections.append({
                    "x1": int(x1), "y1": int(y1), 
                    "x2": int(x2), "y2": int(y2),
                    "label": label_name,
                    "confidence": round(confidence * 100, 1),
                    "mask_detected": is_mask
                })

            # Update your global status if needed (optional)
            # global current_status ... (logic here)

            return jsonify({
                "success": True,
                "detections": detections
            })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# Keep your route to serve the HTML page
@app.route('/mask_detection')
def mask_detection():
    return render_template('mask.html')

@app.route('/status')
def get_status():
    # Returns the JSON status for the JavaScript to read
    return jsonify(current_status)


@app.route('/cv')
def cv():
    # Renders the HTML page you created earlier
    return render_template('router.html')

@app.route('/scanner')
def scanner():
    # Renders the HTML page you created earlier
    return render_template('scanner.html')

# A dummy class to satisfy LangChain's requirements
class IdentityEmbeddings(Embeddings):
    def __init__(self, dim=128):
        self.dim = dim

    def embed_documents(self, texts):
        # Returns a list of zero-vectors
        return [np.zeros(self.dim).tolist() for _ in texts]

    def embed_query(self, text):
        # Returns a single zero-vector
        return np.zeros(self.dim).tolist()

class FaceVectorDB:
    def __init__(self, db_path="face_db_cosine_index"):
        self.db_path = db_path
        self.model_name = "SFace" # Fast model for CPU
        self.embedding_dim = 128  # SFace usually outputs 128 dims
        self.threshold = 0.50     # Cosine Similarity Threshold
        self.vector_store = None
        self._load_or_create_db()

    # def _dummy_embed_fn(self, text):
    #     return np.zeros(self.embedding_dim)

    def _load_or_create_db(self):
        # 1. Try to load existing database
        self.embedding_model = IdentityEmbeddings(dim=self.embedding_dim)
        if os.path.exists(self.db_path):
            print("Loading database...")
            try:
                self.vector_store = FAISS.load_local(
                    self.db_path, 
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading DB: {e}. Creating new one.")
                self._create_new_db()
        else:
            self._create_new_db()

    def _create_new_db(self):
        print("Creating new database...")
        self.embedding_model = IdentityEmbeddings(dim=self.embedding_dim)
        index = faiss.IndexFlatIP(self.embedding_dim)
        self.vector_store = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    # Normalize vector so Dot Product becomes Cosine Similarity
    def _normalize_vector(self, vector):
        vector = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0: return vector
        return (vector / norm).tolist()

    def generate_embedding(self, image_frame):
        try:
            # DeepFace detects face and returns vector
            results = DeepFace.represent(
                img_path=image_frame, 
                model_name=self.model_name, 
                enforce_detection=True,
                detector_backend="opencv"
            )
            return self._normalize_vector(results[0]["embedding"])
        except:
            return None # No face detected

    # --- SAVE USER ---
    def register_user(self, image_frame, name, doc_id="AUTO"):
        embedding = self.generate_embedding(image_frame)
        if embedding:
            # We only really care about the Name now, but we store ID for technical reasons
            metadata = {"name": name, "docId": doc_id}
            
            self.vector_store.add_embeddings(
                text_embeddings=[(name, embedding)], 
                metadatas=[metadata]
            )
            self.vector_store.save_local(self.db_path)
            return True, f"Registered: {name}"
        else:
            return False, "No face detected in camera."

    # --- IDENTIFY USER ---
    def recognize_user(self, image_frame):
        target_embedding = self.generate_embedding(image_frame)
        if not target_embedding: 
            return {"match": False, "status": "No Face Detected"}

        # Search for closest match
        results = self.vector_store.similarity_search_with_score_by_vector(
            target_embedding, k=1
        )
        
        if not results: 
            return {"match": False, "status": "Unknown"}

        best_match_doc, score = results[0]
        print(f"Match Score: {score}") # Debugging

        if score > self.threshold:
            # Return the name found in database
            found_name = best_match_doc.metadata.get('name', 'Unknown')
            return {"match": True, "name": found_name}
        else:
            return {"match": False, "status": "Unknown Person"}

# Initialize Logic
face_db = FaceVectorDB()


def decode_image(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    try:
        img = decode_image(data['image'])

        success, msg = face_db.register_user(img, data['name'])
        
        return jsonify({'success': success, 'message': msg})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'message': "Server Error"})

@app.route('/api/scan', methods=['POST'])
def scan():
    data = request.json
    try:
        img = decode_image(data['image'])
        
        # Run recognition
        result = face_db.recognize_user(img)
        
        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'match': False, 'status': "Server Error"})
       
if __name__ == '__main__':
    app.run(debug=True)