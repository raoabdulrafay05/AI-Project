from typing import TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

from app import app, db, Doctor, Appointment, Patient

# --- 0. DEFINE STRUCTURED OUTPUT MODEL ---
class AppointmentDetails(BaseModel):
    """Details required to schedule a medical appointment."""
    patient_id: int = Field(description="The unique integer ID of the patient")
    doctor_id: int = Field(description="The unique integer ID of the doctor")
    appointment_date: str = Field(description="Date of appointment in YYYY-MM-DD format")
    time_slot: str = Field(description="Time of appointment, e.g., '10:00 AM'")

# --- 1. UNIFIED STATE ---
class ChatState(TypedDict):
    messages: List[BaseMessage]
    intent: Optional[Literal["schedule", "doctor", "other"]]
    
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

def detect_master_intent(state: ChatState) -> ChatState:
    user_text = state["messages"][-1].content
    
    system_prompt = (
        "You are a medical assistant. Classify the user request into one of these categories:\n"
        "1. 'schedule': The user wants to book, make, or schedule an appointment.\n"
        "2. 'doctor': The user is describing symptoms or asking for a doctor.\n"
        "3. 'other': Any other request (greetings, unrelated questions).\n\n"
        "Return ONLY one word: 'schedule', 'doctor', or 'other'."
    )

    result = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ])
    
    state["intent"] = result.content.strip().lower()
    return state

# --- SECTION 2: APPOINTMENT SCHEDULER (Structured) ---

def extract_appointment_details(state: ChatState) -> ChatState:
    user_text = state["messages"][-1].content

    # Create a structured LLM specifically for extraction
    structured_llm = llm.with_structured_output(AppointmentDetails)

    try:
        # The LLM will now return an AppointmentDetails object directly
        details = structured_llm.invoke(user_text)
        
        # Update state directly from the object attributes
        state["patient_id"] = details.patient_id
        state["doctor_id"] = details.doctor_id
        state["appointment_date"] = details.appointment_date
        state["time_slot"] = details.time_slot
        
    except Exception as e:
        # If extraction fails (e.g., user didn't provide enough info), handle gracefully
        print(f"Extraction Error: {e}")
        state["success"] = False

    return state

def schedule_appointment(state: ChatState) -> ChatState:
    # Check if all fields were successfully extracted
    if not all([state.get("patient_id"), state.get("doctor_id"), state.get("appointment_date"), state.get("time_slot")]):
        state["messages"].append(AIMessage(content="I am missing some details. Please provide Patient ID, Doctor ID, Date (YYYY-MM-DD), and Time."))
        state["success"] = False
        return state

    with app.app_context():
        try:
            # Verify existence (Optional but recommended)
            if not Patient.query.get(state["patient_id"]):
                state["messages"].append(AIMessage(content=f"Patient ID {state['patient_id']} not found."))
                return state
            
            if not Doctor.query.get(state["doctor_id"]):
                state["messages"].append(AIMessage(content=f"Doctor ID {state['doctor_id']} not found."))
                return state

            # Create Appointment
            new_appt = Appointment(
                patientid=state["patient_id"],
                doctorid=state["doctor_id"],
                appointmentdate=datetime.strptime(state["appointment_date"], '%Y-%m-%d').date(),
                timeslot=state["time_slot"],
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

def non_doctor_response(state: ChatState) -> ChatState:
    state["messages"].append(AIMessage(content="I can help you schedule appointments or find doctors based on symptoms."))
    return state

# --- SECTION 5: ROUTING & GRAPH ---

def route_by_intent(state: ChatState) -> str:
    intent = state.get("intent")
    if intent == "schedule": return "schedule_flow"
    elif intent == "doctor": return "doctor_flow"
    else: return "other_flow"

workflow = StateGraph(ChatState)

workflow.add_node("detect_master_intent", detect_master_intent)
workflow.add_node("extract_details", extract_appointment_details)
workflow.add_node("schedule_appointment", schedule_appointment)
workflow.add_node("recommend_doctor", recommend_doctor)
workflow.add_node("non_doctor", non_doctor_response)

workflow.set_entry_point("detect_master_intent")

workflow.add_conditional_edges(
    "detect_master_intent",
    route_by_intent,
    {
        "schedule_flow": "extract_details",
        "doctor_flow": "recommend_doctor",
        "other_flow": "non_doctor",
    },
)

workflow.add_edge("extract_details", "schedule_appointment")
workflow.add_edge("schedule_appointment", END)
workflow.add_edge("recommend_doctor", END)
workflow.add_edge("non_doctor", END)

medical_chatbot = workflow.compile()

# --- TEST EXECUTION ---
if __name__ == "__main__":
    print("--- Test 1: Scheduling ---")
    # This input is messy but structured output will handle it perfectly
    input_text = "I want to schedule an appointment for patient 5 with doctor 19 on 2025-10-25 at 02:00 PM."
    
    state_input = {"messages": [HumanMessage(content=input_text)]}
    result = medical_chatbot.invoke(state_input)
    print(result["messages"][-1].content)
    
    
    