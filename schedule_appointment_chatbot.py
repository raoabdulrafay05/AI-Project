# schedule_appointment_chatbot.py

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# IMPORTANT: import from your Flask app file
from app import app, db, Doctor, Appointment, Patient

# =========================================================
# LANGGRAPH STATE
# =========================================================
class ChatState(TypedDict):
    messages: List
    patient_id: Optional[int]
    doctor_id: Optional[int]
    appointment_date: Optional[str]  # YYYY-MM-DD
    time_slot: Optional[str]
    success: Optional[bool]


# =========================================================
# LLM
# =========================================================
llm = ChatOpenAI(model="gpt-5.2", temperature=0)


# =========================================================
# NODE 1: EXTRACT APPOINTMENT DETAILS FROM USER
# =========================================================
def extract_appointment_details(state: ChatState) -> ChatState:
    user_text = state["messages"][-1].content

    system_prompt = (
        "You are a medical assistant.\n"
        "Extract the following details from the user request for an appointment:\n"
        "1. Patient ID (integer)\n"
        "2. Doctor ID (integer)\n"
        "3. Appointment date (YYYY-MM-DD)\n"
        "4. Time slot (e.g., '10:00 AM')\n\n"
        "Return ONLY a JSON object with keys: "
        '{"patient_id": int, "doctor_id": int, "appointment_date": str, "time_slot": str}'
    )

    result = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ])

    try:
        import json
        data = json.loads(result.content)
        state["patient_id"] = int(data.get("patient_id"))
        state["doctor_id"] = int(data.get("doctor_id"))
        state["appointment_date"] = data.get("appointment_date")
        state["time_slot"] = data.get("time_slot")
    except Exception as e:
        state["messages"].append(AIMessage(content=f"Failed to parse appointment details: {e}"))
        state["success"] = False

    return state


# =========================================================
# NODE 2: VALIDATE AND SCHEDULE APPOINTMENT
# =========================================================
def schedule_appointment(state: ChatState) -> ChatState:
    if not all([state.get("patient_id"), state.get("doctor_id"), state.get("appointment_date"), state.get("time_slot")]):
        state["messages"].append(AIMessage(content="Missing required appointment details."))
        state["success"] = False
        return state

    with app.app_context():
        patient = Patient.query.get(state["patient_id"])
        doctor = Doctor.query.get(state["doctor_id"])

        if not patient:
            state["messages"].append(AIMessage(content="Patient ID not found."))
            state["success"] = False
            return state

        if not doctor:
            state["messages"].append(AIMessage(content="Doctor ID not found."))
            state["success"] = False
            return state

        try:
            new_appointment = Appointment(
                patientid=state["patient_id"],
                doctorid=state["doctor_id"],
                appointmentdate=datetime.strptime(state["appointment_date"], '%Y-%m-%d').date(),
                timeslot=state["time_slot"],
                status="Scheduled"
            )
            db.session.add(new_appointment)
            db.session.commit()
            state["messages"].append(AIMessage(content="Appointment successfully scheduled!"))
            state["success"] = True
        except Exception as e:
            db.session.rollback()
            state["messages"].append(AIMessage(content=f"Failed to schedule appointment: {e}"))
            state["success"] = False

    return state


# =========================================================
# LANGGRAPH WORKFLOW
# =========================================================
workflow = StateGraph(ChatState)

workflow.add_node("extract_details", extract_appointment_details)
workflow.add_node("schedule_appointment", schedule_appointment)

workflow.set_entry_point("extract_details")
workflow.add_edge("extract_details", "schedule_appointment")
workflow.add_edge("schedule_appointment", END)

appointment_chatbot = workflow.compile()


# =========================================================
# TEST RUN
# =========================================================
if __name__ == "__main__":
    state: ChatState = {
        "messages": [HumanMessage(content="I want to schedule an appointment for patient 5 with doctor 2 on 2025-12-20 at 10:30 AM")],
        "patient_id": None,
        "doctor_id": None,
        "appointment_date": None,
        "time_slot": None,
        "success": None,
    }

    result = appointment_chatbot.invoke(state)

    for msg in result["messages"]:
        print(msg.content)
