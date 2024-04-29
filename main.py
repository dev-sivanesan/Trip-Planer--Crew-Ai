import streamlit as st
from datetime import datetime
from crewai import Crew, Process
from textwrap import dedent
from trip_crew.trip_agents import TripAgents
from trip_crew.trip_tasks import TripTasks
from langchain_google_genai import ChatGoogleGenerativeAI
import os

os.environ['GOOGLE_API_KEY']=""
llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, temperature=0.1, google_api_key=os.environ.get('GOOGLE_API_KEY', ''))
os.environ['OPENAI_API_KEY'] = ""
os.environ['SERPER_API_KEY'] = ''

class TripCrew:

    def __init__(self, origin, cities, date_range, interests):
        self.cities = cities
        self.origin = origin
        self.interests = interests
        self.date_range = date_range

    def run(self):
        agents = TripAgents()
        tasks = TripTasks()

        city_selector_agent = agents.city_selection_agent()
        local_expert_agent = agents.local_expert()
        travel_concierge_agent = agents.travel_concierge()

        identify_task = tasks.identify_task(
            city_selector_agent,
            self.origin,
            self.cities,
            self.interests,
            self.date_range
        )
        gather_task = tasks.gather_task(
            local_expert_agent,
            self.origin,
            self.interests,
            self.date_range
        )
        plan_task = tasks.plan_task(
            travel_concierge_agent,
            self.origin,
            self.interests,
            self.date_range
        )

        crew = Crew(
            agents=[city_selector_agent, local_expert_agent, travel_concierge_agent],
            tasks=[identify_task, gather_task, plan_task],
            manager_llm=llm,
            process=Process.hierarchical,
            verbose=True
        )

        result = crew.kickoff()
        return result

def main():
    st.title("Trip Planner Crew")
    st.write("Welcome to Trip Planner Crew! Let's plan your next trip.")
    current_date = datetime.now().date()
    location = st.text_input("From where will you be travelling from?")
    cities = st.text_input("What are your cities options you are interested in visiting?")
    date_range = st.text_input("What is the date range you are interested in traveling? (Format: YYYY-MM-DD to YYYY-MM-DD)")
    interests = st.text_input("What are some of your high level interests and hobbies?")

    if "to" in date_range:
        start_str, end_str = date_range.split("to", 1)
        start_date = datetime.strptime(start_str.strip(), "%Y-%m-%d").date()
        end_date = datetime.strptime(end_str.strip(), "%Y-%m-%d").date()
    else:
        start_date = current_date
        end_date = current_date
    if start_date < current_date or end_date < current_date:
        st.error("Please select a start date and end date  starting from today.")
        return
    if start_date > end_date:
        st.error("End date must be after start date.")
        return
    if start_date < current_date or end_date < current_date:
        st.error("Please select a date starting from today.")
        return
    

    if st.button("Plan Trip"):
       with st.spinner("Generating Trip Plan..."):
            trip_crew = TripCrew(location, cities, date_range, interests)
            result = trip_crew.run()
           
       st.empty()
       st.markdown("## Here is your Trip Plan")
       st.write("## Here is your Trip Plan")
       st.write(result)
       st.success("Trip planned successfully!")
     
      
if __name__ == "__main__":
    main()
