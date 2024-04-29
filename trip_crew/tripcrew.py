from crewai import Crew, Process
from trip_crew.trip_agents import TripAgents
from trip_crew.trip_tasks import TripTasks
from langchain_google_genai import ChatGoogleGenerativeAI
import os
llm = ChatGoogleGenerativeAI(model="gemini-pro",verbose = True,temperature = 0.1,google_api_key=os.environ['GOOGLE_API_KEY'])
class TripCrew:
    def _init_(self, origin, cities, date_range, interests):
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
            agents=[
                city_selector_agent, local_expert_agent, travel_concierge_agent
            ],
            tasks=[identify_task, gather_task, plan_task],
          process= Process.hierarchical,
            manager_llm=llm,
            verbose=True
        )

        result = crew.kickoff()
        return result