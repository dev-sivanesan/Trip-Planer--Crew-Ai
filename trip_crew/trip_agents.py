from crewai import Agent
from langchain.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ["OPENAI_MODEL_NAME"]='your model name'
llm = ChatGoogleGenerativeAI(model="gemini-pro",verbose = True,temperature = 0.1,google_api_key='your gookle api key')
from crewai_tools import  WebsiteSearchTool
from crewai_tools import SerperDevTool
searchtool=SerperDevTool()
browsertool= WebsiteSearchTool()



from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools

os.environ ['OPENAI_API_KEY']="your open appi key"
os.environ ['SERPER_API_KEY']='your api key'

class TripAgents():

  def city_selection_agent(self):
    return Agent(
        role='City Selection Expert',
        goal='Select the best city based on weather, season, and prices',
        backstory=
        'An expert in analyzing travel data to pick ideal destinations',
      
      
        tools=[
            
            BrowserTools.scrape_and_summarize_website,
          
          CalculatorTools.calculate,
          searchtool,
          browsertool
        ],
      llm=llm,
     
      
        verbose=True)

  def local_expert(self):
    return Agent(
        role='Local Expert at this city',
        goal='Provide the BEST insights about the selected city',
        backstory="""A knowledgeable local guide with extensive information
        about the city, it's attractions and customs""",
        tools=[
            
            BrowserTools.scrape_and_summarize_website,
          
          CalculatorTools.calculate,
          searchtool,
          browsertool
        ],
      llm=llm,
     
        verbose=True)

  def travel_concierge(self):
    return Agent(
        role='Amazing Travel Concierge',
        goal="""Create the most amazing travel itineraries with budget and 
        packing suggestions for the city""",
        backstory="""Specialist in travel planning and logistics with 
        decades of experience""",
      
        tools=[
            
            BrowserTools.scrape_and_summarize_website,
            CalculatorTools.calculate,
          
          searchtool,
          browsertool
        ],
      llm=llm,
     

      
      
     verbose=True)
