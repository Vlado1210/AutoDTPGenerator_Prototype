# Bring in deps
import os 
from apikey import apikey 
import streamlit as st 
import matplotlib.pyplot as plt
import numpy as np
import base64
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
import matplotlib 
matplotlib.use("TkAgg")



#Load CSS file
with open("styles.css") as style_source:
    st.markdown(f"<style>{style_source.read()}</style>",unsafe_allow_html=True)
    
os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('AutoDTP Generator')
appprompt = st.text_input("What tool do you want to generate")
prompt = None
prompt1 = None
featuresprompt = None
appusers = None
teammembers = None
techniques = None
appinput = None
teaminput = None
if appprompt:
    featuresprompt = st.text_input('What features will this app include')
if featuresprompt:
    appusers = st.text_input("Who will use this app?")
if appusers:
    appinput = ("An application to " + appprompt + " that will include features like " + featuresprompt +
                " and is going to be used by " + appusers)
    st.write(appinput)
    teammembers = st.text_input("Who conforms your team")
if teammembers:
    techniques = st.text_input("Which techniques you are going to use?")
if techniques:
    teaminput = ("The team is made up of " + teammembers + " and this team is going to be using techniques\
                  for project management and software development like " + techniques)
    prompt1 = appinput + ". "+ teaminput
    st.write(prompt1)
    prompt = appinput
#prompt = st.text_input('Plug in your prompt here')

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Your task is to help the design thinking process to create an application,\
          Write a possible name for the application: {topic}'
)

empathymap_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Your task is to help the design thinking process to create an application.\
Create an empathy map for the application: {topic}.'
)

persona_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Your task is to help the design thinking process to create an application. Write a possible persona \
for the application: {topic} . Each person should have a name, date of birth, \
place of birth, and a paragraph explaining this persons problem and how the proposed application \
can help them solve it. Display the result as a 2x2 HTML table with the title "persona",\
the first row of the first column must be empty, the second row of the first column must have the \
paragraph about the person, the first row of the second column must contain the name, date of birth and\
place of birth and the rest of the box will be blank, the fontsize on the table must be 10px'
)

competitors_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Your task is to help the design thinking process to create an application. Create a list of \
applications similar to the application: {topic}, this list should contain\
the name of the similar app and the pros and cons of each with respect to the proposed application.\
Display the result as a HTML table with the title "Competitors" with the matching application in each\
row and the pros and cons in two separate columns next to the application.,the fontsize on the table must be 10px'
)

journeymap_task_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Your task is to help the design thinking process to create an application.\
Create a list of 5 actions or tasks that represents a journey map to perform while \
using the application: {topic}. These actions must try to recreate \
each step of the user while browsing through the application.\
Give me the response listing each task from A to E, writing a line break between \
each one, like in this example: A. task1\n B. task2\n C. task3\n D. task4\n E. task5\n'
)
journeymap_emotions_template = PromptTemplate(
    input_variables = ['journeymaptask'], 
    template='For each of the tasks  given: {journeymaptask},\
        give an example of how the user would feel when performing said task, try to include \
negative and positive emotions, keep each task under 25 words. Your answer must only conatain \
    the feelings and emotions'
)
jouneymap_values_template = PromptTemplate(
    input_variables = ['journeymapemotion'], 
    template='Give each of the emotions felt by the users a value between 1 and 5, being 1 the worst negative experience and \
5 being the best positive experience . The answer should be only the numerical values \
 as in the following example: "4,3,2,5,1" {journeymapemotion} ' 
)
proyectplan_template = PromptTemplate(
    input_variables = ['prompt1'], 
    template='Your task is to help the design thinking process to create an application,\
          Write a detailed proyect plan for the proyect given: {prompt1}, give specific intructions and\
            objectives for each member of the team'
)



# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=1.0) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title')
persona_chain = LLMChain(llm=llm, prompt=persona_template, verbose=True, output_key='persona')
competitors_chain = LLMChain(llm=llm, prompt=competitors_template, verbose=True, output_key='competitors')
empathymap_chain = LLMChain(llm=llm, prompt=empathymap_template, verbose=True, output_key='empathymap')
journeymap_task_chain = LLMChain(llm=llm, prompt=journeymap_task_template, verbose=True, output_key='journeymaptask')
journeymap_emotions_chain = LLMChain(llm=llm, prompt=journeymap_emotions_template, verbose=True, output_key='journeymapemotion')
journeymap_value_chain = LLMChain(llm=llm, prompt=jouneymap_values_template, verbose=True, output_key='journeymapvalue')
proyectplan_chain = LLMChain(llm=llm, prompt=proyectplan_template, verbose=True, output_key='proyectplan')


wiki = WikipediaAPIWrapper()


# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    empathymap = empathymap_chain.run(prompt)
    persona = persona_chain.run(prompt)
    competitors = competitors_chain.run(prompt)
    journeymaptasks = journeymap_task_chain.run(prompt)
    journeymapemotions = journeymap_emotions_chain.run(prompt)
    journeymapvalues = journeymap_value_chain.run(prompt)
    proyectplan = proyectplan_chain.run(prompt)

    with st.expander("Possible App Name"):
        st.write(title)
    with st.expander("Proyect Plan"):
        st.write(proyectplan)

    with st.expander('Personas'): 
        st.markdown(persona, unsafe_allow_html=True)

    with st.expander('Competitors'): 
        st.markdown(competitors, unsafe_allow_html=True)

    with st.expander('Empathymap'): 
        st.write(empathymap)

    with st.expander('Journeymap'): 
        valores = journeymapvalues.split(",")
        listan = [int(valor) for valor in valores]
        y1 = listan
        x_labels = ["A", "B", "C", "D", "E"]
        x_ticks = [0, 1, 2, 3, 4]
        fig, ax = plt.subplots()
        plt.plot(y1, marker="o")
        plt.xlabel("Activity")
        plt.ylabel("NEGATIVE REACTION                 POSITIVE REACTION")
        plt.text(4.5, 4, journeymaptasks, bbox=dict(facecolor='white', edgecolor='black'))
        texto = "TASKS"
        plt.text(6, 5.3, texto, fontsize=12)
        plt.text(4.5, 2, journeymapemotions, bbox=dict(facecolor='white', edgecolor='black'))
        texto = "EMOTIONS"
        plt.text(6, 3.2, texto, fontsize=12)
        plt.xticks(ticks=x_ticks, labels=x_labels)
        ax = plt.gca()
        plt.ylim(1, 5.5)
        ax.set_yticklabels([])
        plt.axhline(y=3, color='black', linestyle='--')
        st.pyplot(fig)
