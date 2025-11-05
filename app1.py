import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain , LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentType
from langchain.agents import Tool , initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
import os

st.set_page_config(page_title="Text to Math Problem solver")
st.title("Math Teacher")
load_dotenv(dotenv_path=r"C:\Users\dewan\Coding\GenAIKN\Langchain\.env")


groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=groq_api_key)

## Initializing tools

wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the web and solving the problem"

)


## Initialize the Math Tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related question. Only input mathmatical expression."

)

prompt=""""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_vatiables=["question"],
    template=prompt
)

# MathProblem Tool

chain=LLMChain(llm=llm,prompt=prompt_template)

resoning_tool=Tool(
    name="Resoning tool",
    func=chain.run,
    description="A tool for answering logic-based and resoning questions"
)

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,resoning_tool,calculator],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "message" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"Assistant",
         "content":"Hi I'm Math ChatBot"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


## Function 
def generate_response(question):
    response=assistant_agent.invoke({'input':question})
    return response

question=st.text_area("Enter Your Question:","Q?")






if st.button("find my answer"):
    if question:
        with st.spinner("Generate Response :"):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,
                                         callbacks=[st_cb])
            
            st.session_state.messages.append({"role":"assistant",
                                             "content":question})
            st.write('### Write')
            st.success(response) 
    else :
        st.warning("Please Enter the question")


































































































