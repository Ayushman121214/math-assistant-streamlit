import streamlit as st  
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain ,LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import   AgentType
from langchain.agents import Tool ,initialize_agent
# from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


# setyp streamlit app
st.set_page_config(page_title="text to math problem sevver and data search assistent")

st.title("text to math problem solver using google gemma 2")

groq_api_key=st.sidebar.text_input(label="groq api key",type="password")

if not groq_api_key:
    st.info("please provide the api key to continue")
    st.stop()
llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)


# initilizattion the Tool
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="a tool for searching the internet to find various infromation on  topic mention"
    
)   

# initilize the math tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculater=Tool(
    name="Calculator",
    func=math_chain.run,
    description="a tool for answering math related question.only give input mathmetical expression"
) 
prompt="""
you  are the agent your task to solve the mathmetical problem.
logically arrive the solution and display it point wise for the question below
Question:{question}
Answer:
"""
prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# combine all the tols into chain
chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Resoning Tool",
    func=chain.run,
    description="a tool for answering logical and reasoning question"
)

# initilize the agents
assistent_agent=initialize_agent(
    tools=[wikipedia_tool,calculater,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    Verbose=True,
    handle_parsing_errors=True
)

if "message" not in st.session_state:
    st.session_state["message"]=[
        {"role":"assistent","content":"hi i am the mathmetical chatbot who can assist you to solve math problems "}
    ]
    
for msg in st.session_state.message:
    st.chat_message(msg["role"]).write(msg["content"])    


# function to generate response
def generate_response(question):
    response=assistent_agent.invoke({"input":question})
    return response

# lets start with the intraction
questionn=st.text_area("enter your question....")

if st.button("find the answer"):
    if questionn:
        with st.spinner("generate response...."):
            st.session_state.message.append({"role":"user","content":questionn})
            st.chat_message("user").write(questionn)
            
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistent_agent.run(st.session_state.message,callbacks=[st_cb])
            
            st.session_state.message.append({"role":"assisant","content":response})
            st.write("### response:")
            st.success(response)
            
            
else:
    st.warning("quesion daal bhosharika")
     