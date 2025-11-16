import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.llm_math.base import LLMMathChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents import Tool
from langchain_classic.agents.initialize import initialize_agent
from langchain_classic.callbacks import StreamlitCallbackHandler

#st config
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant",page_icon="🦜")
st.title("Text to Math Problem Solver")

groq_api_key = st.sidebar.text_input(label="Groq API KEY",type="password")

if not groq_api_key:
    st.info("Please add your groq api key to continue")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=groq_api_key)

wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="A tool for searching the internet and find various information on the mentioned topic"
)

math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name = "Calculator",
    func = math_chain.run,
    description="A tool for answer math related questions.Only input mathematical expression need to be provided."
)

prompt = """
You are a agent tasked for solving users mathematical questions. 
Logically arrive at the solution and provide detailed explanation,
and display it point wise for the question below.
Question:{question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

chain = LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="Tool for answering logic based and reasoning questions"
)

assistant_agent = initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant","content":"Hi I am a Math Chatbot who can answer all your math questions."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

question = st.text_area("Enter your question:")
if st.button("Find my answer"):
    if question:
        with st.spinner("Generate response...."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({"role":"assistant","content":response})
            st.write("### Response:")
            st.success(response)
    else:
        st.warning("Please enter the question")


