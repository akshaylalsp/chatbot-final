import streamlit as st
from modules.langchain_helper import get_chain
from modules.SetupDb import SetupDb




def initialize_chatbot(location):
    SetupDb(location)


def execute_result(nl_question):
    chain = get_chain()
    return chain.invoke(nl_question)

def main():
    st.title("Theatre Chatbot")

    if 'location' not in st.session_state:
        st.session_state.location = ""
    location = st.text_input("Enter location:", value=st.session_state.location)
    st.session_state.location = location

    if st.button("Initialize Chatbot"):
        if location:
            chatbot_instance = initialize_chatbot(location)
            st.success("Chatbot initialized successfully!")
            st.session_state.chatbot_instance = chatbot_instance
        else:
            st.error("Please enter a location.")

    if 'chatbot_instance' in st.session_state:
        if 'nl_question' not in st.session_state:
            st.session_state.nl_question = ""
        nl_question = st.text_input("Enter your question:", value=st.session_state.nl_question)
        st.session_state.nl_question = nl_question

        if st.button("Get Answer"):
            if nl_question:
                result = execute_result(nl_question)
                st.write("Chatbot's Response:")
                st.write(result['result'])
            else:
                st.error("Please enter a question.")

if __name__ == "__main__":
    main()