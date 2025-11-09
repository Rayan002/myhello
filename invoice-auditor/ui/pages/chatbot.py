import sys
import uuid
from pathlib import Path
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

sys.path.append(str(Path(__file__).resolve().parents[2]))
from langgraph_workflows.chatbot_workflow import build_chatbot, retrieve_all_threads, generate_thread_id, get_thread_preview, save_thread_title, update_thread_timestamp


st.set_page_config(page_title="LangGraph Chatbot", layout="wide")

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = []
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    st.session_state['message_history'] = []

def load_conversation(thread_id):
  bot = build_chatbot()
  state = bot.get_state(config={'configurable': {'thread_id': thread_id}})
  messages = state.values.get('messages', [])
  return messages


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
  st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
  st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = list(set(retrieve_all_threads()))

if st.session_state['thread_id'] not in st.session_state['chat_threads']:
    st.session_state['chat_threads'].append(st.session_state['thread_id'])
    st.session_state['chat_threads'] = list(set(st.session_state['chat_threads']))

bot = build_chatbot()

# **************************************** Sidebar UI *********************************
st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
  reset_chat()
  st.rerun()

st.sidebar.header('My Conversations')
unique_threads = list(set(st.session_state['chat_threads']))
for idx, thread_id in enumerate(unique_threads):
    title = get_thread_preview(thread_id)
    if st.sidebar.button(title, key=f"thread_{idx}_{thread_id}", use_container_width=True):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            content = msg.content
            metrics = {}
            if isinstance(msg, AIMessage) and hasattr(msg, 'additional_kwargs'):
                metrics = msg.additional_kwargs.get('rag_metrics', {})
            temp_messages.append({'role': role, 'content': content, 'metrics': metrics})
        st.session_state['message_history'] = temp_messages
        st.rerun()

# **************************************** Main UI ************************************
for idx, message in enumerate(st.session_state['message_history']):
  with st.chat_message(message['role']):
      st.text(message['content'])
      # Show metrics for assistant messages if available
      if message['role'] == 'assistant' and message.get('metrics'):
        metrics = message['metrics']
        if metrics:
          st.markdown("---")
          st.markdown("**RAG Metrics:**", help="Relevance: How relevant the answer is to the question. Groundedness: How well the answer is supported by the context. Context Relevance: How relevant the retrieved context is.")
          col1, col2, col3 = st.columns(3)
          with col1:
            st.caption("Relevance")
            st.text(f"{metrics.get('relevance', 0):.2%}")
          with col2:
            st.caption("Groundedness")
            st.text(f"{metrics.get('groundedness', 0):.2%}")
          with col3:
            st.caption("Context Relevance")
            st.text(f"{metrics.get('context_relevance', 0):.2%}")

user_input = st.chat_input('Type here')

if user_input:
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        with st.spinner("Invoice Agent is thinking..."):
            response = bot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)

        ai_message_obj = response['messages'][-1]
        ai_message = ai_message_obj.content
        metrics = {}
        if isinstance(ai_message_obj, AIMessage) and hasattr(ai_message_obj, 'additional_kwargs'):
            metrics = ai_message_obj.additional_kwargs.get('rag_metrics', {})
        
        st.text(ai_message)
        if metrics:
            st.markdown("---")
            st.markdown("**RAG Metrics:**", help="Relevance: How relevant the answer is to the question. Groundedness: How well the answer is supported by the context. Context Relevance: How relevant the retrieved context is.")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption("Relevance")
                st.text(f"{metrics.get('relevance', 0):.2%}")
            with col2:
                st.caption("Groundedness")
                st.text(f"{metrics.get('groundedness', 0):.2%}")
            with col3:
                st.caption("Context Relevance")
                st.text(f"{metrics.get('context_relevance', 0):.2%}")

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    st.session_state['message_history'].append({
        'role': 'assistant', 
        'content': ai_message,
        'metrics': metrics
    })
    
    update_thread_timestamp(st.session_state['thread_id'])
    
    if len(st.session_state['message_history']) == 2:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a concise, descriptive title (maximum 30 characters) for a chat conversation based on the user's first question. The title should capture the main topic or intent of the question. Return only the title, no additional text."),
            ("human", "User's first question: {question}"),
        ])
        chain = prompt | llm
        title_result = chain.invoke({"question": user_input}).content
        if len(title_result) > 30:
            title_result = title_result[:27] + "..."
        save_thread_title(st.session_state['thread_id'], title_result)
    
    st.rerun()