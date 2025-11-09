import warnings
warnings.filterwarnings("ignore")

from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


def generate_answer(question: str, context: str) -> str:
   from langchain_google_genai import ChatGoogleGenerativeAI
   llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
   
   system_template = """
   You are an AI Invoice Auditor Assistant.

   Follow the rules strictly:

   1. **Greetings or Irrelevant Questions**  
   If the user's message is a greeting, casual chat, or unrelated to invoices (e.g., "hi", "how are you", "what’s up", "who are you", "help", "thank you", etc.), reply EXACTLY with:  
   "I am the report analyzer agent. If you have any question regarding any invoice data or related to it, I can answer."  
   → Do NOT use or refer to any context for these.

   2. **Invoice-Related Questions**  
   If the question is about invoices (e.g., invoice number, PO, vendor, validation, amount, etc.), answer clearly using the provided information.  
   Never mention "context" or "based on context."

   3. **Missing Information**  
   If the question is invoice-related but not answerable, reply:  
   "I don't have enough information to answer this question."

   Flow:  
   Check → Greeting/Irrelevant? (Rule 1) → Invoice-related? (Rule 2) → Else (Rule 3)
   """

   human_template = """
   User Question: {input}
   Context: {context}
   """

   prompt = ChatPromptTemplate.from_messages([
       ("system", system_template),
       ("human", human_template)
   ])
   
   chain = prompt | llm
   result = chain.invoke({
       "input": question,
       "context": context,
   })

   return result.content


__all__ = ["generate_answer"]
