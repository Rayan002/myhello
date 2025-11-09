from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate


def format_context(chunks: List[Dict[str, Any]]) -> str:
   lines: List[str] = []
   for ch in chunks:
       meta = ch.get("metadata", {})
       lines.append(f"{meta}\n{ch.get('page_content','')}")
   return "\n\n".join(lines)


def grade_documents(question: str, ctx: List[Dict[str, Any]]) -> str:
   if not ctx:
       return {"decision": "bad", "reason": "no documents"}
   from langchain_google_genai import ChatGoogleGenerativeAI
   llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
   prompt = ChatPromptTemplate.from_messages([
       ("system", "You judge if the provided documents are enough to answer the question."),
       ("human", "Question: {q}\n\nDocs:\n{ctx}\n\nReply only in one word, either 'good' or 'bad'. No extra stuff"),
   ])
   resp = (prompt | llm).invoke({"q": question, "ctx": ctx})
   text = getattr(resp, "content", str(resp)).strip().lower()
   decision = "good" if text == "good" else "bad"
   return {"decision": decision}


__all__ = ["format_context", "grade_documents"]
