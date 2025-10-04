from dotenv import load_dotenv
import os
import json
import requests
from pypdf import PdfReader
import gradio as gr
import google.generativeai as genai

#TEST UPloading to HF
load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_unknown_question(question: str) -> dict:
    print(f"--- LOG: UNKNOWN QUESTION RECORDED: '{question}' ---")
    push(f"Recording unknown question: {question}")
    return {"status": "success", "message": f"Question '{question}' has been recorded for review."}

def record_user_details(email: str) -> dict:
    print(f"--- LOG: USER EMAIL RECORDED: '{email}' ---")
    push(f"Recording user email: {email}")
    return {"status": "success", "message": f"Thank you! The email '{email}' has been recorded."}


class Me:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        self.model = genai.GenerativeModel("gemini-2.5-flash")  # or "gemini-1.5-pro"
        self.name = "Pawee Indulakshana"

        # load linkedin text
        reader = PdfReader("me/pawee.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        # load summary
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.name
            arguments = dict(tool_call.args)
            
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            
            results.append({
                "role": "tool",
                "parts": [{
                    "function_response": {
                        "name": tool_name,
                        "response": result
                    }
                }]
            })
        return results
    
    def chat(self, message, history):
        full_history = [{"role": "user", "parts": [self.system_prompt()]}]
        for h in history:
            full_history.append({"role": "user", "parts": [h[0]]})
            full_history.append({"role": "model", "parts": [h[1]]})
        full_history.append({"role": "user", "parts": [message]})

        tools = [record_unknown_question, record_user_details]
        
        response = self.model.generate_content(
            full_history,
            tools=tools
        )
        
        while (response.candidates and 
              response.candidates[0].content.parts[0].function_call): 

           function_calls = response.candidates[0].content.parts 
           
           print(f"Model requested {len(function_calls)} tool call(s).", flush=True)
           
           tool_requests = [part.function_call for part in function_calls if hasattr(part, 'function_call')]
           tool_results = self.handle_tool_call(tool_requests)
           
           full_history.append(response.candidates[0].content) #
           full_history.extend(tool_results)
        
           response = self.model.generate_content(
               full_history,
               tools=tools
           )
        
        return response.text
    

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat).launch()
