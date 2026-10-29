import os 
from typing import TypedDict

class pipelinestate(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str

from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

def editor_node(state: pipelinestate) -> dict:
    """Stage 1: Cleans up grammar, removes typos,  and refines the tone."""
    prompt = f"""
    You are a professional editor. Your task is to clean up the grammar, remove typos, and refine the tone of the following text. 
    Please provide the edited text in a clear and concise manner.

    f"Text to edit: {state['raw_input']}"
    Please provide the edited text below:
    """
    response = llm.invoke(prompt)
    return {"edited_text": response.content.strip()}


def scriptwriter_node(state: pipelinestate) -> dict:
    """Stage 2: Converts the edited text into a script format."""
    print("\n--- [stage 2] Executing scriptwriter_node ---\n")
    prompt = f"""
    you are a charmastic Youtuber scriptwriter. Your task is to convert the following text into a script format suitable for a YouTube video.
    it should include an engaging introduction, main content, and a conclusion. Please ensure the script is clear, concise, and engaging for viewers.
    f"Edited text: {state['edited_text']}"
    Please provide the script below:
    """
    response = llm.invoke(prompt)
    return {"script_text": response.content.strip()}

def translator_node(state: pipelinestate) -> dict:
    """Stage 3: Translates the script into natural flowing Hinglish."""
    print("\n--- [stage 3] Executing Hinglish translator_node ---\n")
    prompt = f"""
    You are a professional translator. Your task is to translate the following script into Hinglish while maintaining the original meaning and tone.
    f"Script text: {state['script_text']}"
    Please provide the translated script below:
    """
    response = llm.invoke(prompt)
    return {"final_output": response.content.strip()}


from langgraph.graph import StateGraph, START, END
graph = StateGraph(pipelinestate)
graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwriter_node)
graph.add_node("translator", translator_node)

graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)

app = graph.compile()

result = app.invoke({"raw_input": "Hello, I am a professional editor. I will clean up your text and make it more engaging for your audience."})
print("\n--- Final Output ---\n")
print(result["final_output"])
