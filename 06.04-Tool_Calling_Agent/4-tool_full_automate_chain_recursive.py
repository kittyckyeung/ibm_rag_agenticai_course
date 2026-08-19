## Convert the input prompt to a HumanMessage
## Pass the message to LLM with tools
## Extract tool calls from LLM response
## Update message history with tool results
## Send updated messages back to LLM
## Repeat steps 3-5 as needed
## Finally, extract just the content from the final message using RunnableLambda

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Suppress pytube errors
import logging
pytube_logger = logging.getLogger('pytube')
pytube_logger.setLevel(logging.ERROR)

# Suppress yt-dlp warnings
yt_dpl_logger = logging.getLogger('yt_dlp')
yt_dpl_logger.setLevel(logging.ERROR)

# Configure root logger for debug output
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import re
from pytube import YouTube, Search
from langchain_core.tools import tool
from IPython.display import display, JSON
import yt_dlp
from typing import List, Dict
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
import json
from langchain.agents import create_agent
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.messages import HumanMessage, ToolMessage
import json


# MiniMax-compatible OpenAI API settings
LLM_KEY_LOCATION = r"C:\Users\user\Desktop\Coding\minimax_Key.txt"
LLM_MODEL_NAME = "MiniMax-M3"
#LLM_MODEL_BASE_URL = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"
LLM_MODEL_BASE_URL = "https://api.minimaxi.chat/v1"


def load_api_key(key_file: str) -> str:
    values = dotenv_values(key_file)
    api_key = values.get("MINIMAX_API_KEY")
    group_id = values.get("MINIMAX_GROUP_ID")

    api_key = api_key.strip() if isinstance(api_key, str) else ""
    group_id = group_id.strip() if isinstance(group_id, str) else ""

    if not api_key:
        raise ValueError("MINIMAX_API_KEY is missing or empty.")
    if not group_id:
        raise ValueError("MINIMAX_GROUP_ID is missing or empty.")

    #return api_key, group_id
    return api_key


def call_llm() -> ChatOpenAI:
    """Create and return a configured ChatOpenAI model instance."""
    api_key = load_api_key(LLM_KEY_LOCATION)

    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        openai_api_key=api_key,
        openai_api_base=LLM_MODEL_BASE_URL,
        temperature=0.5,
    )
    return llm


def log_and_invoke(llm, messages, label: str = "LLM"):
    """Invoke an LLM and log messages, responses, and tool calls for debugging."""
    try:
        logger.debug("%s - invoking with messages: %s", label, messages)
        # Some LLM/agent invoke APIs expect a dict payload (e.g. {"input": ...})
        # while other helpers in this script construct lists of HumanMessage/ToolMessage.
        # Normalize to a dict if a list of message-like objects is provided.
        payload = messages
        if isinstance(messages, list):
            try:
                # If list contains message objects with a .content attribute,
                # convert them to the chat API expected format: a list of
                # {'role': 'user'|'system'|'tool', 'content': '...'} dicts.
                msgs = []
                for m in messages:
                    # If it's already a plain dict, use it (but ensure 'content' is string)
                    if isinstance(m, dict):
                        msg = m.copy()
                        # ensure content is string
                        c = msg.get("content")
                        if isinstance(c, list):
                            msg["content"] = "\n".join(str(x) for x in c)
                        msgs.append(msg)
                        continue

                    content = getattr(m, "content", None)
                    if content is None:
                        # skip non-message-like entries
                        continue

                    # normalize content to string if it's a list
                    if isinstance(content, list):
                        content_str = "\n".join(str(x) for x in content)
                    else:
                        content_str = str(content)

                    if isinstance(m, ToolMessage):
                        # convert tool result into an assistant message containing
                        # the tool name and its output text. Avoid using the
                        # 'tool' role or 'tool_call_id' to prevent backend
                        # validation mismatches; the tool result is now plain
                        # assistant content that documents the tool output.
                        tool_name = getattr(m, "name", None) or getattr(m, "tool", None) or "tool"
                        tool_content = f"[tool:{tool_name}] {content_str}"
                        msgs.append({"role": "assistant", "content": tool_content})
                        continue

                    if isinstance(m, HumanMessage):
                        role = "user"
                    else:
                        role = getattr(m, "role", "assistant")

                    msgs.append({"role": role, "content": content_str})
                if msgs:
                    payload = {"messages": msgs}
            except Exception:
                payload = messages

        resp = llm.invoke(payload)

        # Normalize response: some clients return objects with attributes,
        # others return plain dicts. Convert dict responses to a simple
        # object with `content` and `tool_calls` attributes so downstream
        # code can always access `.content` and `.tool_calls`.
        if isinstance(resp, dict):
            content = resp.get("messages", resp)
            # Extract tool_calls from any message entries that include them
            parsed_tool_calls = []
            for msg in resp.get("messages", []) if isinstance(resp.get("messages", None), list) else []:
                if isinstance(msg, dict):
                    for tc in msg.get("tool_calls", []) or []:
                        func = tc.get("function", {}) if isinstance(tc, dict) else {}
                        name = func.get("name") or tc.get("name")
                        args_raw = func.get("arguments") or tc.get("arguments")
                        args = {}
                        if isinstance(args_raw, str):
                            try:
                                args = json.loads(args_raw)
                            except Exception:
                                args = {}
                        elif isinstance(args_raw, dict):
                            args = args_raw
                        parsed_tool_calls.append({"name": name, "args": args, "id": tc.get("id")})

            # Create a lightweight response object
            from types import SimpleNamespace
            wrapped = SimpleNamespace()
            wrapped.content = content
            wrapped.tool_calls = parsed_tool_calls
            # preserve original dict for callers that might inspect it
            wrapped.raw = resp
            logger.debug("%s - response content: %s", label, wrapped.content)
            logger.debug("%s - tool_calls: %s", label, wrapped.tool_calls)
            return wrapped

        content = getattr(resp, "content", resp)
        logger.debug("%s - response content: %s", label, content)
        tool_calls = getattr(resp, "tool_calls", None)
        logger.debug("%s - tool_calls: %s", label, tool_calls)
        return resp
    except Exception as e:
        logger.exception("%s - invoke failed: %s", label, e)
        raise


def build_prompt() -> str:
    """Build a prompt for the LLM."""
    response = call_llm().invoke("What is tool calling in langchain?")
    print("\nResponse Content: ", response.content)
    return response.content if isinstance(response.content, str) else str(response.content)


@tool
def extract_video_id(url: str) -> str:
    """
    Extracts the 11-character YouTube video ID from a URL.
    
    Args:
        url (str): A YouTube URL containing a video ID.

    Returns:
        str: Extracted video ID or error message if parsing fails.
    """
    
    # Regex pattern to match video IDs
    pattern = r'(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else "Error: Invalid YouTube URL"


@tool
def fetch_transcript(video_id: str, language: str = "en") -> str:
    """
    Fetches the transcript of a YouTube video.
    
    Args:
        video_id (str): The YouTube video ID (e.g., "dQw4w9WgXcQ").
        language (str): Language code for the transcript (e.g., "en", "es").
    
    Returns:
        str: The transcript text or an error message.
    """
    
    # Try youtube_transcript_api first, then fall back to yt_dlp if needed
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        # Preferred API
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
            return " ".join([snippet.get("text", "") for snippet in transcript])
        # older versions might expose a different interface
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            # pick a matching language if available
            try:
                t = transcripts.find_transcript([language])
                data = t.fetch()
                return " ".join([s.get("text", "") for s in data])
            except Exception:
                pass
    except Exception as e:
        # Continue to yt_dlp fallback below
        logger.debug("fetch_transcript primary method failed: %s", e)

    # Fallback: try using yt_dlp to extract automatic captions/subtitles
    try:
        ydl_opts = {"quiet": True, "skip_download": True, "logger": yt_dpl_logger}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # automatic_captions or subtitles may contain entries per language
            captions = info.get("automatic_captions") or info.get("subtitles") or {}
            lang_map = captions.get(language) if isinstance(captions, dict) else None
            if lang_map:
                # pick the first available format
                fmt = lang_map[0]
                # try to download the subtitles as a string
                # yt_dlp can return 'requested_subtitles' or 'subtitles' with URLs
                # If no direct text available, return a notice with remote URL
                url_sub = fmt.get("url")
                if url_sub:
                    return f"[captions_url] {url_sub}"

            # As a last resort, try to get the description or chapters as context
            desc = info.get("description")
            if desc:
                return desc[:4000]

    except Exception as e:
        logger.debug("yt_dlp fallback for transcript failed: %s", e)

    return "Error: Transcript not available"


@tool
def search_youtube_details(query: str, max_results: int = 5):
    '''
    Searches YouTube for videos matching the query and returns a list of video details.
    
    Args:
        query (str): The search query string.
        max_results (int): Maximum number of results to return.
    
    Returns:
        List[Dict]: A list of dictionaries containing video title, ID, and URL.
    '''
    
    ydl_opts = {"quiet": True, "skip_download": True, 'logger': yt_dpl_logger}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
    entries = info.get("entries", []) or []
    return [
        {"title": e.get("title"), "video_id": e.get("id"), "url": e.get("webpage_url"), "views": e.get("view_count"), "duration": e.get("duration"), "channel": e.get("uploader"), "likes": e.get("like_count"), "comments": e.get("comment_count"), "chapters": e.get("chapters", [])}
        for e in entries
    ]           


@tool
def get_thumbnails(url: str) -> List[Dict]:
    """
    Get available thumbnails for a YouTube video using its URL.
    
    Args:
        url (str): YouTube video URL (any format)
        
    Returns:
        List of dictionaries with thumbnail URLs and resolutions in YouTube's native order
    """
    
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dpl_logger}) as ydl:
            info = ydl.extract_info(url, download=False)
            
            thumbnails = []
            for t in info.get('thumbnails', []):
                if 'url' in t:
                    thumbnails.append({
                        "url": t['url'],
                        "width": t.get('width'),
                        "height": t.get('height'),
                        "resolution": f"{t.get('width', '')}x{t.get('height', '')}".strip('x')
                    })
            
            return thumbnails

    except Exception as e:
        return [{"error": f"Failed to get thumbnails: {str(e)}"}]


tool_mapping = {
    "get_thumbnails" : get_thumbnails,
    "extract_video_id": extract_video_id,
    "fetch_transcript": fetch_transcript,
    "search_youtube_details": search_youtube_details
}

# Define the processing steps
def execute_tool(tool_call):
    """Execute single tool call and return ToolMessage"""
    try:
        # support dict-like or object tool_call
        if isinstance(tool_call, dict):
            name = tool_call.get("name")
            args = tool_call.get("args", {}) or {}
            call_id = tool_call.get("id")
        else:
            name = getattr(tool_call, "name", None)
            args = getattr(tool_call, "args", {}) or {}
            call_id = getattr(tool_call, "id", None)

        fn = tool_mapping.get(name)
        if fn is None:
            raise ValueError(f"Unknown tool: {name}")

        # call function with dict or sequence args
        if isinstance(args, dict):
            result = fn(**args)
        elif isinstance(args, (list, tuple)):
            result = fn(*args)
        else:
            result = fn(args)

        content = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
    except Exception as e:
        logger.exception("execute_tool failed for %s", tool_call)
        content = f"Error: {str(e)}"

    return ToolMessage(content=content, tool_call_id=call_id)


def process_tool_calls(messages):
    """Recursive tool call processor"""
    last_message = messages[-1]
    
    # Execute all tool calls in parallel
    tool_messages = [
        execute_tool(tc)
        for tc in getattr(last_message, 'tool_calls', [])
    ]

    # Add tool responses to message history
    updated_messages = messages + tool_messages

    # Use the agent (created in __main__) to get next response; log_and_invoke will normalize
    try:
        next_ai_response = log_and_invoke(agent, updated_messages, label="agent")
    except Exception:
        # fallback to direct agent.invoke with dict payload
        next_ai_response = agent.invoke({"messages": updated_messages})

    return updated_messages + [next_ai_response]


def should_continue(messages):
    """Check if you need another iteration"""
    last_message = messages[-1]
    return bool(getattr(last_message, 'tool_calls', None))


def _recursive_chain(messages):
    """Recursively process tool calls until completion"""
    if should_continue(messages):
        new_messages = process_tool_calls(messages)
        return _recursive_chain(new_messages)
    return messages

recursive_chain = RunnableLambda(_recursive_chain)

if __name__ == "__main__":
    print()
    agent = create_agent(
        model=call_llm(),
        tools=[extract_video_id, fetch_transcript, search_youtube_details, get_thumbnails],
        system_prompt="You are a helpful assistant that can extract YouTube video IDs, fetch transcripts, search for videos, and get thumbnails using the provided tools.",
    )
    universal_chain = (
        RunnableLambda(lambda x: [HumanMessage(content=x["query"])])
        | RunnableLambda(lambda messages: messages + [log_and_invoke(agent, messages, label="agent")])
        | recursive_chain
    )
    query_us = {"query": "Show top 3 US trending videos with metadata and thumbnails"}
    try:
        response = universal_chain.invoke(query_us)
        print("\nUS Trending Videos:\n", response[-1])
    except Exception as e:
        print("Non-critical network error while fetching US trending videos:", e)
