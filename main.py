from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import Message, UserMessage, AssistantMessage
from mcp.server.fastmcp.resources import FileResource
from tempfile import TemporaryDirectory
import os
import subprocess
import glob
import shutil
import uuid
import logging
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import save
import base64
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()
logging.info("main.py: Loaded environment variables from .env")

logging.info("main.py: Script started")

mcp = FastMCP("video-gen")
logging.info("main.py: FastMCP instance created")

static_videos = os.path.join(os.getcwd(), "static", "videos")
os.makedirs(static_videos, exist_ok=True)

# --- Helper Function for Subprocesses ---
def run_subprocess(cmd: list[str], cwd: str | None = None, check: bool = True, log_output_on_success: bool = False) -> subprocess.CompletedProcess:
    """Runs a subprocess command, captures output, logs, and handles errors."""
    cmd_str = ' '.join(cmd)
    logging.info(f"Running command: {cmd_str}" + (f" in {cwd}" if cwd else ""))
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check, # Let subprocess handle CalledProcessError if check is True
        )
        if log_output_on_success:
            logging.info(f"Command successful: {cmd_str}")
            logging.info(f"stdout: {result.stdout}")
            logging.info(f"stderr: {result.stderr}")
        elif result.returncode != 0: # Log output if it failed but check=False
             logging.warning(f"Command finished with non-zero exit code ({result.returncode}): {cmd_str}")
             logging.warning(f"stdout: {result.stdout}")
             logging.warning(f"stderr: {result.stderr}")

        return result
    except FileNotFoundError as e:
        logging.error(f"Command not found: {cmd[0]}. Ensure it is installed and in PATH.")
        raise FileNotFoundError(f"Command not found: {cmd[0]}") from e
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with return code {e.returncode}: {cmd_str}")
        logging.error(f"stdout: {e.stdout}")
        logging.error(f"stderr: {e.stderr}")
        # Re-raise a more generic error or let the specific one propagate if needed elsewhere
        raise RuntimeError(f"Command execution failed: {cmd_str} Stderr: {e.stderr}") from e
    except Exception as e: # Catch other potential exceptions during run
        logging.error(f"An unexpected error occurred while running command: {cmd_str}", exc_info=True)
        raise RuntimeError(f"Unexpected error during command execution: {cmd_str}") from e

# --- End Helper Function ---

@mcp.prompt()
def storyboard_prompt(topic: str, num_scenes: int = 3) -> str:
    logging.info(f"Defining storyboard_prompt")
    """Asks to generate a storyboard given a certain topic and number of scenes."""
    system_instructions = (
        f"You are a teaching assistant that outputs a 'storyboard' as valid JSON. "
        f"Given a topic, produce exactly {num_scenes} scenes, each with a title and a short visual description."
    )
    return f"{system_instructions}\n\nTopic: {topic}"

@mcp.prompt()
def code_prompt(scene_index: int, scenes: dict) -> str:
    logging.info(f"Defining code_prompt")
    """Given a scene index and a scene dictionary, which has a 'title' and 'description' property, returns a Manim class to visualize the scene."""

    # Simplified direct instruction
    instruction = (
        f"Generate Python code for a Manim scene class based on the following details for scene {scene_index}:\n\n"
        f"Title: {scenes['title']}\n"
        f"Description: {scenes['description']}\n\n"
        f"Output only the Python code block for the Manim class, nothing else."
    )
    
    return instruction

@mcp.prompt()
def timed_script_prompt(scenes_with_durations: list[dict]) -> str:
    logging.info(f"Defining timed_script_prompt for {len(scenes_with_durations)} scenes")
    """Generates a script prompt asking an LLM to create narration tailored to scene durations."""
    
    WORDS_PER_SECOND = 2.5  # Adjust as needed for desired pace
    PAUSE_BUDGET_SECONDS = 3  # Buffer/pause time per scene

    prompt_parts = [
        "You are a script writer creating narration for an educational video.",
        "The video consists of several scenes, each with a specific duration.",
        "Generate a single, coherent narration script that covers all the scenes below.",
        f"Aim for a conversational pace (around {WORDS_PER_SECOND} words per second).",
        f"For each scene, write narration based on its title and description, but strictly adhere to the target word count calculated from its duration minus a {PAUSE_BUDGET_SECONDS}-second buffer.",
        "Ensure smooth transitions between scenes. Output only the final narration script as a single block of text.",
        "\n---",
        "Scenes Details:",
        "---"
    ]

    total_target_words = 0
    for i, scene in enumerate(scenes_with_durations):
        title = scene.get("title", "Untitled Scene")
        description = scene.get("description", "No description.")
        video_duration = scene.get("video_duration", 0.0)

        target_narration_duration = max(0, video_duration - PAUSE_BUDGET_SECONDS)
        target_word_count = int(target_narration_duration * WORDS_PER_SECOND)
        total_target_words += target_word_count

        prompt_parts.append(f"\nScene {i+1}:")
        prompt_parts.append(f"  Title: {title}")
        prompt_parts.append(f"  Description: {description}")
        prompt_parts.append(f"  Video Duration: {video_duration:.1f}s")
        prompt_parts.append(f"  Target Word Count for Narration: ~{target_word_count} words")

    prompt_parts.append("\n---")
    prompt_parts.append(f"Total Estimated Word Count for Script: ~{total_target_words} words")
    prompt_parts.append("Now, please write the narration script:")
    prompt_parts.append("---")

    final_prompt = "\n".join(prompt_parts)
    # Format the log message separately to avoid issues with f-string parsing
    log_message = f"Generated timed script prompt:\n{final_prompt}"
    logging.info(log_message) # Log the formatted message
    return final_prompt

@mcp.tool()
def gen_viz(scene_name: str, code: str) -> dict:
    """Takes in the scene name and associated manim code, and returns a dictionary containing the path to the rendered MP4 clip and its duration."""
    logging.info(f"gen_viz tool started for scene: {scene_name}")
    with TemporaryDirectory(prefix=f"scene_name") as tmpdir:
        code_file = os.path.join(tmpdir, "scene.py")
        with open(code_file, "w") as f:
            f.write(code)
        # build the docker command to run Manim
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{tmpdir}:/workspace",
            "--workdir", "/workspace",
            "manimcommunity/manim:stable",
            "manim",
            "--silent", "-ql",
            "scene.py", scene_name
        ]
        logging.info(f"Running docker command: {' '.join(cmd)}")
        result = run_subprocess(cmd, cwd=tmpdir, check=True, log_output_on_success=True)
        # No explicit return code check needed here as run_subprocess(check=True) handles it

        #find output mp4
        media_dir = os.path.join(tmpdir, "media", "videos")
        pattern = os.path.join(media_dir, "*", "*", f"{scene_name}.mp4")
        logging.info(f"Attempting to find video file with pattern: {pattern}")
        files = glob.glob(pattern)
        if not files:
            logging.error(f"Rendered file not found matching pattern: {pattern}")
            raise FileNotFoundError("Rendered file not found.")
        video_path = files[0] # find the first generated file
        logging.info(f"Found video file: {video_path}")
        # copy the rendered video to the static dir
        output_path = os.path.join(static_videos, f"{scene_name}_{uuid.uuid4()}.mp4")
        logging.info(f"Copying video to: {output_path}")
        shutil.copy(video_path, output_path)

        # --- Get Video Duration using ffprobe --- 
        duration = 0.0
        try:
            ffprobe_cmd = [
                "ffprobe", 
                "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1",
                output_path # Use the final path after copying
            ]
            # Use the helper function
            probe_result = run_subprocess(ffprobe_cmd, check=True)
            duration = float(probe_result.stdout.strip())
            logging.info(f"Detected video duration: {duration:.2f} seconds")
        except FileNotFoundError:
            # run_subprocess already logs the FileNotFoundError
            logging.error("ffprobe command not found. Is ffmpeg installed and in PATH?")
            # Decide how to handle: return 0, raise error, etc.
            # Returning 0 duration for now, but assembly might be odd.
        except (subprocess.CalledProcessError, RuntimeError) as e: # Catch errors from run_subprocess
            # run_subprocess already logs the error details
            logging.error(f"ffprobe execution failed. See previous logs.")
            # Returning 0 duration for now.
        except ValueError:
             logging.error(f"Could not parse ffprobe duration output: {probe_result.stdout}")
             # Returning 0 duration for now.
        # --- End Get Duration --- 

        # Prepare the return dictionary
        result_data = {
            "path": output_path,
            "duration": duration
        }

        # Add resource using FileResource (if needed and configured)
        try:
            video_resource = FileResource(
                uri=f"file://{output_path}", # Removed .as_posix() based on user edit
                path=output_path,
                name="Generated Video",
                mime_type="video/mp4",
                tags={"generated", f"duration:{duration:.2f}s"} # Add duration tag
            )
            mcp.add_resource(video_resource)
            logging.info(f"Added FileResource for: {output_path}")
            # Optionally, add the resource URI to the result data
            result_data["resource_uri"] = video_resource.uri 
        except NameError:
            logging.warning("FileResource is not defined or imported. Skipping resource addition.")
        except Exception as e:
            logging.error(f"Failed to create or add FileResource: {e}", exc_info=True)

        logging.info(f"gen_viz tool finished for scene: {scene_name}")
        # Return the dictionary instead of just the path string
        # Ensure the calling code expects a dictionary now!
        return result_data 

@mcp.tool()
def assemble_video(clip_paths: list[str], audio_script: str) -> str:
    """Assembles video clips and adds narration using ElevenLabs TTS. (Calls API each time)"""
    logging.info(f"assemble_video started with {len(clip_paths)} clips.")

    # --- Input Type Handling ---
    # Check if audio_script needs conversion from list/dict to JSON string
    audio_script_str = audio_script 
    try:
        parsed_script = json.loads(audio_script_str)
        if isinstance(parsed_script, list) and all(isinstance(item, dict) and 'text' in item for item in parsed_script):
            logging.info("Detected timestamped JSON format in audio_script string. Extracting text.")
            plain_text_parts = [item.get('text', '') for item in parsed_script]
            audio_script_str = " ".join(plain_text_parts)
            logging.info("Converted timestamped JSON string to plain text for TTS.")
    except json.JSONDecodeError:
        logging.info("audio_script is a non-JSON string, assuming plain text.")
    except Exception as e:
        logging.warning(f"Error while checking/converting audio_script string format: {e}. Using as is.")
    # --- End Input Type Handling ---

    audio_bytes = None
    eleven_api_key = os.getenv("ELEVEN_API_KEY")

    if not eleven_api_key:
        logging.warning("ELEVEN_API_KEY not found in environment variables. Skipping TTS.")
    elif not audio_script_str: # Use the processed string variable
        logging.warning("Audio script is empty after processing. Skipping TTS.")
    else:
        try:
            logging.info("Generating TTS audio using ElevenLabs...")
            client = ElevenLabs(
                api_key=eleven_api_key,
            )
            voice_id = "TX3LPaxmHKxFdv7VOQHJ" # Default voice ID (Corrected)
            model_id = "eleven_multilingual_v2" # Default model
            payload = { "text": audio_script_str, "model_id": model_id }
            tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = { "Accept": "audio/mpeg", "xi-api-key": eleven_api_key, "Content-Type": "application/json" }
            timeout = httpx.Timeout(60.0, connect=5.0)
            with httpx.Client(timeout=timeout) as http_client:
                 response = http_client.post(tts_url, headers=headers, json=payload)
                 response.raise_for_status() 
            audio_bytes = response.content
            logging.info("TTS audio generated successfully from standard endpoint.")
        except httpx.HTTPStatusError as e:
            logging.error(f"ElevenLabs API request failed: {e.response.status_code} - {e.response.text}", exc_info=True)
            logging.warning("Proceeding with video assembly without narration due to TTS error.")
            audio_bytes = None
        except Exception as e:
            logging.error(f"ElevenLabs TTS generation/decoding failed: {e}", exc_info=True)
            logging.warning("Proceeding with video assembly without narration due to TTS error.")
            audio_bytes = None

    if not clip_paths:
        logging.error("No clip paths provided for assembly.")
        raise ValueError("No clip paths provided for assembly.")

    with TemporaryDirectory(prefix="assembly_") as tmpdir:
        audio_path = None
        if audio_bytes:
            try:
                audio_filename = f"narration_{uuid.uuid4()}.mp3" 
                audio_path = os.path.join(tmpdir, audio_filename)
                with open(audio_path, 'wb') as f: f.write(audio_bytes)
                logging.info(f"TTS audio saved to temporary file: {audio_path}")
            except Exception as e:
                logging.error(f"Failed to save TTS audio: {e}", exc_info=True)
                audio_path = None
        
        list_file_path = os.path.join(tmpdir, "concat_list.txt")
        with open(list_file_path, 'w') as f:
            for clip_path in clip_paths:
                safe_path = clip_path.replace("'", "'\\''")
                f.write(f"file '{safe_path}'\\n")
        logging.info(f"Generated concatenation list file: {list_file_path}")

        final_video_filename = f"final_video_{uuid.uuid4()}.mp4"
        final_video_path = os.path.join(static_videos, final_video_filename)
        logging.info(f"Final video will be saved to: {final_video_path}")

        cmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file_path]
        if audio_path and os.path.exists(audio_path):
            logging.info(f"Adding audio overlay from: {audio_path}")
            cmd.extend([
                "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", 
                "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-shortest",
            ])
        else:
             logging.warning("No audio file provided or found. Assembling video without narration.")
             cmd.extend(["-c", "copy"]) # Use stream copy if no audio

        cmd.append(final_video_path)
        # Use the helper function
        try:
            result = run_subprocess(cmd, check=True)
            # run_subprocess handles logging the command and basic success/failure
            # Log additional context specific info if needed
            logging.info(f"Video assembly successful: {final_video_path}")
            relative_url = f"/static/videos/{final_video_filename}"
            return f"Final video assembled: {relative_url}"
        except (RuntimeError, FileNotFoundError) as e:
            # Error is already logged by run_subprocess
            # Re-raise or handle specific cleanup if necessary
            raise e # Re-raise the error caught by run_subprocess

@mcp.tool()
def generate_audio(audio_script: str) -> str:
    """Generates audio from text using ElevenLabs TTS and saves it permanently. Expects plain text."""
    logging.info(f"generate_audio started.")

    # --- REMOVED Complex Input Type Handling --- 
    # Assume audio_script is a valid plain string due to type hint & validation
    audio_script_str = audio_script

    eleven_api_key = os.getenv("ELEVEN_API_KEY")
    if not eleven_api_key:
        logging.error("ELEVEN_API_KEY not found in environment variables.")
        raise ValueError("ELEVEN_API_KEY not found.")
    if not audio_script_str:
        logging.error("Audio script is empty.") # Simplified message
        raise ValueError("Audio script is empty.")

    try:
        logging.info("Generating TTS audio using ElevenLabs...")
        voice_id = "TX3LPaxmHKxFdv7VOQHJ" # Default voice ID (Corrected)
        model_id = "eleven_multilingual_v2" # Default model
        payload = { "text": audio_script_str, "model_id": model_id } # Use the input string directly
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = { "Accept": "audio/mpeg", "xi-api-key": eleven_api_key, "Content-Type": "application/json" }
        timeout = httpx.Timeout(60.0, connect=5.0)
        
        with httpx.Client(timeout=timeout) as http_client:
             response = http_client.post(tts_url, headers=headers, json=payload)
             response.raise_for_status()
        audio_bytes = response.content
        logging.info("TTS audio generated successfully.")

        # Save audio permanently
        audio_filename = f"generated_audio_{uuid.uuid4()}.mp3"
        audio_path = os.path.join(static_videos, audio_filename) # Save in static dir
        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)
        logging.info(f"TTS audio saved permanently to: {audio_path}")
        
        # Return the path to the saved audio
        return audio_path

    except httpx.HTTPStatusError as e:
        logging.error(f"ElevenLabs API request failed: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise RuntimeError(f"ElevenLabs API request failed: {e.response.text}") from e
    except Exception as e:
        logging.error(f"ElevenLabs TTS generation or saving failed: {e}", exc_info=True)
        raise RuntimeError(f"TTS generation/saving failed: {e}") from e

@mcp.tool()
def merge_video_audio(clip_paths: list[str], audio_path: str) -> str:
    """Merges video clips using ffmpeg. Optionally overlays audio if audio_path is non-empty and valid."""
    logging.info(f"merge_video_audio started with {len(clip_paths)} clips.")
    logging.info(f"Provided audio_path: '{audio_path}'")

    if not clip_paths:
        logging.error("No clip paths provided for merging.")
        raise ValueError("No clip paths provided for merging.")

    # Validate audio path only if it's not empty
    valid_audio_path = None
    if audio_path:
        if os.path.exists(audio_path):
            valid_audio_path = audio_path
            logging.info(f"Audio file found: {valid_audio_path}")
        else:
            logging.warning(f"Provided audio file not found: {audio_path}. Merging video without audio.")
    else:
        logging.info("Empty audio_path provided, proceeding without audio overlay.")
            
    with TemporaryDirectory(prefix="merge_") as tmpdir:
        # Create ffmpeg concatenation list
        list_file_path = os.path.join(tmpdir, "concat_list.txt")
        with open(list_file_path, 'w') as f:
            for clip_path in clip_paths:
                if not os.path.exists(clip_path):
                     logging.error(f"Input video clip not found: {clip_path}")
                     raise FileNotFoundError(f"Input video clip not found: {clip_path}")
                safe_path = clip_path.replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")
        logging.info(f"Generated concatenation list file: {list_file_path}")

        # Define final output path
        final_video_filename = f"merged_video_{uuid.uuid4()}.mp4"
        final_video_path = os.path.join(static_videos, final_video_filename)
        logging.info(f"Final merged video will be saved to: {final_video_path}")

        # Construct ffmpeg command
        cmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file_path]
        if valid_audio_path:
            logging.info(f"Adding audio overlay from: {valid_audio_path}")
            cmd.extend([
                "-i", valid_audio_path, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
                "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-shortest",
            ])
        else:
             logging.info("Merging video clips without adding new audio.")
             cmd.extend(["-c", "copy"]) # Use stream copy if no audio

        cmd.append(final_video_path)
        # Use the helper function
        try:
            result = run_subprocess(cmd, check=True)
             # run_subprocess handles logging the command and basic success/failure
             # Log additional context specific info if needed
            logging.info(f"Video merge successful: {final_video_path}")
            relative_url = f"/static/videos/{final_video_filename}"
            return f"Final video merged: {relative_url} (Audio {'included' if valid_audio_path else 'not included'})"
        except (RuntimeError, FileNotFoundError) as e:
            # Error is already logged by run_subprocess
            # Re-raise or handle specific cleanup if necessary
            raise e # Re-raise the error caught by run_subprocess

if __name__ == "__main__":
    logging.info("main.py: Starting MCP server via mcp.run()")
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logging.exception("main.py: MCP server crashed!")
        raise
