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

def generate_tts_audio_bytes(script: str) -> bytes | None:
    """Generates audio bytes from text using ElevenLabs TTS. Returns None on failure."""
    logging.info("Attempting to generate TTS audio bytes...")
    eleven_api_key = os.getenv("ELEVEN_API_KEY")
    if not eleven_api_key:
        logging.warning("ELEVEN_API_KEY not found in environment variables. Cannot generate TTS.")
        return None
    if not script:
        logging.warning("Empty script provided to TTS generation.")
        return None

    try:
        voice_id = "TX3LPaxmHKxFdv7VOQHJ" # Example Voice ID
        model_id = "eleven_multilingual_v2"
        timeout = httpx.Timeout(60.0, connect=5.0)
        headers = { "Accept": "audio/mpeg", "xi-api-key": eleven_api_key, "Content-Type": "application/json" }
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        payload = { "text": script, "model_id": model_id }

        with httpx.Client(timeout=timeout) as http_client:
            response = http_client.post(tts_url, headers=headers, json=payload)
            response.raise_for_status()
        audio_bytes = response.content
        logging.info("TTS audio bytes generated successfully.")
        return audio_bytes

    except httpx.HTTPStatusError as e:
        logging.error(f"ElevenLabs API request failed: {e.response.status_code} - {e.response.text}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"ElevenLabs TTS generation failed: {e}", exc_info=True)
        return None

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
        raise RuntimeError(f"Command execution failed: {cmd_str} Stderr: {e.stderr}") from e
    except Exception as e:
        logging.error(f"An unexpected error occurred while running command: {cmd_str}", exc_info=True)
        raise RuntimeError(f"Unexpected error during command execution: {cmd_str}") from e

def run_manim(code: str, scene_name: str) -> str:
    """Runs Manim on the given path and returns the path to the rendered MP4 clip."""
    with TemporaryDirectory(prefix=f"scene_name") as tmpdir:
        code_file = os.path.join(tmpdir, "scene.py")
        with open(code_file, "w") as f:
            f.write(code)
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
    return output_path

@mcp.tool()
def gen_viz(scene_name: str, code: str) -> dict:
    """Takes in the scene name and associated manim code, and returns a dictionary containing the path to the rendered MP4 clip and its duration."""
    logging.info(f"gen_viz tool started for scene: {scene_name}")
    output_path = run_manim(code, scene_name)
    duration = 0.0
    try:
        ffprobe_cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1",
            output_path # Use the final path after copying
        ]
        probe_result = run_subprocess(ffprobe_cmd, check=True)
        duration = float(probe_result.stdout.strip())
        logging.info(f"Detected video duration: {duration:.2f} seconds")
    except FileNotFoundError:
        logging.error("ffprobe command not found. Is ffmpeg installed and in PATH?")
    except (subprocess.CalledProcessError, RuntimeError) as e: # Catch errors from run_subprocess
        logging.error(f"ffprobe execution failed. See previous logs.")
    except ValueError:
            logging.error(f"Could not parse ffprobe duration output: {probe_result.stdout}")

    result_data = {
        "path": output_path,
        "duration": duration
    }

    try:
        video_resource = FileResource(
            uri=f"file://{output_path}",
            path=output_path,
            name="Generated Video",
            mime_type="video/mp4",
            tags={"generated", f"duration:{duration:.2f}s"}
        )
        mcp.add_resource(video_resource)
        logging.info(f"Added FileResource for: {output_path}")
        result_data["resource_uri"] = video_resource.uri 
    except NameError:
        logging.warning("FileResource is not defined or imported. Skipping resource addition.")
    except Exception as e:
        logging.error(f"Failed to create or add FileResource: {e}", exc_info=True)

    logging.info(f"gen_viz tool finished for scene: {scene_name}")
    return result_data

def generate_individual_audio_files(audio_scripts: list[str], tmpdir: str) -> list[str] | None:
    """Generates individual TTS audio files for each script in a temporary directory."""
    logging.info(f"Generating {len(audio_scripts)} individual audio files...")
    temp_audio_paths = []

    for i, script in enumerate(audio_scripts):
        logging.info(f"Generating audio for script {i+1}/{len(audio_scripts)}...")
        audio_bytes = generate_tts_audio_bytes(script)
        if audio_bytes:
            temp_audio_filename = f"audio_{i}.mp3"
            temp_audio_path = os.path.join(tmpdir, temp_audio_filename)
            try:
                with open(temp_audio_path, 'wb') as f:
                    f.write(audio_bytes)
                temp_audio_paths.append(temp_audio_path)
                logging.info(f"Saved temporary audio: {temp_audio_path}")
            except IOError as e:
                logging.error(f"Failed to save temporary audio file {temp_audio_path}: {e}")
                return None
        else:
            logging.error(f"TTS generation failed for script {i+1}.")
            return None

    logging.info(f"Successfully generated {len(temp_audio_paths)} audio files.")
    return temp_audio_paths


def concatenate_audio_files(temp_audio_paths: list[str], tmpdir: str) -> str | None:
    """Concatenates a list of audio files using ffmpeg."""
    if not temp_audio_paths:
        logging.warning("No audio paths provided for concatenation.")
        return None

    logging.info(f"Concatenating {len(temp_audio_paths)} audio files...")
    audio_list_file = os.path.join(tmpdir, "audio_list.txt")
    full_narration_path = os.path.join(tmpdir, "full_narration.mp3")
    try:
        with open(audio_list_file, 'w') as f:
            for path in temp_audio_paths:
                # Correctly escape paths for ffmpeg's concat demuxer format
                # Ensure path exists before adding? Or let ffmpeg fail?
                if not os.path.exists(path):
                     logging.warning(f"Audio file for concatenation not found: {path}. Skipping.")
                     continue
                safe_path_for_concat = path.replace("'", "'\\''")
                f.write(f"file '{safe_path_for_concat}'\\n") # Ensure newline character

        # Check if the list file actually contains any files after existence checks
        if os.path.getsize(audio_list_file) == 0:
            logging.warning("Audio list file is empty after checks. Cannot concatenate.")
            return None

        concat_audio_cmd = [
            "ffmpeg", "-y", # Overwrite output file if it exists
            "-f", "concat", "-safe", "0",
            "-i", audio_list_file, "-c", "copy", full_narration_path
        ]
        run_subprocess(concat_audio_cmd, check=True)
        logging.info(f"Concatenated audio saved to temporary file: {full_narration_path}")
        return full_narration_path
    except (RuntimeError, FileNotFoundError, IOError) as e:
        logging.error(f"Failed during audio concatenation: {e}", exc_info=True) # Log traceback
        return None
    except Exception as e:
        logging.error(f"Unexpected error during audio concatenation: {e}", exc_info=True)
        return None


def generate_and_concatenate_audio(audio_scripts: list[str], tmpdir: str) -> str | None:
    """Generates individual TTS audio files and concatenates them."""
    logging.info("Starting audio generation and concatenation...")

    # Generate individual audio files
    temp_audio_paths = generate_individual_audio_files(audio_scripts, tmpdir)

    if temp_audio_paths:
        full_narration_path = concatenate_audio_files(temp_audio_paths, tmpdir)
        if full_narration_path:
            logging.info("Audio generation and concatenation successful.")
            return full_narration_path
        else:
            logging.warning("Audio concatenation failed.")
            return None
    else:
        logging.warning("Audio generation failed, skipping concatenation.")
        return None

@mcp.tool()
def assemble_video(clip_paths: list[str], audio_paths: list[str]) -> str:
    """Assembles video clips and merges them with pre-generated audio files."""
    logging.info(f"assemble_video started with {len(clip_paths)} clips and {len(audio_paths)} audio files.")

    if not clip_paths:
        logging.error("No clip paths provided for assembly.")
        raise ValueError("No clip paths provided.")

    has_audio_files = bool(audio_paths)
    if has_audio_files and len(clip_paths) != len(audio_paths):
        logging.warning(f"Mismatch between number of clips ({len(clip_paths)}) and audio files ({len(audio_paths)}). Proceeding, but lengths usually match.")
        # Decide if this mismatch should halt execution or just proceed
        # For now, we proceed but log a warning.

    full_narration_path = None
    temp_dir_for_concat = None # Keep track of temp dir if created

    try:
        if has_audio_files:
            # Validate audio paths before attempting concatenation
            valid_audio_paths = []
            for path in audio_paths:
                if os.path.exists(path):
                    valid_audio_paths.append(path)
                else:
                    logging.warning(f"Audio file not found: {path}. Skipping.")
            
            if not valid_audio_paths:
                logging.warning("No valid audio files found after checking existence. Proceeding without audio.")
                has_audio_files = False # Force proceeding without audio
            else:
                # Create a temporary directory specifically for audio concatenation artifacts
                temp_dir_for_concat = TemporaryDirectory(prefix="audio_concat_")
                logging.info(f"Concatenating {len(valid_audio_paths)} audio files in {temp_dir_for_concat.name}")
                full_narration_path = concatenate_audio_files(valid_audio_paths, temp_dir_for_concat.name)
                if not full_narration_path:
                    logging.warning("Audio concatenation failed. Proceeding without audio overlay.")
                    has_audio_files = False # Force proceeding without audio
                else:
                    logging.info(f"Audio concatenation successful. Concatenated audio at: {full_narration_path}")

        # Use the concatenated audio path if successful, otherwise an empty string
        audio_path_for_merge = full_narration_path if has_audio_files and full_narration_path else ""
        
        if not audio_path_for_merge:
             logging.info("Proceeding to merge/concatenate video without audio overlay.")

        # merge_video_audio handles the actual merging/concatenation logic
        result_message = merge_video_audio(clip_paths=clip_paths, audio_path=audio_path_for_merge)
        logging.info(f"assemble_video finished successfully.")
        return result_message

    finally:
        # Clean up the temporary directory used for concatenation if it was created
        if temp_dir_for_concat:
            try:
                temp_dir_for_concat.cleanup()
                logging.info(f"Cleaned up temporary directory: {temp_dir_for_concat.name}")
            except Exception as e:
                logging.error(f"Error cleaning up temporary directory {temp_dir_for_concat.name}: {e}")

@mcp.tool()
def generate_audio(audio_script: str) -> str:
    """Generates audio from text using ElevenLabs TTS and saves it permanently. Expects plain text."""
    logging.info(f"generate_audio tool started.")
    audio_bytes = generate_tts_audio_bytes(audio_script)
    if not audio_bytes:
        raise RuntimeError("Failed to generate audio bytes.")
    
    audio_path = "" # Initialize to handle potential errors before assignment
    try:
        audio_filename = f"generated_audio_{uuid.uuid4()}.mp3"
        audio_path = os.path.join(static_videos, audio_filename) 
        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)
        logging.info(f"TTS audio saved permanently to: {audio_path}")        
        
        # Calculate duration using ffprobe
        duration = 0.0
        try:
            ffprobe_cmd = [
                "ffprobe", 
                "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path 
            ]
            probe_result = run_subprocess(ffprobe_cmd, check=True)
            duration = float(probe_result.stdout.strip())
            logging.info(f"Detected audio duration: {duration:.2f} seconds")
        except FileNotFoundError:
            logging.error("ffprobe command not found. Is ffmpeg installed and in PATH? Returning duration 0.")
        except (subprocess.CalledProcessError, RuntimeError) as e: # Catch errors from run_subprocess
            logging.error(f"ffprobe execution failed for audio duration. Returning duration 0.")
        except ValueError:
            logging.error(f"Could not parse ffprobe duration output: {probe_result.stdout}. Returning duration 0.")

        return {
            "path": audio_path,
            "duration": duration
        }
        # return audio_path # Original return
    except IOError as e:
         logging.error(f"Failed to save permanent audio file {audio_path}: {e}", exc_info=True)
         raise RuntimeError(f"Failed to save audio file: {e}") from e # Re-raise after logging
    except Exception as e:
        logging.error(f"Unexpected error saving audio file: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error saving audio: {e}") from e # Re-raise after logging

# @mcp.tool()
def merge_video_audio(clip_paths: list[str], audio_path: str) -> str:
    """Merges video clips using ffmpeg. Overlays audio if audio_path is valid."""
    logging.info(f"merge_video_audio started with {len(clip_paths)} clips.")

    # Validate input paths
    if not clip_paths:
        raise ValueError("No clip paths provided for merging.")
    for clip_path in clip_paths:
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Input video clip not found: {clip_path}")
    has_audio = audio_path and os.path.exists(audio_path)
    if not has_audio:
        logging.warning("No valid audio path provided. Merging video without audio.")

    with TemporaryDirectory(prefix="merge_") as tmpdir:
        # Create concat list file
        list_file_path = os.path.join(tmpdir, "concat_list.txt")
        with open(list_file_path, 'w') as f:
            for clip_path in clip_paths:
                # Escape single quotes for ffmpeg concat demuxer
                safe_clip_path = clip_path.replace("'", "'\\\\''")
                # Construct the line and write it with a standard newline
                file_line = f"file '{safe_clip_path}'"
                f.write(file_line + "\n")

        # Set up output path
        final_video_filename = f"merged_video_{uuid.uuid4()}.mp4"
        final_video_path = os.path.join(static_videos, final_video_filename)

        # Build ffmpeg command
        cmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file_path]
        if has_audio:
            cmd.extend([
                "-i", audio_path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k", "-shortest"
            ])
        else:
            cmd.extend(["-c", "copy"])
        cmd.append(final_video_path)

        # Execute merge
        run_subprocess(cmd, check=True)
        relative_url = f"/static/videos/{final_video_filename}"
        return f"Final video merged: {relative_url} (Audio {'included' if has_audio else 'not included'})"

if __name__ == "__main__":
    logging.info("main.py: Starting MCP server via mcp.run()")
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logging.exception("main.py: MCP server crashed!")
        raise
