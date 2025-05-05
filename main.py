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
            cmd, # Always pass the command list directly
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check, # Let subprocess handle CalledProcessError if check is True
            shell=False
        )
        # Always log stderr on success if it's not empty, as it might contain warnings
        if result.returncode == 0 and result.stderr:
             logging.info(f"Command successful with warnings/info in stderr: {cmd_str}")
             logging.info(f"stdout: {result.stdout}")
             logging.info(f"stderr: {result.stderr}")
        elif log_output_on_success:
            logging.info(f"Command successful: {cmd_str}")
            logging.info(f"stdout: {result.stdout}")
        elif result.returncode != 0: # Log output if it failed (check=False or check=True)
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
    """Concatenates a list of audio files using ffmpeg (concat filter method)."""
    if not temp_audio_paths:
        logging.warning("No audio paths provided for concatenation.")
        return None

    valid_paths = [p for p in temp_audio_paths if os.path.exists(p)]
    if len(valid_paths) < len(temp_audio_paths):
        logging.warning(f"Found {len(valid_paths)} existing audio files out of {len(temp_audio_paths)} provided.")
    if not valid_paths:
        logging.error("No valid audio files found to concatenate.")
        return None

    num_files = len(valid_paths)
    logging.info(f"Concatenating {num_files} audio files using concat filter...")
    full_narration_path = os.path.join(tmpdir, "full_narration.m4a") # New filename using .m4a extension

    try:
        cmd = ["ffmpeg", "-y"] # Start command

        # Add each valid path as an input
        for path in valid_paths:
            cmd.extend(["-i", path])

        # Construct the filtergraph string
        # e.g., "[0:a][1:a][2:a]concat=n=3:v=0:a=1[aout]"
        filter_streams = "".join([f"[{i}:a]" for i in range(num_files)])
        filtergraph = f"{filter_streams}concat=n={num_files}:v=0:a=1[aout]"

        cmd.extend([
            "-filter_complex", filtergraph,
            "-map", "[aout]", # Map the filter output
            # Choose output codec - AAC is generally robust
            "-c:a", "aac", "-b:a", "192k",
            full_narration_path
        ])

        logging.info(f"Running audio concatenation command: {' '.join(cmd)}")
        run_subprocess(cmd, check=True)
        logging.info(f"Concatenated audio saved to temporary file: {full_narration_path}")
        return full_narration_path
    except (RuntimeError, FileNotFoundError, IOError) as e:
        logging.error(f"Failed during audio concatenation (filter method): {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"Unexpected error during audio concatenation (filter method): {e}", exc_info=True)
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
    """Merges each video clip with its corresponding audio file, then concatenates the results."""
    logging.info(f"assemble_video started with {len(clip_paths)} clips and {len(audio_paths)} audio files for pairwise merging.")

    if not clip_paths:
        logging.error("No clip paths provided for assembly.")
        raise ValueError("No clip paths provided.")

    # If no audio is provided, just concatenate the original clips
    if not audio_paths:
        logging.warning("No audio paths provided. Concatenating video clips without merging audio.")
        return concatenate_videos(clip_paths)

    # If audio is provided, number of clips and audio files must match
    if len(clip_paths) != len(audio_paths):
        logging.error(f"Mismatch between number of clips ({len(clip_paths)}) and audio files ({len(audio_paths)}). Cannot perform pairwise merge.")
        raise ValueError("Number of clips and audio files must match for pairwise merging.")

    intermediate_clip_paths = []
    # Use a single temporary directory for all intermediate merged clips
    with TemporaryDirectory(prefix="intermediate_clips_") as tmpdir:
        logging.info(f"Created temporary directory for intermediate clips: {tmpdir}")
        try:
            for i, (clip_path, audio_path) in enumerate(zip(clip_paths, audio_paths)):
                logging.info(f"Merging clip {i+1}/{len(clip_paths)} ({os.path.basename(clip_path)}) with audio ({os.path.basename(audio_path)})...")
                
                # Validate individual paths
                if not os.path.exists(clip_path):
                    logging.error(f"Input video clip not found: {clip_path}")
                    raise FileNotFoundError(f"Input video clip not found: {clip_path}")
                if not os.path.exists(audio_path):
                    logging.error(f"Input audio file not found: {audio_path}")
                    raise FileNotFoundError(f"Input audio file not found: {audio_path}")

                intermediate_output_path = os.path.join(tmpdir, f"intermediate_{i}.mp4")

                # Probe video duration
                try:
                    probe_cmd = [
                        "ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        clip_path
                    ]
                    probe_result = run_subprocess(probe_cmd, check=True)
                    video_duration = float(probe_result.stdout.strip())
                    logging.info(f"Detected video duration: {video_duration:.2f} seconds")
                except Exception as e:
                    logging.warning(f"Could not determine video duration: {e}. Skipping audio padding.")
                    video_duration = None

                # Build ffmpeg command for merging individual pair with optional audio padding
                if video_duration:
                    merge_cmd = [
                        "ffmpeg", "-y",
                        "-i", clip_path,
                        "-i", audio_path,
                        "-filter_complex", f"[1:a]apad,atrim=duration={video_duration}[aud]",
                        "-map", "0:v:0",
                        "-map", "[aud]",
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        intermediate_output_path
                    ]
                else:
                    merge_cmd = [
                        "ffmpeg", "-y",
                        "-i", clip_path,
                        "-i", audio_path,
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        intermediate_output_path
                    ]

                logging.info(f"Running intermediate merge command: {' '.join(merge_cmd)}")
                run_subprocess(merge_cmd, check=True, log_output_on_success=True)
                intermediate_clip_paths.append(intermediate_output_path)
                logging.info(f"Intermediate clip created: {intermediate_output_path}")

            # After loop, check if we successfully created intermediate clips
            if not intermediate_clip_paths:
                logging.error("No intermediate clips were successfully created.")
                raise RuntimeError("Failed to create any intermediate video clips.")

            # Concatenate the intermediate clips
            logging.info(f"All intermediate clips created. Concatenating {len(intermediate_clip_paths)} clips...")
            final_result_message = concatenate_videos(intermediate_clip_paths)
            logging.info(f"assemble_video finished successfully.")
            return final_result_message
        
        except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f"Error during pairwise merging or final concatenation: {e}", exc_info=True)
            # Return the error message possibly generated by concatenate_videos or raise a new one
            if isinstance(e, subprocess.CalledProcessError):
                 return f"Tool: assemble_video, Result: Failed during processing. Error: {getattr(e, 'stderr', str(e))}"
            else:
                 raise RuntimeError(f"Assemble video failed: {e}") from e
        # Temporary directory tmpdir is automatically cleaned up here

def concatenate_videos(input_clip_paths: list[str]) -> str:
    """Concatenates video clips using ffmpeg's concat demuxer. Assumes clips contain necessary streams (video, audio)."""
    logging.info(f"concatenate_videos started with {len(input_clip_paths)} clips.")

    # Validate input clip paths
    if not input_clip_paths:
        raise ValueError("No clip paths provided for concatenation.")
    for clip_path in input_clip_paths:
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Input video clip not found: {clip_path}")

    # Use a temporary directory for the video list file
    with TemporaryDirectory(prefix="concat_videos_") as tmpdir:
        logging.info(f"Created temporary directory for video concatenation list: {tmpdir}")
        
        # Create concat list file for video clips
        list_file_path = os.path.join(tmpdir, "concat_list.txt")
        with open(list_file_path, 'w') as f:
            for clip_path in input_clip_paths:
                # Using abspath and simple replace for safety
                abs_clip_path = os.path.abspath(clip_path)
                safe_clip_path = abs_clip_path.replace("'", "'\\''") # Escape single quotes
                file_line = f"file '{safe_clip_path}'"
                f.write(file_line + "\n") # Use standard newline
        logging.info(f"Video concatenation list created at: {list_file_path}")

        # Set up output path (in permanent static_videos)
        final_video_filename = f"merged_video_{uuid.uuid4()}.mp4"
        final_video_path = os.path.join(static_videos, final_video_filename)

        # Build ffmpeg command - Use stream copy to preserve timestamps
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file_path,
            "-c", "copy",
            final_video_path
        ]
        
        logging.info(f"Running final concatenation command (stream copy): {' '.join(cmd)}")
        try:
            run_subprocess(cmd, check=True, log_output_on_success=True) 
            logging.info(f"Final concatenated video saved to: {final_video_path}")
        except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.error(f"Final video concatenation failed: {e}", exc_info=True)
            # Clean up potentially partially created output file on error
            if os.path.exists(final_video_path):
                try:
                    os.remove(final_video_path)
                    logging.info(f"Removed incomplete output file: {final_video_path}")
                except OSError as remove_err:
                    logging.error(f"Failed to remove incomplete output file {final_video_path}: {remove_err}")
            return f"Tool: assemble_video, Result: Failed to concatenate video clips. Error: {getattr(e, 'stderr', str(e))}" 

        # Return success message (no longer mentions audio explicitly, as it's assumed part of input clips)
        relative_final_path = os.path.relpath(final_video_path, os.getcwd()) 
        return f"Tool: assemble_video, Result: Final video assembled: /{relative_final_path.replace(os.sep, '/')}"

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

if __name__ == "__main__":
    logging.info("main.py: Starting MCP server via mcp.run()")
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logging.exception("main.py: MCP server crashed!")
        raise
