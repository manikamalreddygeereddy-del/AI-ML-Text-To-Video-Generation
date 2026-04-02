import boto3
import json
import time
import re
import os
import uuid
import random
import botocore.exceptions
from concurrent.futures import ThreadPoolExecutor, as_completed

bedrock = boto3.client("bedrock-runtime")


def call_bedrock(prompt, max_token=3000, retries=10):
    for attempt in range(retries):
        try:
            response = bedrock.invoke_model(
                modelId="global.anthropic.claude-sonnet-4-20250514-v1:0",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_token,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                })
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"]
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                sleep_time = (2 ** attempt) + (random.random()/2)
                print(f"Throttled, retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
            else:
                raise
    raise Exception("Max retries exceeded for Bedrock invoke_model")


# ---------------------------
# 2. Extract Global Context
# ---------------------------
def extract_context(story):
    prompt = f"""
Extract cinematic context.

Return ONLY JSON:
{{
  "characters": [{{"name": "", "description": "<strictly within 10 words only.>"}}],
  "setting": "",
  "tone": ""
}}

Story:
{story}
"""
    output = call_bedrock(prompt)
    cleaned_output = output.replace("```json", "").replace("```", "").strip()
    print(output)
    return json.loads(cleaned_output)


# ---------------------------
# 3. Smart Chunking
# ---------------------------
def chunk_text(text, max_chars=80):
    chunks = []
    current = ""

    # Split by newlines OR periods followed by a space
    # The (?<=\.) keeps the period with the preceding sentence
    sentences = re.split(r'\n|(?<=\. )', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check if adding this sentence (plus a space) exceeds the limit
        if len(current) + len(sentence) + 1 <= max_chars:
            current += (sentence + " ")
        else:
            # If current is empty but one sentence is > max_chars,
            # we force it in to avoid infinite loops
            if current:
                chunks.append(current.strip())
            current = sentence + " "

    if current:
        chunks.append(current.strip())

    return chunks


# ---------------------------
# 4. Generate Scenes
# ---------------------------
def generate_scenes(chunk, context, prev_summary):

    prompt = f"""
    GLOBAL CONTEXT:
Characters: {json.dumps(context['characters'])}
Setting: {context['setting']}
Tone: {context['tone']}

PREVIOUS SUMMARY:
{prev_summary}

STORY TO PROCESS:
{chunk}

TASK:
Divide the "STORY TO PROCESS" into cinematic scenes. Do not add new plot points.

RULES:
- Maintain character consistency and story continuity.
- Strictly adhere to the "STORY TO PROCESS"; do not invent new events.
- **Important** Visuals: Focus 'visual_prompt' on cinematic lighting, environmental textures, and professional composition. Ensure all imagery is appropriate for a general, all-ages audience.
- Return ONLY JSON (no preamble or conversational text).

Each scene must include:
{{
  "scene_number": int,
  "title": string,
  "description": string <Strictly within 10 words length only>,
  "characters": list,
  "visual_prompt": string <Strictly within 10 words length only>,
  "narration": string,
  "camera_angle": string (one word),
  "mood": string (one word),
  "lighting": string (one word),
  "duration_estimate": number (seconds, based on narration length),
  "motion": string (one of: slow_zoom_in, slow_zoom_out, pan_left, pan_right),
  "transition": string (one of: fade, cut)
}}
"""

    output = call_bedrock(prompt, 5000)
    # print(output)
    cleaned_output_scenes = output.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned_output_scenes)


# ---------------------------
# 5. Summarize Scenes (Continuity)
# ---------------------------
def summarize_scenes(scenes):
    prompt = f"""
Summarize the following scenes in 3 lines:

{json.dumps(scenes)}
"""
    return call_bedrock(prompt, 2000)




def save_to_s3(data, bucket, key):
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2),
        ContentType='application/json'
    )
    print(f"Successfully saved output to s3://{bucket}/{key}")


def process_story(story_text, request_id=None):
    """Refactored logic from lambda_handler to be environment-agnostic"""
    if not request_id:
        request_id = os.getenv("REQUEST_ID")

    global_context = extract_context(story_text)
    hook = generate_hook(story_text, global_context)
    chunks = chunk_text(story_text)

    all_scenes_nested = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_index = {
            executor.submit(generate_scenes, chunk, global_context, ""): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                all_scenes_nested[index] = future.result()
            except Exception as exc:
                print(f'Chunk {index} failed: {exc}')

    all_scenes = []
    scene_counter = 1
    for scene_list in all_scenes_nested:
        if scene_list:
            for scene in scene_list:
                scene["scene_number"] = scene_counter
                all_scenes.append(scene)
                scene_counter += 1

    output = {
        "request_id": request_id,
        "context": global_context,
        "scenes": all_scenes
    }

    bucket_name = os.getenv("S3_BUCKET", "canyouimagine-video-assets")
    s3_key = f"outputs/{request_id}/scenes.json"
    save_to_s3(output, bucket_name, s3_key)

    return output


if __name__ == "__main__":
    # For ECR/Container: Read story from an Environment Variable or a test string
    test_story = os.getenv("STORY_INPUT", "A brave knight climbs a mountain to find a dragon.")
    print("Starting scene generation...")
    result = process_story(test_story)
    print(f"Done! Result saved to S3.")
