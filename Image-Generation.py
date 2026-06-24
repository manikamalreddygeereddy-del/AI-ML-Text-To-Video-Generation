import boto3
import json
import base64
import os
import time
import random
import requests

# NVIDIA API Configuration
NVIDIA_IMAGE_TO_IMAGE_API_KEY = os.environ.get("NVIDIA_IMAGE_TO_IMAGE_API_KEY", "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")
NVIDIA_TEXT_TO_IMAGE_API_KEY = os.environ.get("NVIDIA_TEXT_TO_IMAGE_API_KEY", "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")

# Different models for different purposes
NVIDIA_TEXT_TO_IMAGE_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux-1-1-ultra"
NVIDIA_IMAGE_TO_IMAGE_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev"

# Initialize clients
s3 = boto3.client("s3")
bedrock_claude = boto3.client(service_name="bedrock-runtime")

def generate_image_nvidia(prompt, reference_image=None, aspect_ratio="16:9", steps=30, cfg_scale=3.5):
    """
    Generate image using NVIDIA API

    Args:
        prompt: Text prompt for image generation
        reference_image: Base64 encoded image for image-to-image (optional)
        aspect_ratio: Aspect ratio of output image
        steps: Number of generation steps
        cfg_scale: Configuration scale

    Returns:
        Base64 encoded image string
    """
    # Choose the appropriate endpoint and API key based on whether we have a reference image
    if reference_image:
        # Image-to-image model (requires reference image)
        api_url = NVIDIA_IMAGE_TO_IMAGE_URL
        api_key = NVIDIA_IMAGE_TO_IMAGE_API_KEY
        payload = {
            "prompt": prompt,
            "image": f"data:image/png;base64,{reference_image}",
            "aspect_ratio": aspect_ratio,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": 0
        }
    else:
        # Text-to-image model
        api_url = NVIDIA_TEXT_TO_IMAGE_URL
        api_key = NVIDIA_TEXT_TO_IMAGE_API_KEY
        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "seed": 0
        }
        # Note: Flux-1-1-ultra doesn't support steps and cfg_scale parameters

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()
    response_body = response.json()

    # Extract the base64 image from response
    if "image" in response_body:
        # Remove data URI prefix if present
        image_data = response_body["image"]
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]
        return image_data
    elif "images" in response_body and len(response_body["images"]) > 0:
        image_data = response_body["images"][0]
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]
        return image_data
    else:
        raise ValueError(f"Unexpected response format: {response_body}")


def rewrite_prompt_with_claude(original_prompt):
    """Use Claude to sanitize the prompt for Image Generator"""
    rewrite_instruction = f"""
    Rewrite the following image generation prompt to ensure it is completely safe,
    non-violent, non-sensitive, and compliant with AI content policies.
    Keep the meaning but remove or soften anything that could trigger filters.
    Only and only return the rewritten prompt.
    **Important** There should be only less than 500 characters in the prompt.

    Prompt:
    {original_prompt}
    """

    response = bedrock_claude.invoke_model(
        modelId="global.anthropic.claude-sonnet-4-20250514-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": rewrite_instruction}]
        })
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def process_video_generation():
    # 1. Environment Configuration (Set these in your ECS Task Definition)
    source_bucket = os.environ.get("SOURCE_BUCKET", "canyouimagine-video-assets")
    target_bucket = os.environ.get("TARGET_BUCKET", "canyouimagine-video-images")
    event_str = os.getenv("EVENT_DATA", "{}")
    event = json.loads(event_str)

    s3_key = event.get("scene_output", {}).get("s3_key")
    if not s3_key:
        raise ValueError("No s3_key found in input event")


    if not s3_key:
            print("Error: No S3_KEY environment variable provided.")
            return

    character_cache = {}

    # 2. Load the original JSON from S3
    print(f"Fetching input from: {s3_key}")
    response = s3.get_object(Bucket=source_bucket, Key=s3_key)
    data = json.loads(response['Body'].read())
    request_id = data["request_id"]


    for character in data["context"]["characters"]:
        character_name = character["name"]
        character_description = character["description"]

        character_prompt = f"""
        Character reference portrait.

        Name:
        {character_name}

        Description:
        {character_description}

        Full body character.

        Front facing.

        Neutral pose.

        Consistent facial features.

        Consistent hairstyle.

        Consistent clothing.

        Single character only.

        Simple clean background.

        High quality cinematic illustration.
        """

        max_retries = 10
        image_base64 = None
        current_prompt = character_prompt

        for attempt in range(max_retries):
            try:
                image_base64 = generate_image_nvidia(
                    prompt=current_prompt,
                    aspect_ratio="1:1",
                    steps=30,
                    cfg_scale=3.5
                )
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400:
                    # Content filter or validation error
                    print(f"⚠️ Content filter triggered. Rewriting with Claude...{current_prompt}")
                    print(f"Response: {e.response.text}")
                    safe_prompt = rewrite_prompt_with_claude(current_prompt)
                    current_prompt = safe_prompt

                elif e.response.status_code == 429:
                    # Rate limiting
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⏳ Throttled. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)

                else:
                    print(f"❌ HTTP Error {e.response.status_code}: {e.response.text}")
                    raise e
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                raise e

        if not image_base64:
            raise RuntimeError(f"Failed to generate character image after {max_retries} attempts")

        image_bytes = base64.b64decode(image_base64)

        character_key = (
            f"outputs/{request_id}/characters/"
            f"{character_name.lower().replace(' ', '_')}.png"
        )

        s3.put_object(
            Bucket=target_bucket,
            Key=character_key,
            Body=image_bytes,
            ContentType="image/png"
        )

        character_cache[character_name] = {
            "base64": image_base64,
            "s3_path": f"s3://{target_bucket}/{character_key}"
        }

    data["character_references"] = {}

    for character_name, value in character_cache.items():
        data["character_references"][character_name] = {
            "image_s3_path": value["s3_path"]
        }

    # 3. Iterate through scenes
    for scene in data["scenes"]:

        scene_characters = scene.get("characters", [])

        character_reference_text = ""

        for char_name in scene_characters:
            for character in data["context"]["characters"]:
                if character["name"] == char_name:
                    character_reference_text += f"""
                    Character:
                    {character['name']}
                    {character['description']}
                    """
        consistent_prompt = f"""
        Story Setting:
        {data['context']['setting']}

        Story Tone:
        {data['context']['tone']}

        Characters:
        {character_reference_text}

        IMPORTANT:

        The reference image is ONLY for character identity.

        Create a completely new scene.

        Create a new pose.

        Create a new camera composition.

        Create a new environment.

        Create a new background.

        Preserve only:

        - face
        - hairstyle
        - clothing

        Do not recreate the reference image.

        Narration:
        {scene['narration']}

        Visual Prompt:
        {scene['visual_prompt']}

        Scene Description:
        {scene['description']}

        Lighting:
        {scene['lighting']}

        Camera Angle:
        {scene['camera_angle']}

        Mood:
        {scene['mood']}

        Cinematic storytelling image.

        Wide scene composition.

        Professional movie frame.
        """

        reference_image = None
        reference_source = None

        if reference_image is None and scene_characters:

            first_char = scene_characters[0]

            if first_char in character_cache:
                scene_reference = None

                for idx, char_name in enumerate(scene_characters):

                    char_image = character_cache[char_name]["base64"]

                    if scene_reference is None:

                        scene_reference = char_image

                    else:

                        merge_prompt = f"""
                        Add this character to the existing scene.

                        Character:
                        {char_name}

                        Preserve existing characters.

                        Group composition.

                        Cinematic style.
                        """

                        scene_reference = generate_image_nvidia(
                            prompt=merge_prompt,
                            reference_image=scene_reference,
                            aspect_ratio="16:9",
                            steps=30,
                            cfg_scale=3.5
                        )
                reference_source = f"character:{first_char}"

        print(
            f"Scene {scene['scene_number']} "
            f"using {reference_source}"
        )


        # Handle Generation with Retry Logic
        max_retries = 10
        image_base64 = None
        current_prompt = consistent_prompt

        for attempt in range(max_retries):
            try:
                if reference_image:
                    # Image-to-image generation with reference
                    image_base64 = generate_image_nvidia(
                        prompt=current_prompt,
                        reference_image=scene_reference,
                        aspect_ratio="16:9",
                        steps=30,
                        cfg_scale=3.5
                    )
                else:
                    # Text-to-image generation
                    image_base64 = generate_image_nvidia(
                        prompt=current_prompt,
                        aspect_ratio="16:9",
                        steps=30,
                        cfg_scale=3.5
                    )
                break

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400:
                    # Content filter or validation error
                    print(f"⚠️ Content filter triggered. Rewriting with Claude...{current_prompt}")
                    print(f"Response: {e.response.text}")
                    safe_prompt = rewrite_prompt_with_claude(current_prompt)
                    current_prompt = safe_prompt

                elif e.response.status_code == 429:
                    # Rate limiting
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⏳ Throttled. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)

                else:
                    print(f"❌ HTTP Error {e.response.status_code}: {e.response.text}")
                    raise e
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                raise e

        # 4. Save the generated image
        if image_base64:
            image_bytes = base64.b64decode(image_base64)

            scene_num = scene["scene_number"]
            image_filename = f"outputs/{request_id}/scene_{scene_num}.png"

            s3.put_object(
                Bucket=target_bucket,
                Key=image_filename,
                Body=image_bytes,
                ContentType='image/png'
            )

            scene["image_s3_path"] = f"s3://{target_bucket}/{image_filename}"
            scene["generation_metadata"] = {
                "reference_source": reference_source,
                "characters": scene_characters
            }
            print(f"✅ Scene {scene_num} complete.")

    # 5. Save the FINAL JSON back to S3
    final_json_key = f"outputs/{request_id}/final_scenes.json"
    s3.put_object(
        Bucket=target_bucket,
        Key=final_json_key,
        Body=json.dumps(data, indent=2),
        ContentType='application/json'
    )

    # 6. Output for next Step Function state
    output = {
        "status": "Success",
        "final_json_path": f"s3://{target_bucket}/{final_json_key}",
        "s3_key": s3_key
    }
    print(json.dumps(output))
    print(f"🚀 Task Finished. Final JSON: s3://{target_bucket}/{final_json_key}")


if __name__ == "__main__":
    process_video_generation()