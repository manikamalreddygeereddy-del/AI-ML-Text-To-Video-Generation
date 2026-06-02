import boto3
import json
import base64
import os
import time
import random
from botocore.exceptions import ClientError

# Initialize clients
s3 = boto3.client("s3")
bedrock = boto3.client(service_name="bedrock-runtime",
    region_name="us-west-2")
bedrock_claude = boto3.client(service_name="bedrock-runtime")

def rewrite_prompt_with_claude(original_prompt):
    """Use Claude to sanitize the prompt for Titan Image Generator"""
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
    previous_scene_base64 = None

    # 2. Load the original JSON from S3
    print(f"Fetching input from: {s3_key}")
    response = s3.get_object(Bucket=source_bucket, Key=s3_key)
    data = json.loads(response['Body'].read())
    request_id = data["request_id"]

    character_first_appearance = {}

    for character in data["context"]["characters"]:
        character_first_appearance[character["name"]] = False

    for character in data["context"]["characters"]:
        character_name = character["name"]
        character_description = character["description"]

        character_prompt = f"""
        Character reference sheet.
        {character_name}

        Description:
        {character_description}

        Same face.
        Same clothing.
        Same appearance.
        Character turnaround sheet.
        Clean cinematic illustration.
        White background.
        """

        payload = {
            "prompt": character_prompt,
            "mode": "text-to-image",
            "aspect_ratio": "1:1"
        }

        max_retries = 10
        image_response = None
        for attempt in range(max_retries):
            try:
                response = bedrock.invoke_model(
                    modelId="stability.sd3-5-large-v1:0",
                    body=json.dumps(payload)
                )
                break
            except ClientError as e:
                error_code = e.response["Error"]["Code"]

                if error_code == "ValidationException":
                    print(f"⚠️ Content filter triggered. Rewriting with Claude...{character_prompt}")
                    print(f"{e.response}")
                    safe_prompt = rewrite_prompt_with_claude(character_prompt)
                    payload["prompt"] = safe_prompt
                    # Continues to next attempt loop with updated payload

                elif error_code == "ThrottlingException":
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⏳ Throttled. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)

                else:
                    print(f"❌ Permanent Error: {error_code}")
                    raise e

        body = json.loads(response["body"].read())

        image_base64 = body["images"][0]
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
        Story setting:
        {data['context']['setting']}

        Tone:
        {data['context']['tone']}

        {character_reference_text}

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

        Maintain character consistency.
        Use same face.
        Use same clothing.
        Use same appearance.
        """

        reference_image = None
        reference_source = None

        for char_name in scene_characters:

            if (
                    char_name in character_cache and
                    not character_first_appearance[char_name]
            ):
                reference_image = character_cache[char_name]["base64"]
                reference_source = f"character:{char_name}"

                character_first_appearance[char_name] = True
                break

        if reference_image is None and previous_scene_base64:
            reference_image = previous_scene_base64
            reference_source = "previous_scene"

        if reference_image is None and scene_characters:

            first_char = scene_characters[0]

            if first_char in character_cache:
                reference_image = character_cache[first_char]["base64"]
                reference_source = f"character:{first_char}"

        print(
            f"Scene {scene['scene_number']} "
            f"using {reference_source}"
        )


        if reference_image:
            request_payload = {
                "prompt": consistent_prompt,
                "mode": "image-to-image",
                "image": reference_image,
                "strength": 0.35,
                "output_format": "png"
            }
        else:

            request_payload = {
                "prompt": consistent_prompt,
                "mode": "text-to-image",
                "aspect_ratio": "16:9",
                "output_format": "png"
            }

        # Handle Generation with Retry Logic
        max_retries = 10
        image_response = None

        for attempt in range(max_retries):
            try:
                image_response = bedrock.invoke_model(
                    modelId="stability.sd3-5-large-v1:0",
                    body=json.dumps(request_payload)
                )
                break

            except ClientError as e:
                error_code = e.response["Error"]["Code"]

                if error_code == "ValidationException":
                    print(f"⚠️ Content filter triggered. Rewriting with Claude...{consistent_prompt}")
                    print(f"{e.response}")
                    safe_prompt = rewrite_prompt_with_claude(consistent_prompt)
                    request_payload["prompt"] = safe_prompt
                    # Continues to next attempt loop with updated payload

                elif error_code == "ThrottlingException":
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⏳ Throttled. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)

                else:
                    print(f"❌ Permanent Error: {error_code}")
                    raise e

        # 4. Save the generated image
        if image_response:
            response_body = json.loads(image_response.get("body").read())
            previous_scene_base64 = response_body["images"][0]
            image_bytes = base64.b64decode(response_body.get("images")[0])

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
