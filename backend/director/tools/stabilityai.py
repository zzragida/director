import io
import os
import time

import requests
from PIL import Image

PARAMS_CONFIG = {
    "text_to_video": {
        "strength": {
            "type": "number",
            "description": "Image influence on output",
            "minimum": 0,
            "maximum": 1,
        },
        "negative_prompt": {
            "type": "string",
            "description": "Keywords to exclude from output",
        },
        "seed": {
            "type": "integer",
            "description": "Randomness seed for generation",
        },
        "cfg_scale": {
            "type": "number",
            "description": "How strongly video sticks to original image",
            "minimum": 0,
            "maximum": 10,
            "default": 1.8,
        },
        "motion_bucket_id": {
            "type": "integer",
            "description": "Controls motion amount in output video",
            "minimum": 1,
            "maximum": 255,
            "default": 127,
        },
    },
}


class StabilityAITool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.image_endpoint = (
            "https://api.stability.ai/v2beta/stable-image/generate/ultra"
        )
        self.video_endpoint = "https://api.stability.ai/v2beta/image-to-video"
        self.result_endpoint = "https://api.stability.ai/v2beta/image-to-video/result"
        self.polling_interval = 10

    @staticmethod
    def _heartbeat(callback):
        if callback is not None:
            callback()

    def text_to_video(
        self,
        prompt: str,
        save_at: str,
        duration: float,
        config: dict,
        on_request_id=None,
        on_heartbeat=None,
    ):
        """Submit Stability image-to-video work and download it when complete."""
        self._heartbeat(on_heartbeat)
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "accept": "image/*",
        }
        image_payload = {
            "prompt": prompt,
            "output_format": config.get("format", "png"),
            "aspect_ratio": config.get("aspect_ratio", "16:9"),
            "negative_prompt": config.get("negative_prompt", ""),
        }
        image_response = requests.post(
            self.image_endpoint,
            headers=headers,
            files={"none": ""},
            data=image_payload,
        )
        if image_response.status_code != 200:
            raise Exception("Stability image generation failed")

        self._heartbeat(on_heartbeat)
        image = Image.open(io.BytesIO(image_response.content))
        new_width = 1024
        new_height = int(new_width * (576 / 1024))
        scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        temp_image_path = f"{save_at}.temp.png"
        scaled_image.save(temp_image_path)

        try:
            video_headers = {"authorization": f"Bearer {self.api_key}"}
            video_payload = {
                "seed": config.get("seed", 0),
                "cfg_scale": config.get("cfg_scale", 1.8),
                "motion_bucket_id": config.get("motion_bucket_id", 127),
            }
            self._heartbeat(on_heartbeat)
            with open(temp_image_path, "rb") as img_file:
                video_response = requests.post(
                    self.video_endpoint,
                    headers=video_headers,
                    files={"image": img_file},
                    data=video_payload,
                )
            if video_response.status_code != 200:
                raise Exception("Stability video submission failed")

            generation_id = video_response.json().get("id")
            if not generation_id:
                raise Exception("Stability did not return a generation ID")

            if on_request_id is not None:
                on_request_id(str(generation_id))
            self._heartbeat(on_heartbeat)
            return self.resume_text_to_video(
                str(generation_id),
                save_at,
                on_heartbeat=on_heartbeat,
            )
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    def resume_text_to_video(
        self,
        request_id: str,
        save_at: str,
        on_heartbeat=None,
    ):
        """Resume polling/downloading an already submitted Stability job."""
        headers = {
            "accept": "video/*",
            "authorization": f"Bearer {self.api_key}",
        }
        while True:
            self._heartbeat(on_heartbeat)
            result_response = requests.get(
                f"{self.result_endpoint}/{request_id}",
                headers=headers,
            )
            if result_response.status_code == 202:
                time.sleep(self.polling_interval)
                continue
            if result_response.status_code == 200:
                self._heartbeat(on_heartbeat)
                with open(save_at, "wb") as file:
                    file.write(result_response.content)
                return None
            raise Exception("Stability video result retrieval failed")
