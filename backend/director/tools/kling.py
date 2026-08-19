import requests
import time
import jwt

PARAMS_CONFIG = {
    "text_to_video": {
        "model": {
            "type": "string",
            "description": "Model to use for video generation",
            "enum": ["kling-v1"],
            "default": "kling-v1",
        },
        "negative_prompt": {
            "type": "string",
            "description": "Negative text prompt",
            "maxLength": 5200,
        },
        "cfg_scale": {
            "type": "number",
            "description": "Flexibility in video generation. The higher the value, the lower the model's degree of flexibility and the stronger the relevance to the user's prompt",
            "minimum": 0,
            "maximum": 1,
            "default": 0.5,
        },
        "mode": {
            "type": "string",
            "description": "Video generation mode",
            "enum": ["std", "pro"],
            "default": "std",
        },
        "camera_control": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Type of camera movement",
                    "enum": [
                        "simple", "none", "down_back", "forward_up",
                        "right_turn_forward", "left_turn_forward",
                    ],
                    "default": "none",
                },
                "config": {
                    "type": "object",
                    "properties": {
                        "horizontal": {"type": "number", "minimum": -10, "maximum": 10, "default": 0},
                        "vertical": {"type": "number", "minimum": -10, "maximum": 10, "default": 0},
                        "pan": {"type": "number", "minimum": -10, "maximum": 10, "default": 0},
                        "tilt": {"type": "number", "minimum": -10, "maximum": 10, "default": 0},
                        "roll": {"type": "number", "minimum": -10, "maximum": 10, "default": 0},
                        "zoom": {"type": "number", "minimum": -10, "maximum": 10, "default": 0},
                    },
                },
            },
        },
    }
}


class KlingAITool:
    def __init__(self, access_key: str, secret_key: str):
        self.api_route = "https://api.klingai.com"
        self.video_endpoint = f"{self.api_route}/v1/videos/text2video"
        self.access_key = access_key
        self.secret_key = secret_key
        self.polling_interval = 30

    def get_authorization_token(self):
        headers = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iss": self.access_key,
            "exp": int(time.time()) + 1800,
            "nbf": int(time.time()) - 5,
        }
        return jwt.encode(payload, self.secret_key, headers=headers)

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
        """Submit a Kling task and download it when complete."""
        self._heartbeat(on_heartbeat)
        api_key = self.get_authorization_token()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "model": config.get("model", "kling-v1"),
            "duration": duration,
            **config,
        }

        response = requests.post(self.video_endpoint, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception("Kling video submission failed")

        job_id = response.json().get("data", {}).get("task_id")
        if not job_id:
            raise Exception("Kling did not return a task ID")

        if on_request_id is not None:
            on_request_id(str(job_id))
        self._heartbeat(on_heartbeat)
        return self.resume_text_to_video(
            str(job_id),
            save_at,
            on_heartbeat=on_heartbeat,
        )

    def resume_text_to_video(
        self,
        request_id: str,
        save_at: str,
        on_heartbeat=None,
    ):
        """Resume polling/downloading an already submitted Kling task."""
        api_key = self.get_authorization_token()
        result_endpoint = f"{self.api_route}/v1/videos/text2video/{request_id}"
        headers = {"Authorization": f"Bearer {api_key}"}

        while True:
            self._heartbeat(on_heartbeat)
            response = requests.get(result_endpoint, headers=headers)
            response.raise_for_status()
            data = response.json().get("data", {})
            status = data.get("task_status")

            if status == "succeed":
                videos = data.get("task_result", {}).get("videos", [])
                if not videos or not videos[0].get("url"):
                    raise Exception("Kling task completed without a video URL")
                self._heartbeat(on_heartbeat)
                video_response = requests.get(videos[0]["url"])
                video_response.raise_for_status()
                self._heartbeat(on_heartbeat)
                with open(save_at, "wb") as file:
                    file.write(video_response.content)
                return None

            time.sleep(self.polling_interval)
