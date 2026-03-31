import os

from locust import HttpUser, between, task


class PredictUser(HttpUser):
    """
    Load-tests the /predict endpoint.
    """

    wait_time = between(0.1, 0.5)
    image_path = os.getenv("IMAGE_PATH", "AffectNet/happy")

    def _pick_image_path(self) -> str:
        if os.path.isdir(self.image_path):
            for name in os.listdir(self.image_path):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    return os.path.join(self.image_path, name)
        return self.image_path

    @task
    def predict(self):
        fp = self._pick_image_path()
        if not os.path.exists(fp):
            return self.client.post("/predict", json={"error": "image_not_found"})

        with open(fp, "rb") as f:
            files = {"image": (os.path.basename(fp), f, "image/jpeg")}
            data = {"threshold": "0.5"}
            self.client.post("/predict", files=files, data=data)

