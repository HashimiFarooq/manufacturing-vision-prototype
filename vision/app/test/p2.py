# 1. Import the library
from inference_sdk import InferenceHTTPClient

# 2. Connect to your workspace
client = InferenceHTTPClient(
  api_url="https://serverless.roboflow.com",
  api_key="buhQlFHUJSznlUF3heBb"
)

# 3. Run your workflow on an image
result = client.run_workflow(
  workspace_name="heitan-kandasamy",
  workflow_id="general-segmentation-api",
  images={
    "image": "bolt_head.jpeg"  # Path to your image file
  },
  parameters={
    "classes": "Screw, Nut, Bolt, Washer, bolt_head"
  },
  use_cache=True  # cache workflow definition for 15 minutes
)

# 4. Get your results
print(result)