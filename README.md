# Phishing API Deployment (ONNX Optimized)

This deployment uses an ultra-lightweight ONNX runtime instead of PyTorch, resulting in a much smaller container and faster inference speeds.

## 1. Get the Model
Download the ONNX model files from Google Drive: `https://drive.google.com/file/d/16yGOF0--kL3etijg-UflSbnzxpMT7Wof/view?usp=drive_link`
Extract the contents into a folder named `model` in the root of this project.

The structure must look exactly like this:
├── Dockerfile
├── main.py
├── requirements.txt
└── model/
    ├── config.json
    ├── model.onnx
    ├── tokenizer.json
    └── tokenizer_config.json

## 2. Deploy with Docker
Build the image:
`docker build -t phishing-api-onnx .`

Run the container, linking the .env file:
`docker run -d -p 8000:8000 --env-file .env --name phishing-container phishing-api-onnx`

## 3. Usage
The API will be running at `http://localhost:8000`. 
Visit `http://localhost:8000/docs` to test it via the Swagger UI.
