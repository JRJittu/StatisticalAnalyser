# StatisticalAnalyser

This project is a Streamlit-based statistical analysis tool. It is fully containerized using Docker for easy deployment and reproducibility.

## Project Structure
- Multiple Python files (main entry: `frontend.py`)
- Uses a `.env` file for API keys and secrets (not included in the Docker image)

## Prerequisites
- [Docker](https://www.docker.com/get-started) installed on your system
- `.env` file with required environment variables (API keys, etc.) in the project directory

## Building the Docker Image

Open a terminal in the `StatisticalAnalyser` directory and run:

```sh
docker build -t statistical-analyser .
```

## Running the Application

To run the app and make your `.env` variables available inside the container:

```sh
docker run -it --rm -p 8501:8501 --env-file .env statistical-analyser
```

- The app will be available at [http://localhost:8501](http://localhost:8501)
- Do **not** use `0.0.0.0` in your browser; use `localhost` or `127.0.0.1`

## .env File
- The `.env` file should be in the same directory as your Docker commands.
- It is **ignored** by Docker (see `.dockerignore`), so secrets are not included in the image.
- Example `.env`:
  ```env
  API_KEY=your_api_key_here
  ANOTHER_SECRET=your_secret_here
  ```

## Notes
- All Python files in the directory are copied into the container, so imports between them will work.
- If you add new dependencies, update `requirements.txt` and rebuild the image.

## Troubleshooting
- If you see `URL: http://0.0.0.0:8501` in the logs, open [http://localhost:8501](http://localhost:8501) in your browser.
- If you get errors about missing environment variables, check your `.env` file and that you are using `--env-file .env`.

---

For further help, contact the project maintainer. 