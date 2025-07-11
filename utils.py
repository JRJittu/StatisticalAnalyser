import numpy as np

class util_functions:
    def extract_json_from_response(self, response_text):
        if "```json" in response_text:
            return response_text.split("```json")[1].split("```")[0].strip()
        elif "```python" in response_text:
            return response_text.split("```python")[1].split("```")[0].strip()
        elif "```" in response_text:
            return response_text.split("```")[1].split("```")[0].strip()

        # Handle non-backtick formats like starting with 'python\n'
        if response_text.lower().startswith("python\n"):
            return response_text.split("\n", 1)[1].strip()

        if response_text.lower().startswith("json\n"):
            return response_text.split("\n", 1)[1].strip()

        # Fallback: return entire text
        return response_text.strip()

    def convert_to_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj