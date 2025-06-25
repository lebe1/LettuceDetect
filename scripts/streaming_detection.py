import httpx
import re
import sys

OLLAMA_URL = "http://localhost:11434/api/generate"

DIGIT_REGEX = re.compile(r"\d")
VALID_CHAR_REGEX = re.compile(r"^[\x20-\x7E\n\r\t]+$")  # Basic printable chars

def extract_digits(text: str) -> set:
    """Returns a set of all digits in the text."""
    return set(DIGIT_REGEX.findall(text))

def validate_token(token: str, allowed_digits: set) -> bool:
    """Check if token contains any new digits not in allowed set."""
    found_digits = set(DIGIT_REGEX.findall(token))
    invalid_digits = found_digits - allowed_digits
    if invalid_digits:
        print(f"\n[!] Invalid digits detected: {', '.join(invalid_digits)}")
        return False
    return True

def validate_output(output_so_far: str) -> bool:
    """Optional: Ensure output is printable ASCII (sanity check)."""
    return VALID_CHAR_REGEX.fullmatch(output_so_far) is not None

def stream_and_validate(prompt: str, model: str = "deepseek-r1:latest"):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    allowed_digits = extract_digits(prompt)
    print(f"[i] Allowed digits from prompt: {sorted(allowed_digits) or 'None'}")

    with httpx.stream("POST", OLLAMA_URL, json=payload, headers=headers, timeout=None) as response:
        if response.status_code != 200:
            print(f"Failed to connect to Ollama. Status code: {response.status_code}")
            return


        output_so_far = ""

        try:
            for line in response.iter_lines():
                if not line.strip():
                    continue
                data = httpx.Response(200, content=line).json()
                token = data.get("response", "")

                # Sanity check: printable
                output_so_far += token
                if not validate_output(output_so_far):
                    print("\n[!] Invalid characters detected. Stopping generation.")
                    break

                # Digit hallucination check
                if not validate_token(token, allowed_digits):
                    print("[!] Token generation stopped due to hallucinated digit.")
                    break

                print(token, end="", flush=True)

        except Exception as e:
            print(f"\n[!] Error during streaming: {e}", file=sys.stderr)

if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    stream_and_validate(user_prompt)
