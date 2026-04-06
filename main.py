import os

from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from agents-with-langchain!")
    OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
    ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL")
    GOOGLE_KEY = os.getenv("GOOGLE_KEY")
    # print(f"OPEN_AI_KEY: {OPEN_AI_KEY}")
    # print(f"ANTHROPIC_KEY: {ANTHROPIC_KEY}")
    # print(f"CLAUDE_MODEL: {CLAUDE_MODEL}")
    # print(f"GOOGLE_KEY: {GOOGLE_KEY}")


if __name__ == "__main__":
    main()
