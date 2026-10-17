"""Print the supported model families and what changes for each of them compared to GPT-2."""
from tiny_llm.families import markdown_table

if __name__ == "__main__":
    print(markdown_table())
