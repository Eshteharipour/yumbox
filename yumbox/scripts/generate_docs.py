import os
import subprocess


def generate_cli_docs(output_file="docs/cli.md"):
    result = subprocess.run(
        ["metrics-cli", "analyze", "--help"], capture_output=True, text=True
    )
    with open(output_file, "w") as f:
        f.write("# CLI Documentation\n\n")
        f.write("## `metrics-cli analyze`\n\n")
        f.write("```bash\n")
        f.write(result.stdout)
        f.write("```\n")
    print(f"Documentation written to {output_file}")


if __name__ == "__main__":
    generate_cli_docs()
