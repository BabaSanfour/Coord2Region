import os

# Set the directory to your repository root
repo_dir = "."  # Change this to your repository folder if needed
output_file = "combined_python_code.txt"

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                outfile.write(f"# File: {file_path}\n")
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                outfile.write("\n\n")  # Add spacing between files

print(f"All Python files have been combined into '{output_file}'")
