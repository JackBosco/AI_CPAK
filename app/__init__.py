import os

# Set the working directory to the project root (one level up from the package folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
os.chdir(project_root)

