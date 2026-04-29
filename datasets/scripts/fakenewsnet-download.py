import kagglehub

# Download latest version
path = kagglehub.dataset_download("mdepak/fakenewsnet", output_dir="./datasets/fakenewsnet")

print("Path to dataset files:", path)