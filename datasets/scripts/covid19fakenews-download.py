import kagglehub

# Download latest version
path = kagglehub.dataset_download("arashnic/covid19-fake-news", output_dir='./datasets/covid19-fakenews')

print("Path to dataset files:", path)
