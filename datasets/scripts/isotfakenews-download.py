import kagglehub

# Download latest version
path = kagglehub.dataset_download("csmalarkodi/isot-fake-news-dataset", output_dir='./datasets/isotfakenews')

print("Path to dataset files:", path)
