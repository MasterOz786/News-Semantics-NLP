import kagglehub

# Download latest version
path = kagglehub.dataset_download("doanquanvietnamca/liar-dataset", output_dir="./datasets/liar-dataset")

print("Path to dataset files:", path)