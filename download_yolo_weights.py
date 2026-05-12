import requests
import os

def download_yolov12_weights():
    url = "https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12n-cls.pt"
    target = "yolov12n-cls.pt"
    
    if os.path.exists(target):
        print(f"{target} already exists.")
        return
    
    print(f"Downloading {target} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading weights: {e}")

if __name__ == "__main__":
    download_yolov12_weights()
