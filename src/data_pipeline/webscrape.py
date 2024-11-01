from bing_image_downloader import downloader
from utils import IMAGE_QUERIES, RAW_IMAGE_DIR, NUM_PER_QUERY

def main(): 

    for query in IMAGE_QUERIES:
        downloader.download(query, limit=NUM_PER_QUERY, output_dir=RAW_IMAGE_DIR, adult_filter_off=True, force_replace=False, timeout=60)

if __name__ == "__main__":
    main()