import os
import re
import json
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urljoin
from langchain.schema import Document

load_dotenv()


def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.verify = False
    return session


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def is_downloadable_file(url: str) -> bool:
    return url.lower().endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx'))


def download_file(url: str, download_dir: str, session: requests.Session):
    try:
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        file_path = os.path.join(download_dir, os.path.basename(url))
        if os.path.exists(file_path):
            return

        response = session.get(url, timeout=30, verify=False)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {file_path}")
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")


def crawl_web(url_data):
    try:
        loader = RecursiveUrlLoader(
            urls=[url_data],
            max_depth=4, 
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        docs = loader.load()

        for doc in docs:
            if is_downloadable_file(doc.metadata.get('source', '')):
                download_file(doc.metadata['source'],
                              'data/downloads', create_session())

        print(f'Documents loaded from {url_data}: {len(docs)}')

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=500)
        all_splits = text_splitter.split_documents(docs)
        return all_splits

    except Exception as e:
        print(f"Error crawling {url_data}: {str(e)}")
        return []


def crawl_multiple_urls(urls):
    all_documents = []
    download_dir = 'data/downloads'
    downloaded_files = set()
    for url in urls:
        print(f"Crawling URL with RecursiveUrlLoader: {url}")
        docs = crawl_web(url)
        all_documents.extend(docs)

        for doc in docs:
            file_url = doc.metadata.get('source', '')
            if is_downloadable_file(file_url) and file_url not in downloaded_files:
                download_file(file_url, download_dir, create_session())
                downloaded_files.add(file_url)

    return all_documents


def save_data_locally(documents, filename, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)

    for i, doc in enumerate(documents):
        file = f"raw/{i}_{doc.metadata['title'].replace("/", "_")}"
        data_save = {'page_content': doc.page_content,
                     'metadata': doc.metadata}
        with open(file, 'w', encoding='utf-8') as file:
            json.dump(data_save, file, ensure_ascii=False, indent=4)
        print(f'Data saved to {file_path}')


def main():
    urls = [
        'https://tuyensinh.ctu.edu.vn/',
        'https://tansinhvien.ctu.edu.vn/',
    ]

    data = crawl_multiple_urls(urls)
    save_data_locally(data, 'school_data.json', 'data')


if __name__ == "__main__":
    main()
