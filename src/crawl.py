import os
import re
import json
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from langchain_community.document_loaders import RecursiveUrlLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urljoin
from langchain.schema import Document

load_dotenv()


def create_session():
    """Create requests session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.verify = False  # Disable SSL verification
    return session


def bs4_extractor(html: str) -> str:
    """
    Hàm trích xuất và làm sạch nội dung từ HTML
    Args:
        html: Chuỗi HTML cần xử lý
    Returns:
        str: Văn bản đã được làm sạch, loại bỏ các thẻ HTML và khoảng trắng thừa
    """
    soup = BeautifulSoup(html, "html.parser")  # Phân tích cú pháp HTML
    # Xóa khoảng trắng và dòng trống thừa
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def is_downloadable_file(url: str) -> bool:
    """Check if the URL points to a downloadable file (PDF, Word, Excel)."""
    return url.lower().endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx'))


def download_file(url: str, download_dir: str, session: requests.Session):
    """Download the file from the given URL to the specified directory."""
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
    """
    Hàm crawl dữ liệu từ URL với chế độ đệ quy
    Args:
        url_data (str): URL gốc để bắt đầu crawl
    Returns:
        list: Danh sách các Document object, mỗi object chứa nội dung đã được chia nhỏ
              và metadata tương ứng
    """
    try:
        # Custom headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Create session with custom settings
        session = requests.Session()
        session.verify = False
        session.headers.update(headers)
        retries = Retry(total=5, backoff_factor=0.1)
        session.mount('https://', HTTPAdapter(max_retries=retries))

        # Get initial page
        response = session.get(url_data, timeout=30)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text content
        text_content = soup.get_text(separator='\n', strip=True)
        metadata = {'source': url_data,
                    'title': soup.title.string if soup.title else ''}
        doc = Document(page_content=text_content, metadata=metadata)
        docs = [doc]

        # Get links but limit crawling depth
        links = set()
        for a in soup.find_all('a', href=True):
            link = a['href']
            if link.startswith('/'):
                link = urljoin(url_data, link)
            if link.startswith('http') and 'ctu.edu.vn' in link:
                links.add(link)

        # Limit number of links to crawl
        for link in list(links)[:10]:  # Only process first 10 links
            try:
                response = session.get(link, timeout=10)
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
                metadata = {'source': link,
                            'title': soup.title.string if soup.title else ''}
                docs.append(Document(page_content=text, metadata=metadata))
            except Exception as e:
                continue

        print(f'Documents loaded from {url_data}: {len(docs)}')

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=500)
        all_splits = text_splitter.split_documents(docs)
        return all_splits

    except Exception as e:
        print(f"Error crawling {url_data}: {str(e)}")
        return []


def crawl_multiple_urls(urls):
    """Crawl multiple URLs and download files if applicable."""
    all_documents = []
    download_dir = 'data/download'
    downloaded_files = set()  # Set to track downloaded files

    # Tạo thư mục download nếu chưa tồn tại
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    for url in urls:
        print(f"Crawling URL: {url}")
        docs = crawl_web(url)
        all_documents.extend(docs)  # Collect all documents

        # Download files if they are PDF, Word, or Excel
        for doc in docs:
            file_url = doc.metadata.get('source', '')
            if is_downloadable_file(file_url) and file_url not in downloaded_files:
                download_file(file_url, download_dir, create_session())
                # Add to the set after downloading
                downloaded_files.add(file_url)

    return all_documents


def save_data_locally(documents, filename, directory):
    """
    Lưu danh sách documents vào file JSON
    Args:
        documents (list): Danh sách các Document object cần lưu
        filename (str): Tên file JSON (ví dụ: 'data.json')
        directory (str): Đường dẫn thư mục lưu file
    Returns:
        None: Hàm không trả về giá trị, chỉ lưu file và in thông báo
    """
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)  # Tạo đường dẫn đầy đủ

    # Chuyển đổi documents thành định dạng có thể serialize
    data_to_save = [{'page_content': doc.page_content,
                     'metadata': doc.metadata} for doc in documents]
    # Lưu vào file JSON
    with open(file_path, 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print(f'Data saved to {file_path}')  # In thông báo lưu thành công


def main():
    """
    Hàm chính điều khiển luồng chương trình:
    1. Crawl dữ liệu từ nhiều URL
    2. Lưu dữ liệu đã crawl vào file JSON
    3. In kết quả crawl để kiểm tra
    """
    urls = [
        'https://scs.ctu.edu.vn/',
    ]

    # Crawl dữ liệu từ nhiều URL
    data = crawl_multiple_urls(urls)

    # Lưu dữ liệu vào thư mục data
    try:
        save_data_locally(data, 'school_data.json', 'data')
    except Exception as e:
        print(f"Error saving data: {e}")
    print('data: ', data)  # In dữ liệu đã crawl


# Kiểm tra nếu file được chạy trực tiếp
if __name__ == "__main__":
    main()
