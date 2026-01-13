import os
import shutil
import torch
import gc
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

# ================= 配置区域 =================
DATA_PATH = "./data"  # 只放 PDF 文件
DB_PATH = "./vector_db"  # 数据库路径
EMBEDDING_MODEL = "moka-ai/m3e-base"

# 显存保护：每次写入 1000 条，防止 3060 爆显存
BATCH_SIZE = 1000
# 垃圾过滤：如果一页读出来的字少于 20 个，视为无效页（可能是封面图、目录或扫描页），跳过
MIN_CHAR_LIMIT = 20


def load_pdf_pure(file_path):
    """
    只读取 PDF 中的可复制文本。
    返回: (有效页面列表, 状态描述)
    """
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        valid_pages = []

        for p in pages:
            # 清洗：去除多余空白符
            content = p.page_content.strip()
            # 只有字数超过阈值的页才算有效
            if len(content) > MIN_CHAR_LIMIT:
                valid_pages.append(p)

        if not valid_pages:
            return [], "Skipped (No Text)"

        return valid_pages, "Loaded"
    except Exception as e:
        return [], f"Error: {str(e)}"


def create_vector_db():
    # 1. 环境准备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️  运行设备: {device.upper()}")
    if device == "cuda":
        print(f"   - 显卡: {torch.cuda.get_device_name(0)}")

    # 清理旧库，保证纯净
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    # 2. 加载 PDF
    print("🚀 1. 开始读取 PDF (纯文本模式)...")
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"❌ '{DATA_PATH}' 文件夹不存在。请创建并放入 PDF。")
        return

    all_docs = []
    # 只看 .pdf 文件
    files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith('.pdf')]

    if not files:
        print("⚠️  data 文件夹里没有 PDF 文件。")
        return

    # 遍历文件
    for filename in tqdm(files, desc="解析进度"):
        file_path = os.path.join(DATA_PATH, filename)

        docs, status = load_pdf_pure(file_path)

        # 状态反馈
        if status.startswith("Skipped"):
            print(f"   ⚠️ 跳过 {filename}: 纯图片/扫描件/字数太少")
        elif status.startswith("Error"):
            print(f"   ❌ 错误 {filename}: {status}")

        if docs:
            # 注入源文件名，RAG 必备
            for d in docs:
                d.metadata['source'] = filename
            all_docs.extend(docs)

    print(f"\n📄 有效文本片段: {len(all_docs)}")
    if not all_docs:
        print("❌ 未提取到有效文本，请检查 PDF 是否为文字版。")
        return

    # 3. 切分文本
    print("✂️ 2. 文本切分...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_docs)
    print(f"   - 切分出 {len(chunks)} 个块")

    # 4. 向量化写入 (GPU 加速)
    print(f"🧠 3. 加载 Embedding 模型 ({device})...")

    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    print(f"💾 4. 正在写入向量库 (分批处理 BATCH={BATCH_SIZE})...")

    # 初始化数据库
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # 分批写入，显存无压力
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="写入进度"):
        batch = chunks[i: i + BATCH_SIZE]
        vectordb.add_documents(batch)

    print(f"✅ 知识库构建完成！路径: {DB_PATH}")
    print("   (现在可以去运行 RAG 对话脚本了)")


if __name__ == "__main__":
    create_vector_db()