import sys
import os

print("🔍 正在诊断 LangChain 环境...")
print(f"Python 路径: {sys.executable}")

try:
    import langchain

    print(f"✅ 成功导入 langchain")
    print(f"📂 真实文件位置: {langchain.__file__}")
    print(f"ℹ️ 版本号: {getattr(langchain, '__version__', '未知')}")

    # 检查是否被本地文件覆盖 (Shadowing)
    if "site-packages" not in langchain.__file__:
        print("\n⚠️  【严重警告】发现冲突！")
        print("你的 Python 加载了当前文件夹下的 langchain 文件，而不是官方库！")
        print("👉 请检查文件夹里是不是有一个叫 'langchain.py' 的文件或 'langchain' 文件夹？请立刻改名！")
    else:
        print("✅ 路径正常 (在 site-packages 中)")

    # 尝试导入 retrievers
    try:
        from langchain import retrievers

        print(f"✅ 成功导入 retrievers 模块: {retrievers}")
    except ImportError as e:
        print(f"❌ 导入 retrievers 失败: {e}")
        print("尝试直接导入 EnsembleRetriever...")
        try:
            from langchain.retrievers import EnsembleRetriever

            print("✅ EnsembleRetriever 居然导入成功了？")
        except:
            print("❌ EnsembleRetriever 依然失败。可能是安装包损坏。")

except ImportError:
    print("❌ 根本找不到 langchain 包！请重新安装。")
