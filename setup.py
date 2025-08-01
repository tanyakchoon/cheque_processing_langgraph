from setuptools import setup, find_packages

setup(
    name="cheque_processing_langgraph",
    version="1.2.0",
    author="AI Assistant",
    description="A multi-agent cheque processing system using LangGraph and Google Gemini with a Gradio UI.",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.0.48",
        "langchain-google-genai>=1.0.3",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "scikit-image>=0.21.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
        "gradio>=4.20.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "cheque-processor-cli=cheque_processing_langgraph.__main__:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)