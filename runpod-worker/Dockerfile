FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# Pre-download the HF dataset on build so cold starts are fast
RUN python -c "from datasets import load_dataset; load_dataset('soyeb-jim285/ocr-handwriting-data')"

CMD ["python", "-u", "handler.py"]
