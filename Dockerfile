FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app.py ./
COPY state.py ./
COPY tabs/ ./tabs/
COPY utils/ ./utils/

EXPOSE 8501

CMD ["sh", "-c", "uv run streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
