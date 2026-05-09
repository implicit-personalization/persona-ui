FROM ghcr.io/astral-sh/uv:0.11.5-python3.12-trixie-slim
# Using the uv image

WORKDIR /app

# Setting up dependencies and stuff to be sure compilation is also working
# RUN apt-get update \
#  && apt-get install -y --no-install-recommends build-essential python3-dev \
#  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app.py ./
COPY state.py ./
COPY .streamlit/ ./.streamlit/
COPY tabs/ ./tabs/
COPY utils/ ./utils/

EXPOSE 8501

ENV PATH="/app/.venv/bin:$PATH"

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
