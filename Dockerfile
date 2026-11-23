FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .
ENV SCOUT_DB=/app/data/web.db
EXPOSE 8000
CMD ["sh", "-c", "scout web --host 0.0.0.0 --port ${PORT:-8000}"]
