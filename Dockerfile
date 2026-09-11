FROM public.ecr.aws/lambda/python:3.12

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
 && uv export --frozen --no-dev --no-emit-project --no-hashes -o requirements.txt \
 && pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y uv \
 && rm requirements.txt

COPY README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

CMD ["leadgate.handler.handler"]
