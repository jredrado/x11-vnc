FROM python:3.9.5-slim-buster

ENV PYTHONUNBUFFERED 1

WORKDIR /app

ADD requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

ADD app.py ./app.py
ADD chainlit.py ./chainlit.py
ADD chainlit.md ./chainlit.md

COPY ./vectorstore/db_faiss ./vectorstore/db_faiss

COPY ./public ./public
COPY .chainlit .chainlit

COPY ./doc ./doc

EXPOSE 80

CMD [ "chainlit", "run", "chainlit.py", "--port","80"]