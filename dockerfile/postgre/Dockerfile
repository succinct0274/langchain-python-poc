FROM postgres:15.3

RUN apt update && apt upgrade -y 
RUN apt install -y git build-essential
RUN apt install -y postgresql-server-dev-all
RUN git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
RUN cd pgvector && make && make install
ENV LANG=en_US.utf8
