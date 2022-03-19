FROM python:3.8-slim
RUN useradd --create-home --shell /bin/bash cli_user
WORKDIR /home/cli_user
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chown -R cli_user *
RUN chgrp -R cli_user *
USER cli_user
CMD ["bash"]