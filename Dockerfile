# syntax=docker/dockerfile:1

# Build with (while in Dockerfile folder):
# $ docker build -t csi_images .
# Start container to run stuff detached, with a volume mounted, and then delete itself:
# $ docker run -d --rm -v [HOST_PATH]:[CONTAINER_PATH] csi_images command
# Interactive example:
# $ docker run -it --rm -v /mnt/HDSCA_Development:/mnt/HDSCA_Development -v /mnt/csidata:/mnt/csidata --entrypoint bash csi_images

FROM python:3.12-slim-bookworm

# ARGs are erased after FROM statements, so these need to be here
ARG PACKAGE_NAME=csi_images

WORKDIR /$PACKAGE_NAME

# To avoid odd requests during apt install; also used for headless Python logic
ENV DEBIAN_FRONTEND=noninteractive

# Prepare venv
RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH

# Copy over package and install
COPY $PACKAGE_NAME /$PACKAGE_NAME/$PACKAGE_NAME
COPY examples /$PACKAGE_NAME/examples
COPY tests /$PACKAGE_NAME/tests
COPY pyproject.toml requirements.txt README.md /$PACKAGE_NAME/
RUN pip install .

ENTRYPOINT [ "bash" ]
