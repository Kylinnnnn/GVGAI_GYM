ARG VARIANT="3.8"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# 移除不必要的 yarn 源（避免 apt update 警告）
RUN rm -f /etc/apt/sources.list.d/yarn.list

# 使用官方 Debian HTTPS 源，
RUN sed -i 's@http://deb.debian.org@https://deb.debian.org@g' /etc/apt/sources.list && \
    sed -i 's@http://security.debian.org@https://deb.debian.org@g' /etc/apt/sources.list

# 安装系统依赖：Java、编译工具、Fortran、BLAS/LAPACK 等
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    openjdk-11-jdk \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# 使用系统 Python 创建虚拟环境并通过 pip 安装依赖
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install scipy==1.5.3 gym==0.26.2 numpy==1.24.4 pillow==10.4.0

# 将虚拟环境加入 PATH
ENV PATH="/opt/venv/bin:${PATH}"

# 安装 gym_gvgai 本身（项目根目录会被挂载到 /workspaces/GVGAI_GYM）
# 注意：此时项目尚未挂载，所以我们先不安装，而是在 postCreateCommand 中或容器启动后手动安装。
# 如果希望容器构建时就安装，可以将项目文件复制进去，但为了保持挂载灵活性，这里不复制。
# 我们可以在 postCreateCommand 中执行 pip install -e .，但需要确保项目已挂载。
# 由于项目会自动挂载到 /workspaces/GVGAI_GYM，我们可以在容器启动后手动执行安装。
# 为了自动化，可以在 postCreateCommand 中添加安装命令（见 devcontainer.json）。


# 容器启动时默认命令（保持运行）
CMD ["sleep", "infinity"]