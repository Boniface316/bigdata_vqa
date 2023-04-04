# FROM nvcr.io/nvidia/cuquantum-appliance:22.11
FROM nvcr.io/nvidia/cuda-quantum:0.3.0


USER root

WORKDIR /workspace/

RUN sudo apt-get update -y && \
    sudo apt-get install -y openssh-server sysstat

RUN pip install jupyterthemes && \
    jt -t oceans16 -T -N -kl -cursw 3 -cursc r -cellw 88% -T -N && \
    echo "jupyter notebook --ip='0.0.0.0' --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''" > /workspace/start_jupyter.sh && \
    chmod +x /workspace/start_jupyter.sh 

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" && \
    echo "zsh" >> ~/.bashrc


RUN wget https://github.com/zellij-org/zellij/releases/latest/download/zellij-x86_64-unknown-linux-musl.tar.gz && \
    tar -xvf zellij-x86_64-unknown-linux-musl.tar.gz && \
    chmod +x zellij && \
    rm zellij-x86_64-unknown-linux-musl.tar.gz

RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
    git clone --depth 1 -- https://github.com/marlonrichert/zsh-autocomplete.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autocomplete && \
    git clone https://github.com/zsh-users/zsh-autosuggestions.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zdharma-continuum/fast-syntax-highlighting.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/fast-syntax-highlighting

COPY .zshrc /root/.zshrc

EXPOSE 8888 3000

CMD ["./start_jupyter.sh","--allow-root"]
