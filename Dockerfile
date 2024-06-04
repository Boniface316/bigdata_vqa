
FROM nvcr.io/nvidia/quantum/cuda-quantum:0.7.1

USER root

WORKDIR /workspace/

RUN sudo apt-get update -y && \
    sudo apt-get install -y openssh-server sysstat && \
    sudo apt-get install cmake -y

RUN pip install jupyterthemes && \
    jt -t oceans16 -T -N -kl -cursw 3 -cursc r -cellw 88% -T -N && \
    echo "jupyter notebook --ip='0.0.0.0' --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''" > /workspace/start_jupyter.sh && \
    chmod +x /workspace/start_jupyter.sh 

EXPOSE 8888 3000

CMD ["./start_jupyter.sh","--allow-root"]
