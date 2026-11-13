FROM tensorflow/tensorflow:latest-jupyter

WORKDIR /workspace

# Install additional dependencies
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scikit-learn \
    torch \
    joblib \
    jupyter \
    ipykernel

# Copy the notebook
COPY swat-lstm-autoencoder-model-97.ipynb /workspace/

# Expose Jupyter port
EXPOSE 8888

# Run Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
