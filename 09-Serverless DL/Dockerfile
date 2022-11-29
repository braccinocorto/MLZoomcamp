FROM svizor42/zoomcamp-dino-dragon-lambda:v2


RUN pip install numpy
RUN pip install pillow 
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl

COPY the_file_for_docker.py . 
ENV MODEL_NAME=dino-vs-dragon-v2.tflite

CMD ["the_file_for_docker.lambda_handler"]