# This is a Dockerfile that extends an installation of itk and modules
#in the dsarchive/histomicstk:v0.1.3


FROM dsarchive/histomicstk:v0.1.3

MAINTAINER Bilal Salam <bilal.salam@kitware.com>
#install requirements of pygco
RUN cd / && \
    git clone https://github.com/yujiali/pygco.git && \
    cd pygco && \
    make && \
    echo 'export PYTHONPATH=$PYTHONPATH:/pygco' >> ~/.bashrc 


ENV my_plugin_path=$htk_path/../my_plugin
RUN mkdir -p $my_plugin_path
COPY . $my_plugin_path
WORKDIR $my_plugin_path


# use entrypoint provided by HistomicsTK
ENTRYPOINT ["/build/miniconda/bin/python" ,"/HistomicsTK/server/cli_list_entrypoint.py"] 

