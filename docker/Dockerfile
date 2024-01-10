# Use a base image (AlmaLinux 9)
FROM --platform=linux/amd64 almalinux:9
ENV LANG=C.UTF-8

ARG ROOT_BIN=root_v6.30.02.Linux-almalinux9.3-x86_64-gcc11.4.tar.gz

WORKDIR /opt

COPY packages packages

RUN dnf update -q -y \
 && dnf install -y epel-release \
 && dnf install -y $(cat packages)\
 && rm -f packages \
 && curl -O https://root.cern/download/${ROOT_BIN} \
 && tar -xzvf ${ROOT_BIN} \
 && rm -f ${ROOT_BIN} \
 && echo /opt/root/lib >> /etc/ld.so.conf \
 && ldconfig

ENV ROOTSYS /opt/root
ENV PATH $ROOTSYS/bin:$PATH
ENV PYTHONPATH $ROOTSYS/lib:$PYTHONPATH
ENV CLING_STANDARD_PCH none
# ENV GSL_DIR /usr
# ENV FFTW_DIR /usr
# ENV PATH /usr/lib64/qt5/bin:$PATH

# Geant4
ARG GEANT4_V=v11.0.2
RUN wget https://gitlab.cern.ch/geant4/geant4/-/archive/${GEANT4_V}/geant4-${GEANT4_V}.tar.gz \
 && mkdir -p /opt/geant4/src \
 && mkdir -p /opt/geant4/build \
 && mkdir -p /opt/geant4/install \
 && mkdir -p /opt/geant4/data \
 && tar -xzf geant4-${GEANT4_V}.tar.gz -C /opt/geant4/src \
 && cd /opt/geant4/build \
 && cmake -DCMAKE_INSTALL_PREFIX=/opt/geant4/install \
          -DGEANT4_INSTALL_DATA=ON \
          -DGEANT4_INSTALL_DATADIR=/opt/geant4/data \
          -DGEANT4_BUILD_MULTITHREADED=ON \
          -DGEANT4_INSTALL_EXAMPLES=OFF \
          -DGEANT4_USE_SYSTEM_EXPAT=OFF \
          -DBUILD_STATIC_LIBS=ON \
          -DBUILD_SHARED_LIBS=OFF \
	  -DGEANT4_USE_QT=ON \
          ../src/geant4-${GEANT4_V} \
 && make -j 8 \
 && make install \
 && export GEANT4_DIR=/opt/geant4

# ENV GEANT4_DIR /opt/geant4/install/
ENV GSL_DIR=/usr
ENV FFTW_DIR=/usr
ENV PATH /usr/lib64/qt5/bin:$PATH

# G4Beamline
RUN source /opt/geant4/install/bin/geant4.sh \
 && wget https://www.dropbox.com/scl/fi/8i8oqgt8nrit9bz0829rq/G4beamline-3.08-source.tgz?rlkey=jwumoxe21riytghcr7pfy7wli -O G4beamline-3.08-source.tgz \
 && tar -xzf G4beamline-3.08-source.tgz \
 && rm G4beamline-3.08-source.tgz \
 && mkdir G4beamline-3.08 \
 && cd G4beamline-3.08 \
 && cmake ../G4beamline-3.08-source \
 && cmake --build . --config Release --target install

# python dependencies
RUN pip3 install matplotlib pandas tqdm scipy

# set proper access to folders
RUN chmod -R 777 /opt

# setup script
RUN touch /opt/setup.sh \
 && echo 'source /opt/geant4/install/bin/geant4.sh' >> /opt/setup.sh \
 && echo 'source /opt/G4beamline-3.08/bin/g4bl-setup.sh' >> /opt/setup.sh