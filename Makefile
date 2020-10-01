TAG=pdi-mdp
PWD:=$(shell pwd)
WORK_PATH=/home/jovyan/work

build:
	docker build -t $(TAG) .

run:
	docker run \
		-v "$(PWD):$(WORK_PATH)" \
		-v "$(IMG_DATA_PATH):$(WORK_PATH)/data" \
		-p 8888:8888 $(TAG)
