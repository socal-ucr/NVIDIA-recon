all:
	cd suite/vector-addition;	make;
	cd suite/peer-peer;         make;
	cd scripts/memory_size;     make;

clean:
	cd suite/vector-addition;	make clean;
	cd suite/peer-peer;         make clean;
	cd scripts/memory_size;     make clean;
