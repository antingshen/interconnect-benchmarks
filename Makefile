OBJS = main.o helpers.o
EXENAME = main

#for Titan:
#CC = CC
#CCOPTS = -c -O3 -g -I$(MPICH_DIR)/include
#LINK = CC

#For Firebox (fbox.mill):
CC = /opt/openmpi/bin/mpiCC
CCOPTS = -c -O3 -g -I/opt/openmpi/include -msse -msse2 -mavx -mf16c
LINK = /opt/openmpi/bin/mpiCC
LINKOPTS = -I/opt/openmpi/include
 
all : $(EXENAME)

$(EXENAME) : $(OBJS)
	$(LINK) -o $(EXENAME) $(OBJS) $(LINKOPTS)

main.o : main.cpp helpers.h
	$(CC) $(CCOPTS) main.cpp

helpers.o : helpers.cpp helpers.h
	$(CC) $(CCOPTS) helpers.cpp

clean : 
	rm -f *.o $(EXENAME) 2>/dev/null
	rm -f empty

