CFLAGS = `pkg-config opencv --cflags`
LIBS = `pkg-config opencv --libs`

libtensorfilters.so: cov6d.o
	g++ -shared -W -o libtensorfilters.so cov6d.o utils.o $(LIBS)

cov6d.o: cov6d.cpp
	g++ -c -fPIC cov6d.cpp utils.cpp $(CFLAGS) $(LIBS)

clean:
	rm -f cov6d.o libtensorfilters.so cov6d.h.gch utils.o

# g++ -shared -Wl,-soname,libtensorfilters.so -o libtensorfilters.so  cov6d.o `pkg-config opencv --libs`
# g++ -std=c++11 -c -fPIC  -o cov6d.o `pkg-config opencv --cflags` cov6d.cpp `pkg-config opencv --libs`
